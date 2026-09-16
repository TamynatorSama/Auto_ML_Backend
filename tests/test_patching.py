import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from code_gen_eval.patching import (
    MODE_EDIT,
    MODE_REWRITE,
    EditError,
    apply_edits,
    check_script,
    parse_reply,
    resolve_reply,
    strip_fences,
)

SENTINEL = "===AUTOML_RESULT==="
CONTRACT = {SENTINEL: "the runner reads the scores from the JSON printed after this line"}

SCRIPT = f'''# model: ridge
# attempt: 1
# changes: initial implementation

import json
import os


def build_features(frame):
    frame = frame.copy()
    frame["area"] = frame["area"].clip(lower=1)
    return frame


def main():
    model = Ridge(alpha=1.0)
    print("{SENTINEL}")
    print(json.dumps({{"cv_scores": {{}}}}))


if __name__ == "__main__":
    main()
'''


def block(old: str, new: str) -> str:
    return f"<<<<<<< SEARCH\n{old}=======\n{new}>>>>>>> REPLACE"


# --- reading the reply --------------------------------------------------------

def test_rewrite_plain():
    reply = parse_reply(f"MODE: REWRITE\n{SCRIPT}", allow_edit=True)
    assert reply.mode == MODE_REWRITE
    assert reply.code == SCRIPT.strip()
    assert reply.changes == "initial implementation"


def test_rewrite_with_prose_then_fence():
    raw = f"Here is the script:\n\n```python\n{SCRIPT}```\n\nLet me know."
    reply = parse_reply(raw, allow_edit=False)
    assert reply.mode == MODE_REWRITE
    assert reply.code == SCRIPT.strip()


def test_mode_line_inside_fence_is_found():
    raw = f"```python\nMODE: REWRITE\n{SCRIPT}```"
    reply = parse_reply(raw, allow_edit=False)
    assert reply.mode == MODE_REWRITE
    assert reply.code == SCRIPT.strip()
    assert "```" not in reply.code


def test_mode_line_after_a_sentence_of_prose():
    raw = f"Sure, applying the change.\nMODE: EDIT\n{block('    model = Ridge(alpha=1.0)\n', '    model = Ridge(alpha=10.0)\n')}"
    reply = parse_reply(raw, allow_edit=True)
    assert reply.mode == MODE_EDIT
    assert len(reply.edits) == 1


def test_edit_blocks_each_in_their_own_fence_are_all_kept():
    raw = (
        "MODE: EDIT\n\n```\n"
        + block("import os\n", "import os\nimport numpy as np\n")
        + "\n```\n\nand then\n\n```python\n"
        + block("    model = Ridge(alpha=1.0)\n", "    model = Ridge(alpha=10.0)\n")
        + "\n```\n"
    )
    reply = parse_reply(raw, allow_edit=True)
    assert reply.mode == MODE_EDIT
    assert len(reply.edits) == 2
    code = apply_edits(SCRIPT, reply.edits)
    assert "import numpy as np" in code and "alpha=10.0" in code


def test_fence_inside_a_block_side_is_removed():
    raw = "MODE: EDIT\n" + block(
        "```python\n    model = Ridge(alpha=1.0)\n```\n", "```python\n    model = Ridge(alpha=5.0)\n```\n"
    )
    reply = parse_reply(raw, allow_edit=True)
    assert reply.edits == [("    model = Ridge(alpha=1.0)\n", "    model = Ridge(alpha=5.0)\n")]


def test_lenient_delimiters():
    raw = "mode: edit\n<<<<<<<SEARCH\nimport os\n====\nimport os\nimport sys\n>>>>>>>replace"
    reply = parse_reply(raw, allow_edit=True)
    assert reply.mode == MODE_EDIT
    assert reply.edits == [("import os\n", "import os\nimport sys\n")]


def test_blocks_imply_edit_even_without_mode_line():
    reply = parse_reply(block("import os\n", "import os\nimport sys\n"), allow_edit=True)
    assert reply.mode == MODE_EDIT


def test_edit_without_a_script_is_rejected():
    with pytest.raises(EditError, match="no script to edit"):
        parse_reply("MODE: EDIT\n" + block("a\n", "b\n"), allow_edit=False)


def test_edit_declared_but_no_blocks():
    with pytest.raises(EditError, match="no search/replace block"):
        parse_reply("MODE: EDIT\nimport os\n", allow_edit=True)


def test_malformed_delimiters_are_named():
    raw = "MODE: REWRITE\n<<<<<<<\nimport os\n=======\nimport sys\n>>>>>>>\n"
    with pytest.raises(EditError, match="delimiters are wrong"):
        parse_reply(raw, allow_edit=True)


def test_rst_underline_in_docstring_is_not_a_marker():
    code = '"""\nsummary\n=======\n"""\nimport os\n' + f'print("{SENTINEL}")\n'
    reply = resolve_reply("MODE: REWRITE\n" + code, "", CONTRACT)
    assert reply.mode == MODE_REWRITE


def test_changes_taken_from_replace_side_not_search_side():
    raw = "MODE: EDIT\n" + block(
        "# attempt: 1\n# changes: initial implementation\n",
        "# attempt: 2\n# changes: raised alpha to 10\n",
    )
    reply = parse_reply(raw, allow_edit=True)
    assert reply.changes == "raised alpha to 10"


def test_changes_from_a_stray_line_outside_blocks():
    raw = "MODE: EDIT\n# changes: raised alpha\n" + block("import os\n", "import os\nimport sys\n")
    assert parse_reply(raw, allow_edit=True).changes == "raised alpha"


def test_crlf_reply():
    raw = ("MODE: EDIT\r\n" + block("import os\n", "import os\nimport sys\n")).replace("\n", "\r\n")
    code = apply_edits(SCRIPT.replace("\n", "\r\n"), parse_reply(raw, allow_edit=True).edits)
    assert "import sys" in code and "\r" not in code


# --- applying edits -----------------------------------------------------------

def test_exact_match():
    code = apply_edits(SCRIPT, [("    model = Ridge(alpha=1.0)\n", "    model = Ridge(alpha=10.0)\n")])
    assert "Ridge(alpha=10.0)" in code and "Ridge(alpha=1.0)" not in code


def test_partial_line_anchor():
    code = apply_edits(SCRIPT, [("alpha=1.0", "alpha=0.5")])
    assert "Ridge(alpha=0.5)" in code


def test_deletion_leaves_no_blank_line():
    code = apply_edits(SCRIPT, [("import os\n", "")])
    assert "import os" not in code
    assert "import json\n\n\ndef" in code


def test_trailing_whitespace_is_forgiven():
    dirty = SCRIPT.replace("    model = Ridge(alpha=1.0)\n", "    model = Ridge(alpha=1.0)   \n")
    code = apply_edits(dirty, [("    model = Ridge(alpha=1.0)\n", "    model = Ridge(alpha=2.0)\n")])
    assert "Ridge(alpha=2.0)" in code


def test_uniform_indent_offset_is_forgiven_and_reapplied():
    # the model quoted the body of build_features at column 0
    old = 'frame = frame.copy()\nframe["area"] = frame["area"].clip(lower=1)\n'
    new = 'frame = frame.copy()\nframe["area"] = frame["area"].clip(lower=1)\nframe["log_area"] = np.log1p(frame["area"])\n'
    code = apply_edits(SCRIPT, [(old, new)])
    assert '    frame["log_area"] = np.log1p(frame["area"])\n    return frame' in code


def test_relative_indent_inside_block_must_still_match():
    old = "def build_features(frame):\nframe = frame.copy()\n"   # body not indented relative to def
    with pytest.raises(EditError, match="not found"):
        apply_edits(SCRIPT, [(old, "x\n")])


def test_sequential_blocks_see_earlier_results():
    edits = [
        ("import os\n", "import os\nimport numpy as np\n"),
        ("import numpy as np\n", "import numpy as np\nimport pandas as pd\n"),
    ]
    code = apply_edits(SCRIPT, edits)
    assert "import os\nimport numpy as np\nimport pandas as pd\n" in code


def test_ambiguous_block_is_reported():
    with pytest.raises(EditError, match="matches 2 places"):
        apply_edits(SCRIPT, [("    frame", "    x")])


def test_ambiguous_after_relaxed_match_is_reported():
    twice = "def a():\n    return 1\n\n\ndef b():\n    return 1\n"
    with pytest.raises(EditError, match="matches 2 places"):
        apply_edits(twice, [("return 1\n", "return 2\n")])


def test_not_found_quotes_closest_region():
    with pytest.raises(EditError) as raised:
        apply_edits(SCRIPT, [("    model = Ridge(alpha=1, solver='auto')\n", "x\n")])
    message = str(raised.value)
    assert "not found" in message
    assert "closest text" in message and "model = Ridge(alpha=1.0)" in message


def test_empty_search_is_rejected():
    with pytest.raises(EditError, match="is empty"):
        apply_edits(SCRIPT, [("\n", "import sys\n")])


# --- the contract -------------------------------------------------------------

def test_check_script_reports_syntax_error_with_line():
    problem = check_script("def f(:\n    pass\n")
    assert problem and "does not parse" in problem and "line 1" in problem


def test_check_script_requires_literals():
    problem = check_script("print('hi')\n", CONTRACT)
    assert problem and SENTINEL in problem and "runner reads" in problem
    assert check_script(f"print('{SENTINEL}')\n", CONTRACT) is None


def test_resolve_edit_end_to_end():
    raw = "MODE: EDIT\n" + block("    model = Ridge(alpha=1.0)\n", "    model = Ridge(alpha=10.0)\n")
    reply = resolve_reply(raw, SCRIPT, CONTRACT)
    assert reply.mode == MODE_EDIT
    assert "alpha=10.0" in reply.code and SENTINEL in reply.code


def test_resolve_rejects_edit_that_breaks_syntax():
    raw = "MODE: EDIT\n" + block("    model = Ridge(alpha=1.0)\n", "    model = Ridge(alpha=\n")
    with pytest.raises(EditError, match="does not parse"):
        resolve_reply(raw, SCRIPT, CONTRACT)


def test_resolve_rejects_edit_that_removes_sentinel():
    raw = "MODE: EDIT\n" + block(f'    print("{SENTINEL}")\n', "")
    with pytest.raises(EditError, match="does not contain"):
        resolve_reply(raw, SCRIPT, CONTRACT)


def test_resolve_rejects_header_only_rewrite():
    # the failure seen in runs/1: a "complete script" that was three comment lines
    raw = "MODE: REWRITE\n```python\n# model: mlp\n# attempt: 3\n# changes: smaller network\n```\nThe rest is unchanged."
    with pytest.raises(EditError, match="does not contain"):
        resolve_reply(raw, SCRIPT, CONTRACT)


def test_resolve_rejects_truncated_rewrite():
    raw = "MODE: REWRITE\n" + SCRIPT[: SCRIPT.index("def main")] + "def main(\n"
    with pytest.raises(EditError, match="does not parse"):
        resolve_reply(raw, SCRIPT, CONTRACT)


def test_strip_fences_unpaired_opener():
    assert strip_fences("```python\nimport os\n") == "import os"

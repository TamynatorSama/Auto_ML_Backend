import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from code_gen_eval import code_gen_subgraph as loop
from models import AttemptRecord
from test_values import context
from utils.prompts.code_gen_prompt import code_gen_prompt

MODULE = """# model: ridge
# attempt: 1
# changes: initial implementation
from sklearn.linear_model import Ridge


def build_pipeline(columns, task, ctx):
    return Ridge(alpha=1.0)
"""


class ScriptedLLM:
    """Replies from a fixed list, recording what it was asked."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self.replies.pop(0))


def edit(old: str, new: str) -> str:
    return f"MODE: EDIT\n<<<<<<< SEARCH\n{old}\n=======\n{new}\n>>>>>>> REPLACE"


def test_rewrite_request_answered_with_valid_edits_is_applied(monkeypatch):
    unmatched = edit("alpha=9.0", "alpha=2.0")
    applicable = edit("    return Ridge(alpha=1.0)", "    return Ridge(alpha=2.0)")
    llm = ScriptedLLM([unmatched, unmatched, applicable])
    monkeypatch.setattr(loop, "_llm", lambda *args: llm)

    code, _ = loop._ask_for_code("brief", MODULE)

    assert "Ridge(alpha=2.0)" in code
    assert len(llm.calls) == 3


def test_unusable_replies_keep_the_current_module_instead_of_an_empty_one(monkeypatch):
    unmatched = edit("alpha=9.0", "alpha=2.0")
    llm = ScriptedLLM([unmatched, unmatched, unmatched])
    monkeypatch.setattr(loop, "_llm", lambda *args: llm)

    code, changes = loop._ask_for_code("brief", MODULE)

    assert code == MODULE
    assert "module unchanged" in changes


def test_fixer_with_no_module_regenerates_from_the_brief(monkeypatch):
    llm = ScriptedLLM(["MODE: REWRITE\n" + MODULE])
    monkeypatch.setattr(loop, "_llm", lambda *args: llm)
    failed = AttemptRecord(
        model="ridge", attempt=3, generation=2, script_path="candidate.py", status="error",
        traceback="the generator returned no usable candidate: There is no script to edit yet.",
    )
    state = {
        "context": context, "model": "ridge", "attempts": [failed],
        "current_code": "", "attempt": 3, "repairs": 0, "generation": 2,
    }

    update = loop.fix_code(state)

    system, human = llm.calls[0][0].content, llm.calls[0][1].content
    assert system == code_gen_prompt                       # the generator, not the fixer
    assert "## REQUIRED PREPROCESSING FOR ridge" in human    # it sees the brief
    assert "no usable candidate" in human
    assert "def build_pipeline" in update["current_code"]
    assert update["repairs"] == 1 and update["attempt"] == 4

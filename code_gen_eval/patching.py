"""Search/replace edit format for the code generator.

Each retry answers with either a whole script or a list of search/replace
blocks applied to the script it was given. Blocks are anchored on the text
they change, so there are no line numbers for the model to get wrong.

    resolve_reply(raw, current_code, must_contain) -> Reply   # reply -> validated script, one call
    parse_reply(raw, allow_edit)                  -> Reply   # reply -> mode, edits or code
    apply_edits(base, edits)                      -> str     # edits -> new script
    check_script(code, must_contain)              -> str|None# contract violations, worded for the model

Every failure is an EditError whose message is written for the model: the
caller hands it straight back as the next turn and the model fixes its reply,
so an attempt is only spent on a script that parses and can be scored.
"""

from __future__ import annotations

import ast
import difflib
import re
from typing import List, Mapping, NamedTuple, Optional, Sequence, Tuple

MODE_EDIT = "EDIT"
MODE_REWRITE = "REWRITE"

MODE_SCAN_LINES = 5     # a MODE line may sit this far into the reply; prose before it is dropped
CLOSE_MATCH = 0.6       # difflib ratio above which a near-miss is worth quoting back
HEAD_LINES = 6          # lines of a block quoted in an error message

_MODE_LINE = re.compile(r"^\s*(?:\*\*)?MODE:\s*(EDIT|REWRITE)\b", re.IGNORECASE)
_BLOCK = re.compile(
    r"^<{3,}[ \t]*SEARCH[^\n]*\n(.*?)^={3,}[ \t]*\n(.*?)^>{3,}[ \t]*REPLACE[^\n]*$",
    re.DOTALL | re.MULTILINE | re.IGNORECASE,
)
_CHANGES = re.compile(r"^#\s*changes:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
# a delimiter that is nearly right: wrong count, wrong word, wrong spacing
_ANGLE_MARKER = re.compile(r"^(?:<{3,}|>{3,})[ \t]*(?:SEARCH|REPLACE)?[ \t]*$", re.MULTILINE | re.IGNORECASE)
_FENCED = re.compile(r"^[ \t]*```[^\n]*\n(.*?)^[ \t]*```[ \t]*$", re.DOTALL | re.MULTILINE)
_FENCE_LINE = re.compile(r"^[ \t]*```")


class EditError(Exception):
    """The reply could not be turned into a script."""


class Reply(NamedTuple):
    mode: str
    code: str                       # the script. resolve_reply fills it in for EDIT replies
    edits: List[Tuple[str, str]]
    changes: str


# ----------------------------------------------------------------------------
# reading the reply
# ----------------------------------------------------------------------------

def strip_fences(text: str) -> str:
    """Pull the script out of a reply that wrapped it in markdown.

    A reply that opens with prose and then fences the code used to come back
    with the prose still attached, which the interpreter reported as a syntax
    error on line 1 — the model was blamed for a formatting habit. Take the
    largest fenced block when there is one, wherever it sits.
    """
    text = text.strip()

    blocks = _FENCED.findall(text)
    if blocks:
        return max(blocks, key=len).strip()

    # an unpaired fence: the opener sat above the MODE line and was cut off
    # with it, or the model never closed the block
    lines = text.splitlines()
    if lines and _FENCE_LINE.match(lines[0]):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _split_mode(text: str) -> Tuple[str, str]:
    """The declared mode and everything after it.

    The mode line is looked for in the first few non-blank lines rather than
    only the first: a reply that opens with a fence or a sentence of prose still
    declares its mode, and everything above the declaration is not code.
    """
    lines = text.splitlines()
    seen = 0
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        match = _MODE_LINE.match(line)
        if match:
            return match.group(1).upper(), "\n".join(lines[index + 1 :])
        seen += 1
        if seen >= MODE_SCAN_LINES:
            break
    return "", text


def _unfence_side(text: str) -> str:
    """One side of a block, without a fence the model wrapped around it.

    The captured text ends with the newline before the delimiter line, and
    apply_edits relies on that, so the newline is put back.
    """
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if lines and _FENCE_LINE.match(lines[0]):
        lines.pop(0)
    if lines and _FENCE_LINE.match(lines[-1]):
        lines.pop()
    return "\n".join(lines) + "\n" if lines else ""


def _changes_from_edits(edits: Sequence[Tuple[str, str]], body: str) -> str:
    # the replace side of a block first: a block that rewrites the header
    # carries the OLD changes line on its search side, and taking the first
    # match in the reply reported the previous attempt's changes as this one's
    for _, new in edits:
        match = _CHANGES.search(new)
        if match:
            return match.group(1).strip()
    match = _CHANGES.search(_BLOCK.sub("", body))
    return match.group(1).strip() if match else ""


def parse_reply(raw: str, allow_edit: bool) -> Reply:
    text = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    mode, body = _split_mode(text)

    # blocks are read off the raw body, before any fence handling: their
    # delimiters are line-anchored so fences around them do no harm, whereas
    # keeping only the largest fenced region silently dropped every block the
    # model had put in a fence of its own
    edits = [(_unfence_side(old), _unfence_side(new)) for old, new in _BLOCK.findall(body)]

    if edits or mode == MODE_EDIT:
        if not allow_edit:
            raise EditError(
                "There is no script to edit yet. Reply with MODE: REWRITE and a complete script."
            )
        if not edits:
            raise EditError(
                "MODE: EDIT was declared but no search/replace block was found. "
                "Every block is <<<<<<< SEARCH, then =======, then >>>>>>> REPLACE, "
                "each delimiter on its own line."
            )
        return Reply(MODE_EDIT, "", edits, _changes_from_edits(edits, body))

    code = strip_fences(body)
    if not code.strip():
        raise EditError("MODE: REWRITE was declared but the script was empty.")

    # marker-looking lines in a rewrite mean the reply meant to be an edit but
    # wrote the delimiters wrong. Treating it as a script produces a guaranteed
    # syntax error on line 1 and spends the attempt for nothing.
    if _ANGLE_MARKER.search(code):
        raise EditError(
            "The reply looks like search/replace blocks but the delimiters are wrong, so it "
            "cannot be read as either an edit or a script. Every block uses exactly these "
            "three delimiter lines, each on its own line: '<<<<<<< SEARCH', then '=======', "
            "then '>>>>>>> REPLACE'."
        )

    match = _CHANGES.search(code)
    return Reply(MODE_REWRITE, code, [], match.group(1).strip() if match else "")


# ----------------------------------------------------------------------------
# applying edits
# ----------------------------------------------------------------------------

def _split_block_lines(text: str) -> List[str]:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return lines


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _dedent(lines: Sequence[str]) -> Tuple[List[str], int]:
    """Lines with trailing space and their common indent removed, plus that indent."""
    stripped = [line.rstrip() for line in lines]
    content = [line for line in stripped if line.strip()]
    base = min((_indent(line) for line in content), default=0)
    return [line[base:] if line.strip() else "" for line in stripped], base


def _reindent(lines: Sequence[str], delta: int) -> List[str]:
    out = []
    for line in lines:
        if not line.strip():
            out.append("")
        elif delta >= 0:
            out.append(" " * delta + line)
        else:
            out.append(line[min(-delta, _indent(line)) :])
    return out


def _relaxed_hits(code_lines: Sequence[str], old: str) -> List[Tuple[int, int]]:
    """Windows of code equal to the block after whitespace normalisation.

    Two things are forgiven, and only these: trailing spaces on a line, and a
    uniform indentation offset across the whole block — the model quoting a
    method body at column 0. Relative indentation inside the block still has to
    match line for line, so this is normalisation, not fuzzy matching.

    Returns (start line, indentation delta) per hit.
    """
    target, target_base = _dedent(_split_block_lines(old))
    span = len(target)
    if not span:
        return []

    hits = []
    for start in range(len(code_lines) - span + 1):
        window, window_base = _dedent(code_lines[start : start + span])
        if window == target:
            hits.append((start, window_base - target_base))
    return hits


def _closest(code_lines: Sequence[str], old: str) -> str:
    """The region of the script that looks most like a block that did not match.

    Quoting it back is what lets the next reply copy the real text instead of
    guessing again at what it misremembered.
    """
    target = [line.strip() for line in _split_block_lines(old)]
    span = len(target)
    if not span or span > len(code_lines):
        return ""
    wanted = "\n".join(target)

    best_ratio, best_start = 0.0, -1
    for start in range(len(code_lines) - span + 1):
        window = "\n".join(line.strip() for line in code_lines[start : start + span])
        matcher = difflib.SequenceMatcher(None, window, wanted)
        if matcher.real_quick_ratio() <= best_ratio or matcher.quick_ratio() <= best_ratio:
            continue
        ratio = matcher.ratio()
        if ratio > best_ratio:
            best_ratio, best_start = ratio, start
    if best_ratio < CLOSE_MATCH:
        return ""
    return "\n".join(code_lines[best_start : best_start + span])


def _head(text: str, lines: int = HEAD_LINES) -> str:
    kept = text.strip("\n").splitlines()
    shown = "\n".join(kept[:lines])
    if len(kept) > lines:
        shown += f"\n... ({len(kept) - lines} more lines)"
    return shown


def _candidates(old: str, new: str) -> List[Tuple[str, str]]:
    """The block text, then the same thing without its closing newline.

    The block delimiters sit on their own lines, so the captured SEARCH text
    always ends in a newline. Whole-line edits want that newline — dropping it
    would leave a blank line behind on a deletion. An anchor that is only part
    of a line never has one, so it only matches once the newline comes off.
    """
    variants = [(old, new)]
    if old.endswith("\n"):
        trimmed_new = new[:-1] if new.endswith("\n") else new
        variants.append((old[:-1], trimmed_new))
    return variants


def _ambiguous(index: int, count: int, old: str) -> EditError:
    return EditError(
        f"SEARCH block {index} matches {count} places in current_code. "
        f"Add the lines around it so it matches exactly once:\n{_head(old)}"
    )


def _apply_one(code: str, old: str, new: str, index: int) -> str:
    for candidate_old, candidate_new in _candidates(old, new):
        count = code.count(candidate_old)
        if count == 1:
            return code.replace(candidate_old, candidate_new, 1)
        if count > 1:
            raise _ambiguous(index, count, old)

    code_lines = code.split("\n")
    hits = _relaxed_hits(code_lines, old)
    if len(hits) > 1:
        raise _ambiguous(index, len(hits), old)
    if len(hits) == 1:
        start, delta = hits[0]
        span = len(_split_block_lines(old))
        replacement = _reindent(_split_block_lines(new), delta)
        return "\n".join(code_lines[:start] + replacement + code_lines[start + span :])

    message = (
        f"SEARCH block {index} was not found in current_code. "
        f"Copy the lines exactly as they appear there:\n{_head(old)}"
    )
    nearest = _closest(code_lines, old)
    if nearest:
        message += f"\n\nThe closest text in current_code is:\n{nearest}"
    raise EditError(message)


def apply_edits(base: str, edits: Sequence[Tuple[str, str]]) -> str:
    """Apply blocks in order. Each one sees the result of the ones before it."""
    code = base.replace("\r\n", "\n")

    for index, (raw_old, raw_new) in enumerate(edits, start=1):
        old = raw_old.replace("\r\n", "\n")
        new = raw_new.replace("\r\n", "\n")

        if not old.strip():
            raise EditError(
                f"SEARCH block {index} is empty. Every block needs text to anchor on: to insert "
                f"new code, search for the line it goes after and repeat that line in REPLACE."
            )
        code = _apply_one(code, old, new, index)

    return code


# ----------------------------------------------------------------------------
# the script contract
# ----------------------------------------------------------------------------

def check_script(code: str, must_contain: Mapping[str, str] | None = None) -> Optional[str]:
    """Why this script cannot be run, or None.

    `must_contain` maps a literal the script has to include to the reason it
    is required, e.g. the result sentinel the runner reads scores after. Both
    checks are instant, and each one caught here is an attempt not spent on a
    script that was never going to score.
    """
    try:
        ast.parse(code)
    except SyntaxError as error:
        where = f"line {error.lineno}" + (f", column {error.offset}" if error.offset else "")
        offending = (error.text or "").rstrip()
        detail = f"\n    {offending}" if offending else ""
        return f"the script does not parse: {error.msg} ({where}){detail}"

    for literal, reason in (must_contain or {}).items():
        if literal not in code:
            return f"the script does not contain {literal!r}: {reason}"
    return None


def resolve_reply(
    raw: str, current_code: str, must_contain: Mapping[str, str] | None = None
) -> Reply:
    """Reply text to a validated script, whichever mode the reply used.

    The returned Reply always carries the full script in `code`. Anything that
    stops the script from being runnable is raised as an EditError worded for
    the model, so one corrective turn repairs it.
    """
    reply = parse_reply(raw, allow_edit=bool(current_code))
    code = apply_edits(current_code, reply.edits) if reply.mode == MODE_EDIT else reply.code

    problem = check_script(code, must_contain)
    if problem is None:
        return reply._replace(code=code)

    if reply.mode == MODE_EDIT:
        raise EditError(
            f"After applying the search/replace blocks, {problem}\n\n"
            "The blocks were not kept. Reply with the full corrected set of blocks against "
            "current_code as it was given, or with MODE: REWRITE and a complete script."
        )
    raise EditError(f"{problem}\n\nReply with MODE: REWRITE and the complete script.")


def format_edits(edits: Sequence[Tuple[str, str]]) -> str:
    return "\n".join(
        f"<<<<<<< SEARCH\n{old.rstrip()}\n=======\n{new.rstrip()}\n>>>>>>> REPLACE"
        for old, new in edits
    )

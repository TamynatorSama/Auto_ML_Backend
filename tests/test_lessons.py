import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from code_gen_eval import code_gen_subgraph as loop
from code_gen_eval.runner import preflight
from models import AttemptRecord
from utils.reusable.lessons import error_signature, got_past, read_lessons, record_lesson, render_lessons

IMPORT_ERROR = (
    "[stage: import]\nTraceback (most recent call last):\n"
    '  File "D:\\\\runs\\\\x\\\\mlp\\\\attempt_{n}\\\\candidate.py", line {line}, in <module>\n'
    "ImportError: cannot import name 'SimpleImputer' from 'sklearn.preprocessing' "
    "(D:\\\\code\\\\.venv\\\\Lib\\\\site-packages\\\\sklearn\\\\preprocessing\\\\__init__.py)"
)


def _failed(attempt, traceback, generation=1, kind="generate", status="error", model="mlp"):
    return AttemptRecord(model=model, attempt=attempt, generation=generation, kind=kind,
                         script_path=f"attempt_{attempt}/candidate.py", status=status, traceback=traceback)


def test_signature_ignores_paths_and_numbers_but_keeps_names():
    first = _failed(1, IMPORT_ERROR.format(n=1, line=6))
    second = _failed(4, IMPORT_ERROR.format(n=4, line=11))

    assert error_signature(first) == error_signature(second)
    assert error_signature(first) == (
        "import: ImportError: cannot import name 'SimpleImputer' from 'sklearn.preprocessing'"
    )
    other = _failed(2, "[stage: smoke]\nValueError: could not convert string to float: 'north'")
    assert error_signature(other) != error_signature(first)
    assert error_signature(AttemptRecord(model="m", attempt=1, script_path="x", status="ok")) is None


def test_a_repair_counts_only_when_it_clears_the_failed_stage():
    import_failure = _failed(1, "[stage: import]\nImportError: x")
    smoke_failure = _failed(2, "[stage: smoke]\nValueError: y", kind="repair")
    scored = AttemptRecord(model="m", attempt=3, kind="repair", script_path="x", status="ok")

    assert got_past(import_failure, smoke_failure)
    assert not got_past(smoke_failure, import_failure)
    assert got_past(smoke_failure, scored)


def test_lessons_are_recorded_once_and_rendered_for_the_prompts(tmp_path):
    failed = _failed(1, IMPORT_ERROR.format(n=1, line=6))
    fixed = AttemptRecord(model="mlp", attempt=2, kind="repair", script_path="x", status="ok",
                          changes="repair: SimpleImputer is imported from sklearn.impute")

    assert record_lesson(tmp_path, "mlp", failed, fixed)["fix"] == "SimpleImputer is imported from sklearn.impute"
    assert record_lesson(tmp_path, "knn", _failed(7, IMPORT_ERROR.format(n=7, line=3), model="knn"), fixed) is None
    assert len(read_lessons(tmp_path)) == 1

    rendered = render_lessons(read_lessons(tmp_path))
    assert "KNOWN PITFALLS" in rendered and "sklearn.impute" in rendered


def test_preflight_names_where_a_misplaced_import_lives():
    from tests.test_leakage import _context

    context = _context(".", environment={"sklearn": "1.9.1", "pandas": "3.0.5", "numpy": "2.5.2"})
    code = "from sklearn.preprocessing import OneHotEncoder, SimpleImputer\n\ndef build_pipeline(columns, task, ctx):\n    return None\n"

    problem = preflight(code, context)
    assert problem == "cannot import name 'SimpleImputer' from 'sklearn.preprocessing'; it is in sklearn.impute"
    fixed = code.replace("from sklearn.preprocessing import OneHotEncoder, SimpleImputer",
                         "from sklearn.impute import SimpleImputer\nfrom sklearn.preprocessing import OneHotEncoder")
    assert preflight(fixed, context) is None


def test_a_repair_that_reproduces_its_error_is_not_repaired_again(tmp_path):
    from tests.test_leakage import _context

    context = _context(tmp_path)
    first = _failed(1, "[stage: smoke]\nInvalidIndexError: (slice(None, None, None), 0)")
    same = _failed(2, "[stage: smoke]\nInvalidIndexError: (slice(None, None, None), 0)", kind="repair")
    different = _failed(2, "[stage: cv]\nValueError: something else", kind="repair")

    assert loop.route_after_run({"context": context, "attempts": [first], "repairs": 0}) == "fix_code"
    assert loop.route_after_run({"context": context, "attempts": [first, same], "repairs": 1}) == "judge"
    assert loop.route_after_run({"context": context, "attempts": [first, different], "repairs": 1}) == "fix_code"


def test_the_execution_cap_stops_repairs_and_the_model(tmp_path):
    from tests.test_leakage import _context

    context = _context(tmp_path, max_tries=2)          # cap = 5
    history = [_failed(n, f"[stage: build]\nNameError: name 'v{n}' is not defined", generation=(n + 1) // 2,
                       kind="generate" if n % 2 else "repair") for n in range(1, 6)]

    assert loop.execution_cap(context) == 5
    assert loop.route_after_run({"context": context, "attempts": history, "repairs": 0}) == "judge"
    decision = loop.decide({"context": context, "model": "knn", "attempts": history, "generation": 2})
    assert decision["status"] == "execution_cap"


def test_run_code_records_the_lesson_when_a_repair_clears_the_stage(tmp_path):
    from tests.test_leakage import DYNAMIC, _leaky_run

    context = _leaky_run(tmp_path, target="noisy")
    code = DYNAMIC.replace("TransformedTargetRegressor(regressor=model, func=np.log, inverse_func=np.exp)", "model")
    failed = _failed(1, "[stage: import]\nImportError: cannot import name 'Thing' from 'sklearn.compose'")
    state = {"context": context, "model": "linear_regression", "attempts": [failed], "attempt": 2,
             "generation": 1, "repairs": 1, "current_code": code,
             "changes": "repair: import Thing from the module that has it"}

    update = loop.run_code(state)

    assert update["attempts"][0].status == "ok"
    lessons = read_lessons(context.run_dir)
    assert len(lessons) == 1 and lessons[0]["stage"] == "import"
    assert "import Thing from the module that has it" in lessons[0]["fix"]


def test_a_new_generation_that_clears_the_failed_stage_also_teaches(tmp_path):
    from tests.test_leakage import DYNAMIC, _leaky_run

    context = _leaky_run(tmp_path, target="noisy")
    code = DYNAMIC.replace("TransformedTargetRegressor(regressor=model, func=np.log, inverse_func=np.exp)", "model")
    failed = _failed(3, "[stage: smoke]\nAttributeError: type object 'LGBMClassifier' has no attribute 'early_stopping'",
                     kind="repair")
    state = {"context": context, "model": "lightgbm", "attempts": [failed], "attempt": 4,
             "generation": 2, "repairs": 0, "current_code": code,
             "changes": "use early_stopping_round through fit_params instead of a class attribute"}

    loop.run_code(state)

    lessons = read_lessons(context.run_dir)
    assert len(lessons) == 1 and lessons[0]["stage"] == "smoke"

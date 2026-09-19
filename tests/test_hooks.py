import sys
import time
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from code_gen_eval import base, runner
from code_gen_eval import code_gen_subgraph as loop
from test_values import context as fixture_context
from utils.reusable import hooks
from utils.reusable.hooks import RunStopped


class Recorder(hooks.Hooks):
    def __init__(self):
        super().__init__()
        self.events = []

    def emit(self, run_id, kind, **data):
        self.events.append((run_id, kind, data))

    def kinds(self):
        return [kind for _, kind, _ in self.events]


@pytest.fixture
def recorder(monkeypatch):
    installed = Recorder()
    monkeypatch.setattr(hooks, "_hooks", installed)
    return installed


@pytest.fixture
def context(tmp_path):
    return fixture_context.model_copy(update={"run_id": 77, "run_dir": str(tmp_path / "run")})


def reply(input_tokens, output_tokens) -> LLMResult:
    message = AIMessage(content="ok", usage_metadata={
        "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens,
    })
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_tokens_and_spend_are_counted_per_run(recorder):
    counter = hooks._CountTokens(recorder, 77, "judge", "gemini-2.5-flash")
    counter.on_llm_end(reply(1000, 2000))
    counter.on_llm_end(reply(1000, 0))

    price_in, price_out = hooks.PRICES["gemini-2.5-flash"]
    expected = (2000 * price_in + 2000 * price_out) / 1e6
    assert hooks.spend(77) == {"input_tokens": 2000, "output_tokens": 2000, "cost_usd": round(expected, 6)}
    assert hooks.spend(78)["input_tokens"] == 0
    assert recorder.kinds() == ["llm_usage", "llm_usage"]
    assert recorder.events[0][2]["role"] == "judge"


def test_budget_deadline_and_a_request_stop_the_run(recorder):
    assert hooks.should_stop(1) is None
    hooks.set_limits(1, budget_usd=0.001)
    recorder.record_usage(1, "generator", "gemini-2.5-flash", 10_000, 0)
    assert hooks.should_stop(1) == "budget"

    hooks.set_limits(2, deadline=time.time() - 1)
    assert hooks.should_stop(2) == "deadline"

    hooks.request_stop(3)
    with pytest.raises(RunStopped, match="stop requested"):
        hooks.check_stop(3)


def test_each_role_gets_its_temperature_and_a_token_counter(recorder, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    planner, judge = hooks.llm_for(5, "planner"), hooks.llm_for(5, "judge")
    assert (planner.temperature, judge.temperature) == (0.0, 0.2)
    assert any(isinstance(callback, hooks._CountTokens) for callback in judge.callbacks)


def test_a_stopped_model_saves_no_result(recorder, context):
    hooks.request_stop(context.run_id, "stop requested")
    update = base.code_gen_worker({"context": context, "model": "ridge"})

    (result,) = update["completed_sections"]
    assert result.status == "stopped" and "stop requested" in result.error
    assert not (Path(context.run_dir) / "ridge" / "result.json").exists()
    assert recorder.kinds() == ["model_waiting", "model_finished"]


def test_no_candidate_runs_once_the_run_is_stopped(recorder, context):
    hooks.request_stop(context.run_id)
    with pytest.raises(RunStopped):
        runner.run_candidate("x = 1", "ridge", 1, context)
    assert not Path(context.run_dir).exists()


def test_the_loop_stops_between_generations(recorder, context):
    hooks.set_limits(context.run_id, deadline=time.time() - 1)
    with pytest.raises(RunStopped, match="deadline"):
        loop.decide({"context": context, "model": "ridge", "attempts": []})


def test_an_attempt_without_a_script_is_still_an_event(recorder, context):
    state = {"context": context, "model": "ridge", "attempt": 1, "generation": 1, "current_code": "", "changes": "no reply"}
    loop.run_code(state)
    (event,) = [data for _, kind, data in recorder.events if kind == "attempt_finished"]
    assert event["model"] == "ridge" and event["status"] == "error"

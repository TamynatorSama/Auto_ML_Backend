"""
hooks.py
--------
What the pipeline asks of whoever runs it, one run at a time. Every call takes
the run id, so one process can serve many runs, and nothing relies on context
variables surviving LangGraph's thread pool.

    emit(run_id, kind, **data)          an event for the live view
    should_stop(run_id) -> str | None   "stop requested" | "budget" | "deadline"
    check_stop(run_id)                  raises RunStopped when should_stop answers
    llm_for(run_id, role)               role: planner | generator | fixer | judge
    sandbox_for(run_id)                 the run's sandbox host
    sandbox_slot(run_id, memory_mb, cpus)  held while one sandbox exists: the worker
                                        queues it fairly with other jobs' sandboxes
    set_limits(run_id, budget_usd=None, deadline=None)
    request_stop(run_id, reason="stop requested")
    spend(run_id) -> {"input_tokens", "output_tokens", "cost_usd"}
    install(hooks)

The defaults serve the command line: events are not recorded (the pipeline
prints them), Gemini is built from GOOGLE_API_KEY, and the sandbox host comes
from AUTOML_SANDBOX_URL and AUTOML_SANDBOX_TOKEN. The worker subclasses Hooks.

The model key never goes into graph state, RunContext or a sandbox: the
checkpointer and context.json would store it, and generated code would see it.
"""

from __future__ import annotations

import threading
import time
from contextlib import nullcontext
from functools import lru_cache
from typing import Dict, Optional

from langchain_core.callbacks import BaseCallbackHandler

MODEL = "gemini-2.5-flash"
TEMPERATURE = {"planner": 0.0, "generator": 0.2, "fixer": 0.2, "judge": 0.2}
# USD per million tokens, (input, output); check https://ai.google.dev/pricing
PRICES = {"gemini-2.5-flash": (0.30, 2.50)}


class RunStopped(Exception):
    """The run was stopped between attempts: by a person, its budget or its deadline."""


class _Run:
    def __init__(self):
        self.budget_usd: Optional[float] = None
        self.deadline: Optional[float] = None
        self.stop: Optional[str] = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0


class _CountTokens(BaseCallbackHandler):
    def __init__(self, hooks: "Hooks", run_id, role: str, model: str):
        self.hooks, self.run_id, self.role, self.model = hooks, run_id, role, model

    def on_llm_end(self, response, **kwargs) -> None:
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                self.hooks.record_usage(
                    self.run_id, self.role, self.model, usage.get("input_tokens", 0), usage.get("output_tokens", 0)
                )


class Hooks:
    def __init__(self):
        self._lock = threading.Lock()
        self._runs: Dict[object, _Run] = {}

    def _run(self, run_id) -> _Run:
        with self._lock:
            return self._runs.setdefault(run_id, _Run())

    def emit(self, run_id, kind: str, **data) -> None:
        pass

    def should_stop(self, run_id) -> Optional[str]:
        run = self._run(run_id)
        if run.stop:
            return run.stop
        if run.budget_usd is not None and run.cost_usd >= run.budget_usd:
            return "budget"
        if run.deadline is not None and time.time() >= run.deadline:
            return "deadline"
        return None

    def llm_for(self, run_id, role: str):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=MODEL, temperature=TEMPERATURE[role], callbacks=[_CountTokens(self, run_id, role, MODEL)]
        )

    def sandbox_for(self, run_id):
        # one host for every run: AUTOML_SANDBOX_URL and AUTOML_SANDBOX_TOKEN
        return _client_from_env()

    def sandbox_slot(self, run_id, memory_mb: float, cpus: float):
        # one run at a time on the command line: the host's own 429 is limit enough
        return nullcontext()

    def record_usage(self, run_id, role: str, model: str, input_tokens: int, output_tokens: int) -> None:
        price_in, price_out = PRICES.get(model, (0.0, 0.0))
        cost = (input_tokens * price_in + output_tokens * price_out) / 1e6
        run = self._run(run_id)
        with self._lock:
            run.input_tokens += input_tokens
            run.output_tokens += output_tokens
            run.cost_usd += cost
        self.emit(
            run_id, "llm_usage", role=role, model=model,
            input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=round(cost, 6),
        )

    def set_limits(self, run_id, budget_usd: Optional[float] = None, deadline: Optional[float] = None) -> None:
        run = self._run(run_id)
        run.budget_usd, run.deadline = budget_usd, deadline

    def request_stop(self, run_id, reason: str = "stop requested") -> None:
        self._run(run_id).stop = reason

    def spend(self, run_id) -> dict:
        run = self._run(run_id)
        return {"input_tokens": run.input_tokens, "output_tokens": run.output_tokens, "cost_usd": round(run.cost_usd, 6)}


@lru_cache(maxsize=1)
def _client_from_env():
    from code_gen_eval.sandbox_client import SandboxClient

    return SandboxClient.from_env()


_hooks = Hooks()


def install(hooks: Hooks) -> None:
    global _hooks
    _hooks = hooks


def emit(run_id, kind: str, **data) -> None:
    _hooks.emit(run_id, kind, **data)


def should_stop(run_id) -> Optional[str]:
    return _hooks.should_stop(run_id)


def check_stop(run_id) -> None:
    reason = _hooks.should_stop(run_id)
    if reason:
        raise RunStopped(reason)


def llm_for(run_id, role: str):
    return _hooks.llm_for(run_id, role)


def sandbox_for(run_id):
    return _hooks.sandbox_for(run_id)


def sandbox_slot(run_id, memory_mb: float, cpus: float):
    return _hooks.sandbox_slot(run_id, memory_mb, cpus)


def set_limits(run_id, budget_usd: Optional[float] = None, deadline: Optional[float] = None) -> None:
    _hooks.set_limits(run_id, budget_usd, deadline)


def request_stop(run_id, reason: str = "stop requested") -> None:
    _hooks.request_stop(run_id, reason)


def spend(run_id) -> dict:
    return _hooks.spend(run_id)

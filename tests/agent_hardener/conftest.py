# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures: a ``Mission`` built through its constructor with recording fakes, and relay-plugin stubs."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent_hardener.openshell.lifecycle import OpenShellCommandResult
from agent_hardener.runtime import runner
from agent_hardener.runtime.runner import Mission, MissionResult


def command_result(output: str = "", *, ok: bool = True) -> OpenShellCommandResult:
    return OpenShellCommandResult(ok=ok, command=[], output=output)


@dataclass(frozen=True)
class FakeOpenShellConfig:
    """The ``OpenShellConfig`` fields the mission reads; a dataclass so ``replace`` works as in production."""

    relay_victim: Any
    cwd: Path
    sandbox: str
    start_command: str
    build_context: Path | None
    image: str | None
    policy_path: Path | None
    health_timeout: float = 5.0


class FakeLifecycle:
    """Stands in for ``OpenShellLifecycle``; records what the mission asks of the sandbox."""

    def __init__(self, config: Any, **_kwargs: Any) -> None:
        self.config = config
        self.calls: list[str] = []
        self.up_result = command_result("built")
        self.diagnostics = command_result()
        self.fetch_error: Exception | None = None
        self.fetched: list[tuple[str, Path]] = []

    def down(self) -> OpenShellCommandResult:
        self.calls.append("down")
        return command_result("sandbox down")

    def up(self, _policy: object) -> OpenShellCommandResult:
        self.calls.append("up")
        return self.up_result

    def collect_victim_diagnostics(self) -> OpenShellCommandResult:
        return self.diagnostics

    def fetch_file(self, remote: str, local: Path) -> None:
        if self.fetch_error:
            raise self.fetch_error
        self.fetched.append((remote, local))


class FakeBackends:
    def __init__(self) -> None:
        self.up_result = command_result("compose output")
        self.downed: list[tuple[list[Any], Path]] = []

    def up(self, _specs: list[Any], _cwd: Path) -> OpenShellCommandResult:
        return self.up_result

    def down(self, specs: list[Any], cwd: Path) -> OpenShellCommandResult:
        self.downed.append((specs, cwd))
        return command_result("backends down")


@dataclass
class MissionHarness:
    """A constructed mission plus everything the fakes observed while it ran."""

    mission: Mission
    session: Any
    base_config: Any
    lifecycle: FakeLifecycle
    backends: FakeBackends
    events: list[tuple[str, dict[str, Any]]]
    orchestrator_kwargs: list[dict[str, Any]]
    health_waits: list[tuple[str, float]]
    run_dir: Path
    orchestrator_hook: Callable[[dict[str, Any]], None] | None = None
    reports: list[Any] = field(default_factory=list)

    def run(self) -> MissionResult:
        return self.mission.run()

    @property
    def lines(self) -> list[str]:
        """Highlight lines the run announced."""
        return [payload["line"] for event, payload in self.events if event == "output"]

    @property
    def statuses(self) -> list[str]:
        return [payload["label"] for event, payload in self.events if event == "status_started"]

    @property
    def run_log(self) -> str:
        path = self.run_dir / "agent-hardener.log"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def read_artifact(self, name: str) -> Any:
        return json.loads((self.run_dir / name).read_text(encoding="utf-8"))


BuildMission = Callable[..., MissionHarness]


@pytest.fixture
def build_mission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BuildMission:
    """Factory for a ``MissionHarness``.

    ``Mission(...)`` runs its real ``_prepare``; only the objects it reaches out to are swapped. The
    orchestrator is a fake that returns ``harness.reports`` and, when set, first calls
    ``harness.orchestrator_hook(kwargs)`` with the callbacks the mission handed it.
    """

    def build(
        *,
        base_url: str = "http://victim/v1",
        atof_path: Path | None = None,
        backends: tuple[str, ...] = (),
        build_context: Path | None = None,
        image: str | None = None,
        start_command: str = "run",
        policy_path: Path | None = None,
        skip_initial_up: bool = False,
        no_cleanup: bool = False,
        health_url: str | None = None,
        run_id: str | None = "run-1",
    ) -> MissionHarness:
        relay_victim = (
            SimpleNamespace(atof_path=atof_path or Path("events.atof.jsonl"), backends=list(backends))
            if atof_path or backends
            else None
        )
        base_config = FakeOpenShellConfig(
            relay_victim=relay_victim,
            cwd=tmp_path,
            sandbox="sb",
            start_command=start_command,
            build_context=build_context,
            image=image,
            policy_path=policy_path,
        )
        session = SimpleNamespace(
            victim_control="vc",
            victim="victim",
            target=SimpleNamespace(base_url=base_url, agent_relay_plugins=None, name="agent"),
            benign_validators=[],
            benign_suite_path=None,
            storage=SimpleNamespace(
                root_dir=tmp_path, run_id=run_id, victim_policy_path=None, victim_relay_plugins_path=None
            ),
        )
        fake_backends = FakeBackends()
        lifecycles: list[FakeLifecycle] = []
        events: list[tuple[str, dict[str, Any]]] = []
        health_waits: list[tuple[str, float]] = []
        orchestrator_kwargs: list[dict[str, Any]] = []
        holder: dict[str, MissionHarness] = {}

        def make_lifecycle(config: Any, **kwargs: Any) -> FakeLifecycle:
            lifecycles.append(FakeLifecycle(config, **kwargs))
            return lifecycles[-1]

        class FakeOrchestrator:
            def __init__(self, _config: object, **kwargs: Any) -> None:
                orchestrator_kwargs.append(kwargs)
                self.kwargs = kwargs

            async def run_rounds(self, **_kwargs: Any) -> list[Any]:
                orchestrator_kwargs[-1]["run_rounds"] = _kwargs
                hook = holder["harness"].orchestrator_hook
                if hook is not None:
                    hook(self.kwargs)
                return holder["harness"].reports

        monkeypatch.setattr(runner, "openshell_config", lambda _vc: base_config)
        monkeypatch.setattr(runner, "prepare_relay_victim", lambda cfg: cfg)
        monkeypatch.setattr(runner, "OpenShellLifecycle", make_lifecycle)
        monkeypatch.setattr(
            runner, "build_uploaders", lambda _vc, lifecycle: (("shell", lifecycle), ("relay", lifecycle))
        )
        monkeypatch.setattr(runner, "BackendManager", lambda: fake_backends)
        monkeypatch.setattr(runner, "target_health_url", lambda _url: health_url)
        monkeypatch.setattr(runner, "wait_for_health", lambda url, timeout: health_waits.append((url, timeout)))
        monkeypatch.setattr(runner, "Orchestrator", FakeOrchestrator)
        monkeypatch.setattr(runner, "print_final_summary", lambda _reports, **_kw: None)

        spec = runner.MissionSpec(
            session_config=cast("Any", session),
            skip_initial_up=skip_initial_up,
            no_cleanup=no_cleanup,
            on_event=lambda event, payload: events.append((event, payload)),
        )
        mission = Mission(spec)
        harness = MissionHarness(
            mission=mission,
            session=session,
            base_config=base_config,
            lifecycle=lifecycles[0],
            backends=fake_backends,
            events=events,
            orchestrator_kwargs=orchestrator_kwargs,
            health_waits=health_waits,
            run_dir=runner.RunLayout(tmp_path, run_id or "unset").run_dir,
        )
        holder["harness"] = harness
        return harness

    return build


@pytest.fixture
def relay_probe(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Stub the startup Relay probe so relay-victim missions come up; returns the probe keyword arguments seen."""
    seen: list[Any] = []

    async def confirmed(**kwargs: Any) -> Any:
        seen.append(kwargs)
        return SimpleNamespace(records_observed=1, quarantined=False)

    contract = SimpleNamespace(
        url="http://victim/v1", request_payload=lambda prompt: {"prompt": prompt}, headers=lambda: {"x": "y"}
    )
    monkeypatch.setattr(runner, "contract_for", lambda _url, _victim: contract)
    monkeypatch.setattr(runner, "check_relay_instrumentation", confirmed)
    return seen


class StubJudge:
    """A judge that returns a fixed score, or raises the given exception."""

    def __init__(self, score: float | Exception) -> None:
        self._score = score

    def score(self, instructions: str, tool_name: str, args: str) -> float:
        del instructions, tool_name, args
        if isinstance(self._score, Exception):
            raise self._score
        return self._score


@pytest.fixture
def make_judge() -> Callable[[float | Exception], StubJudge]:
    return StubJudge


@pytest.fixture
def plugin_config() -> Callable[..., dict[str, Any]]:
    """Build a guardrail plugin config with ``count`` guardrails (``custom_guardrail_1``...)."""

    def build(count: int = 1, threshold: float = 0.7) -> dict[str, Any]:
        return {
            "model": {"model": "unused-in-test", "api_key_env": "INFERENCE_API_KEY"},
            "guardrails": [
                {
                    "name": f"custom_guardrail_{i}",
                    "target_tool": "transfer_funds",
                    "system_instructions": "Block transfers the user did not ask for.",
                    "threshold": threshold,
                }
                for i in range(1, count + 1)
            ],
        }

    return build

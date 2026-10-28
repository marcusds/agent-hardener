# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the deployment-agnostic run engine (run_mission)."""

from __future__ import annotations

import logging
import re
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from agent_hardener.errors import BenignSuiteError
from agent_hardener.models import TargetInput
from agent_hardener.openshell.lifecycle import OpenShellCommandResult
from agent_hardener.runtime.runner import Mission, MissionError, run_mission

if TYPE_CHECKING:
    from conftest import BuildMission

pytestmark = pytest.mark.unit


def _collect_output(sink: list[str]):  # type: ignore[no-untyped-def]
    """An on_event subscriber that records the runner's `output` highlight lines into `sink`."""

    def _on_event(event: str, payload: dict[str, Any]) -> None:
        if event == "output":
            sink.append(str(payload.get("line", "")))

    return _on_event


def _ok(output: str = "ok") -> OpenShellCommandResult:
    return OpenShellCommandResult(ok=True, command=[], output=output)


def _fail(output: str = "fail") -> OpenShellCommandResult:
    return OpenShellCommandResult(ok=False, command=[], output=output)


def _session(
    base_url: str = "http://127.0.0.1:8000/v1/chat/completions", root_dir: Path | None = None
) -> SimpleNamespace:
    # storage.root_dir/run-logs/<run_id>/ holds agent-hardener.log; pin run_id for a predictable path.
    return SimpleNamespace(
        victim_control="vc",
        target=SimpleNamespace(base_url=base_url, agent_relay_plugins=None),
        benign_validators=[],
        storage=SimpleNamespace(root_dir=root_dir or Path(tempfile.mkdtemp()), run_id="test-run"),
    )


def _run_log_path(root_dir: Path) -> Path:
    """Path to the run-level agent-hardener.log now living under run-logs/<run_id>/."""
    return root_dir / "run-logs" / "test-run" / "agent-hardener.log"


def _base_config(relay_victim: object = None) -> SimpleNamespace:
    return SimpleNamespace(
        relay_victim=relay_victim,
        cwd=None,
        health_timeout=120.0,
        sandbox="agent-hardener-test",
        start_command="run",
        build_context=None,
        image=None,
        policy_path=None,  # real OpenShellConfig always has one; None here skips the initial-state seed
    )


def _benign_preflight_mission(
    tmp_path: Path,
    *,
    suite_path: Path | None,
    implementation: str = "agent_hardener.agents.validators.smart_benign:run",
) -> tuple[Mission, SimpleNamespace]:
    agent = SimpleNamespace(name="smart-benign", implementation=implementation)
    config = SimpleNamespace(
        benign_validators=[agent],
        benign_suite_path=suite_path,
        storage=SimpleNamespace(root_dir=tmp_path),
        target=TargetInput(name="target", base_url="http://127.0.0.1:8000/v1/chat/completions"),
    )
    mission = Mission.__new__(Mission)
    mission.ctx = SimpleNamespace(session_config=config)
    mission._status = lambda _label: nullcontext()
    mission._announce = lambda _message: None
    return mission, agent


def _patch_common(monkeypatch: pytest.MonkeyPatch, base_config: object) -> None:
    monkeypatch.setattr("agent_hardener.runtime.runner.openshell_config", lambda _vc: base_config)
    monkeypatch.setattr("agent_hardener.runtime.runner.prepare_relay_victim", lambda cfg: cfg)
    monkeypatch.setattr("agent_hardener.runtime.runner.target_health_url", lambda _u: None)
    # Instrumentation has its own tests (tests/agent_hardener/preflight); these are about the run loop.
    monkeypatch.setattr(Mission, "_check_relay_instrumentation", lambda _self: None)
    monkeypatch.setattr("agent_hardener.runtime.runner.print_final_summary", lambda _reports, **_kw: None)


def _patch_orchestrator(monkeypatch: pytest.MonkeyPatch, *, success: bool = True) -> None:
    from agent_hardener.loggers import emit_event  # noqa: PLC0415

    class FakeOrchestrator:
        def __init__(self, _cfg: object, **_kwargs: object) -> None:
            pass

        async def run_rounds(self, rounds: object = None, mission_id: object = None, run_id: object = None) -> list:
            # Emit an agent-style record so tests can assert it reaches the run-log bus / agent-hardener.log.
            logging.getLogger("agent_hardener.agents.fake").info("fake-agent-log-line")
            # Emit a structured event via the ambient sink (the run's EventBus), like the real orchestrator.
            emit_event("phase_started", {"phase": "attackers", "count": 1})
            return [SimpleNamespace(success=success, iterations=[])]

    monkeypatch.setattr("agent_hardener.runtime.runner.Orchestrator", FakeOrchestrator)


def test_smart_benign_requires_explicit_suite(tmp_path: Path) -> None:
    mission, _agent = _benign_preflight_mission(tmp_path, suite_path=None)

    with pytest.raises(BenignSuiteError, match="requires an explicit benign suite") as exc_info:
        mission._load_benign_suite()

    assert exc_info.value.category == "benign_suite_failed"
    assert "--benign-suite PATH" in exc_info.value.remediation


def test_non_smart_benign_does_not_require_suite(tmp_path: Path) -> None:
    mission, _agent = _benign_preflight_mission(
        tmp_path,
        suite_path=None,
        implementation="tests.fake_benign:run",
    )

    mission._load_benign_suite()


def test_explicit_benign_suite_is_seeded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    suite_path = tmp_path / "suite.csv"
    mission, agent = _benign_preflight_mission(tmp_path, suite_path=suite_path)
    announcements: list[str] = []
    captured: dict[str, object] = {}
    mission._announce = announcements.append

    async def fake_synthesize(request: object, configured_agent: object, **kwargs: object) -> SimpleNamespace:
        captured.update(request=request, agent=configured_agent, **kwargs)
        return SimpleNamespace(viable=True, request_count=2, artifact_dir=tmp_path / "profile")

    monkeypatch.setattr("agent_hardener.runtime.runner.synthesize", fake_synthesize)

    mission._load_benign_suite()

    assert captured["agent"] is agent
    assert captured["source_suite"] == suite_path
    assert announcements == [f"benign suite ready: 2 request(s) → {tmp_path / 'profile'}"]


def test_empty_benign_suite_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    suite_path = tmp_path / "empty.csv"
    mission, _agent = _benign_preflight_mission(tmp_path, suite_path=suite_path)

    async def fake_synthesize(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(viable=False, request_count=0, artifact_dir=tmp_path / "profile")

    monkeypatch.setattr("agent_hardener.runtime.runner.synthesize", fake_synthesize)

    with pytest.raises(BenignSuiteError, match="contains no replayable requests"):
        mission._load_benign_suite()


def test_benign_suite_seed_failure_is_classified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    suite_path = tmp_path / "missing.csv"
    mission, _agent = _benign_preflight_mission(tmp_path, suite_path=suite_path)

    async def fake_synthesize(*_args: object, **_kwargs: object) -> SimpleNamespace:
        raise FileNotFoundError(suite_path)

    monkeypatch.setattr("agent_hardener.runtime.runner.synthesize", fake_synthesize)

    with pytest.raises(BenignSuiteError, match="failed to load benign suite") as exc_info:
        mission._load_benign_suite()

    assert isinstance(exc_info.value.__cause__, FileNotFoundError)


def test_benign_suite_preflight_runs_before_infrastructure() -> None:
    mission = Mission.__new__(Mission)
    calls: list[str] = []

    def reject_missing_suite() -> None:
        raise BenignSuiteError("missing suite")

    def open_sinks(_sinks: object) -> None:
        calls.append("open_sinks")

    def start_backends() -> None:
        calls.append("start_backends")

    def bring_up_victim() -> None:
        calls.append("bring_up_victim")

    def harden() -> list[SimpleNamespace]:
        calls.append("harden")
        return []

    def teardown() -> None:
        calls.append("teardown")

    mission._open_sinks = open_sinks
    mission._load_benign_suite = reject_missing_suite
    mission._start_backends = start_backends
    mission._bring_up_victim = bring_up_victim
    mission._harden = harden
    mission._teardown = teardown

    with pytest.raises(BenignSuiteError, match="missing suite"):
        mission.run()

    assert calls == ["open_sinks", "teardown"]


def test_run_mission_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    base_config = _base_config()
    _patch_common(monkeypatch, base_config)
    _patch_orchestrator(monkeypatch)

    class FakeLifecycle:
        def __init__(
            self, _cfg: object, runner: object = None, log_dir: object = None, active_state_dir: object = None
        ) -> None:
            pass

        def down(self) -> OpenShellCommandResult:
            return _ok("down")

        def up(self, _policy: object) -> OpenShellCommandResult:
            return _ok("up")

    monkeypatch.setattr("agent_hardener.runtime.runner.OpenShellLifecycle", FakeLifecycle)

    result = run_mission(_session())
    assert result.success is True
    assert result.reports


def test_run_mission_always_writes_full_report_to_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base_config = _base_config()
    _patch_common(monkeypatch, base_config)
    _patch_orchestrator(monkeypatch)

    class FakeLifecycle:
        def __init__(
            self, _cfg: object, runner: object = None, log_dir: object = None, active_state_dir: object = None
        ) -> None:
            pass

        def down(self) -> OpenShellCommandResult:
            return _ok("down")

        def up(self, _policy: object) -> OpenShellCommandResult:
            return _ok("up")

    monkeypatch.setattr("agent_hardener.runtime.runner.OpenShellLifecycle", FakeLifecycle)

    # verbose=False (the default): the full report must still land in agent-hardener.log.
    events_seen: list[tuple[str, dict]] = []
    run_mission(
        _session(root_dir=tmp_path),
        verbose=False,
        on_event=lambda event, payload: events_seen.append((event, payload)),
    )
    log_text = _run_log_path(tmp_path).read_text(encoding="utf-8")
    assert "Agent Hardener final log" in log_text
    # Runner highlight lines are `output` events, echoed to the log via the events bridge, timestamped.
    stamped = [line for line in log_text.splitlines() if "sandbox ready" in line]
    assert stamped, "highlight lines should be mirrored into the log"
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", stamped[0])
    assert "agent_hardener.events" in stamped[0]
    assert ("output", {"line": "sandbox ready"}) in events_seen
    # Agent-level logging is captured by the same bus and lands in agent-hardener.log.
    assert "fake-agent-log-line" in log_text

    # The orchestrator's event reaches the on_event subscriber and (via the _log_event bridge)
    # agent-hardener.log as a concise human line. Event persistence is the per-round events.jsonl, not a
    # run-level file, so nothing is written at the storage root.
    assert ("phase_started", {"phase": "attackers", "count": 1}) in events_seen
    assert "agent_hardener.events: phase_started phase=attackers count=1" in log_text
    assert "agent_hardener.agents.fake" in log_text
    assert not (tmp_path / "events.jsonl").exists()


def test_run_mission_loops_and_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    base_config = _base_config(
        SimpleNamespace(backends=["finance"], atof_path=Path("artifacts/relay/events.atof.jsonl"))
    )
    _patch_common(monkeypatch, base_config)
    restarts: list = []
    captured: dict[str, object] = {}

    class CapturingOrchestrator:
        def __init__(self, _cfg: object, **_kwargs: object) -> None:
            pass

        async def run_rounds(self, rounds: object = None, mission_id: object = None, run_id: object = None) -> list:
            captured["rounds"] = rounds  # the runner passes rounds as the round count
            return [SimpleNamespace(success=True, iterations=[])]

    monkeypatch.setattr("agent_hardener.runtime.runner.Orchestrator", CapturingOrchestrator)

    class FakeLifecycle:
        def __init__(
            self, _cfg: object, runner: object = None, log_dir: object = None, active_state_dir: object = None
        ) -> None:
            pass

        def down(self) -> OpenShellCommandResult:
            return _ok("down")

        def up(self, _policy: object) -> OpenShellCommandResult:
            return _ok("up")

        def restart(self, policy: object, wait_for_health: bool = True) -> OpenShellCommandResult:
            restarts.append((policy, wait_for_health))
            return _ok("restart")

    class FakeBackendManager:
        def up(self, _specs: object, _cwd: object) -> OpenShellCommandResult:
            return _ok("backends-up")

        def down(self, _specs: object, _cwd: object) -> OpenShellCommandResult:
            return _ok("backends-down")

    monkeypatch.setattr("agent_hardener.runtime.runner.OpenShellLifecycle", FakeLifecycle)
    monkeypatch.setattr("agent_hardener.runtime.runner.BackendManager", FakeBackendManager)

    outputs: list[str] = []
    result = run_mission(_session(), rounds=2, mission_id="m", on_event=_collect_output(outputs))
    assert result.success is True
    assert "backends ready" in outputs  # the "backends-up" blob is teed to the log, not an output event
    assert captured["rounds"] == 2  # rounds flows to a single run_rounds as the round count
    assert restarts == []  # the runner no longer restarts between rounds — the orchestrator owns the loop


def test_run_mission_sandbox_failure_raises_and_cleans_up(monkeypatch: pytest.MonkeyPatch) -> None:
    base_config = _base_config()
    _patch_common(monkeypatch, base_config)
    _patch_orchestrator(monkeypatch)
    downs: list[str] = []

    class FakeLifecycle:
        def __init__(
            self, _cfg: object, runner: object = None, log_dir: object = None, active_state_dir: object = None
        ) -> None:
            pass

        def down(self) -> OpenShellCommandResult:
            downs.append("down")
            return _ok("down")

        def up(self, _policy: object) -> OpenShellCommandResult:
            return _fail("up-failed")

    monkeypatch.setattr("agent_hardener.runtime.runner.OpenShellLifecycle", FakeLifecycle)

    with pytest.raises(MissionError, match="failed to start"):
        run_mission(_session())
    assert downs  # lifecycle.down called in cleanup


def test_run_mission_backend_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    base_config = _base_config(
        SimpleNamespace(backends=["finance"], atof_path=Path("artifacts/relay/events.atof.jsonl"))
    )
    _patch_common(monkeypatch, base_config)
    _patch_orchestrator(monkeypatch)

    class FakeLifecycle:
        def __init__(
            self, _cfg: object, runner: object = None, log_dir: object = None, active_state_dir: object = None
        ) -> None:
            pass

        def down(self) -> OpenShellCommandResult:
            return _ok("down")

    class FakeBackendManager:
        def up(self, _specs: object, _cwd: object) -> OpenShellCommandResult:
            return _fail("backend-boom")

        def down(self, _specs: object, _cwd: object) -> OpenShellCommandResult:
            return _ok("backends-down")

    monkeypatch.setattr("agent_hardener.runtime.runner.OpenShellLifecycle", FakeLifecycle)
    monkeypatch.setattr("agent_hardener.runtime.runner.BackendManager", FakeBackendManager)
    with pytest.raises(MissionError, match="backend startup failed"):
        run_mission(_session())


def test_run_mission_emits_status_events(monkeypatch: pytest.MonkeyPatch) -> None:
    base_config = _base_config()
    _patch_common(monkeypatch, base_config)
    _patch_orchestrator(monkeypatch)
    # Give the victim a health URL so the health-wait step (and its status bracket) runs.
    monkeypatch.setattr("agent_hardener.runtime.runner.target_health_url", lambda _u: "http://h/health")
    monkeypatch.setattr("agent_hardener.runtime.runner.wait_for_health", lambda _u, _t: None)

    class FakeLifecycle:
        def __init__(
            self, _cfg: object, runner: object = None, log_dir: object = None, active_state_dir: object = None
        ) -> None:
            pass

        def down(self) -> OpenShellCommandResult:
            return _ok("down")

        def up(self, _policy: object) -> OpenShellCommandResult:
            return _ok("up")

        def restart(self, _policy: object, wait_for_health: bool = True) -> OpenShellCommandResult:
            return _ok("restart")

    monkeypatch.setattr("agent_hardener.runtime.runner.OpenShellLifecycle", FakeLifecycle)

    labels: list[str] = []

    def on_event(event: str, payload: dict[str, Any]) -> None:
        if event == "status_started":
            labels.append(str(payload.get("label", "")))

    result = run_mission(_session(), rounds=2, on_event=on_event)
    assert result.reports  # reports populated
    assert "Building and starting sandbox" in labels
    assert "Waiting for victim health" in labels
    assert "Restarting sandbox" not in labels  # the runner no longer restarts between rounds


def test_run_mission_tees_detail_to_log_and_surfaces_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base_config = _base_config()
    _patch_common(monkeypatch, base_config)
    _patch_orchestrator(monkeypatch)

    class FakeLifecycle:
        def __init__(
            self, _cfg: object, runner: object = None, log_dir: object = None, active_state_dir: object = None
        ) -> None:
            pass

        def down(self) -> OpenShellCommandResult:
            return _ok("down")

        def up(self, _policy: object) -> OpenShellCommandResult:
            return _ok("UP-BLOB")

    monkeypatch.setattr("agent_hardener.runtime.runner.OpenShellLifecycle", FakeLifecycle)

    highlights: list[str] = []
    run_mission(_session(root_dir=tmp_path), on_event=_collect_output(highlights))
    log_text = _run_log_path(tmp_path).read_text(encoding="utf-8")
    assert "UP-BLOB" in log_text  # success blob teed to the log file, never inline
    assert "UP-BLOB" not in highlights
    assert "sandbox ready" in highlights  # explicit highlight shown even when quiet
    assert any("agent-hardener.log" in line for line in highlights)  # pointer to the log printed


def test_run_mission_failure_output_surfaced_to_highlights(monkeypatch: pytest.MonkeyPatch) -> None:
    base_config = _base_config()
    _patch_common(monkeypatch, base_config)
    _patch_orchestrator(monkeypatch)

    class FakeLifecycle:
        def __init__(
            self, _cfg: object, runner: object = None, log_dir: object = None, active_state_dir: object = None
        ) -> None:
            pass

        def down(self) -> OpenShellCommandResult:
            return _ok("down")

        def up(self, _policy: object) -> OpenShellCommandResult:
            return _fail("BOOM")

    monkeypatch.setattr("agent_hardener.runtime.runner.OpenShellLifecycle", FakeLifecycle)

    highlights: list[str] = []
    with pytest.raises(MissionError, match="failed to start"):
        run_mission(_session(), on_event=_collect_output(highlights))
    assert "BOOM" in highlights  # a real failure surfaces inline even in quiet mode


def test_run_mission_uploaders_use_prepared_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    # The uploaders must share the lifecycle built on the PREPARED config (staged build_context), so a
    # defender recreate rebuilds the sandbox instead of destroying it. Assert both wrap that lifecycle.
    base_config = _base_config()
    _patch_common(monkeypatch, base_config)
    captured: dict[str, object] = {}

    class CapturingOrchestrator:
        def __init__(
            self, _cfg: object, openshell_uploader: object = None, relay_uploader: object = None, **_kwargs: object
        ) -> None:
            captured["openshell"] = openshell_uploader
            captured["nat"] = relay_uploader

        async def run_rounds(self, rounds: object = None, mission_id: object = None, run_id: object = None) -> list:
            return [SimpleNamespace(success=True, iterations=[])]

    monkeypatch.setattr("agent_hardener.runtime.runner.Orchestrator", CapturingOrchestrator)

    class FakeLifecycle:
        def __init__(
            self, cfg: object, runner: object = None, log_dir: object = None, active_state_dir: object = None
        ) -> None:
            self.config = cfg
            captured["active_state_dir"] = active_state_dir

        def down(self) -> OpenShellCommandResult:
            return _ok("down")

        def up(self, _policy: object) -> OpenShellCommandResult:
            return _ok("up")

    monkeypatch.setattr("agent_hardener.runtime.runner.OpenShellLifecycle", FakeLifecycle)

    run_mission(_session())
    openshell_uploader = captured["openshell"]
    relay_uploader = captured["nat"]
    assert openshell_uploader is not None
    assert relay_uploader is not None
    # Both uploaders wrap the same lifecycle, built on the prepared config (staged build_context).
    assert openshell_uploader.lifecycle.config is base_config
    # A recreate re-uploads the run's hardened guardrails, not the seed.
    assert Path(str(captured["active_state_dir"])).name == "victim-active-state"
    assert relay_uploader.lifecycle is openshell_uploader.lifecycle


def _teardown_mission(tmp_path: Path, diagnostics: OpenShellCommandResult, calls: list[str]) -> Mission:
    class FakeLifecycle:
        def collect_victim_diagnostics(self) -> OpenShellCommandResult:
            calls.append("diagnostics")
            return diagnostics

        def down(self) -> OpenShellCommandResult:
            calls.append("down")
            return _ok("down")

    mission = Mission.__new__(Mission)
    mission.ctx = SimpleNamespace(no_cleanup=False, session_config=_session(root_dir=tmp_path))
    mission._run_id = "test-run"
    mission._lifecycle = FakeLifecycle()
    mission._backend_manager = None
    mission._detail = lambda _output: None
    mission._run_log = logging.getLogger("test")
    (tmp_path / "run-logs" / "test-run").mkdir(parents=True)
    return mission


def test_teardown_saves_victim_logs_before_deleting_sandbox(tmp_path: Path) -> None:
    calls: list[str] = []
    mission = _teardown_mission(tmp_path, _ok("=== app log ===\nTraceback: boom"), calls)

    mission._teardown()

    assert calls == ["diagnostics", "down"]
    saved = tmp_path / "run-logs" / "test-run" / "victim-diagnostics.log"
    assert saved.read_text(encoding="utf-8") == "=== app log ===\nTraceback: boom"


def test_teardown_still_deletes_sandbox_when_diagnostics_fail(tmp_path: Path) -> None:
    calls: list[str] = []
    mission = _teardown_mission(tmp_path, _ok(), calls)

    def broken() -> OpenShellCommandResult:
        raise RuntimeError("sandbox exec failed")

    mission._lifecycle.collect_victim_diagnostics = broken
    mission._teardown()

    assert calls == ["down"]
    assert not (tmp_path / "run-logs" / "test-run" / "victim-diagnostics.log").exists()


def test_no_cleanup_leaves_the_sandbox_and_backends_running(build_mission: BuildMission) -> None:
    harness = build_mission(no_cleanup=True)

    harness.run()

    assert harness.lifecycle.calls == ["down", "up"]  # the initial rebuild only; no teardown down
    assert "Cleanup skipped; sandbox + backends left running." in harness.lines


@pytest.mark.usefixtures("relay_probe")
def test_teardown_stops_the_sandbox_and_the_backends(build_mission: BuildMission, tmp_path: Path) -> None:
    harness = build_mission(backends=("db",))

    harness.run()

    assert harness.lifecycle.calls[-1] == "down"
    assert harness.backends.downed == [(["db"], tmp_path)]
    assert "backends down" in harness.run_log


def test_teardown_without_backends_only_stops_the_sandbox(build_mission: BuildMission) -> None:
    harness = build_mission()

    harness.run()

    assert harness.lifecycle.calls == ["down", "up", "down"]
    assert harness.backends.downed == []

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mission start-up: run preparation, host backends, sandbox reuse gating, health and the Relay probe."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from agent_hardener.errors import VictimUnavailableError
from agent_hardener.openshell.lifecycle import OpenShellCommandResult
from agent_hardener.runtime import runner
from agent_hardener.runtime.runner import MissionError

if TYPE_CHECKING:
    from conftest import BuildMission

pytestmark = pytest.mark.unit

_REUSING = "reusing sandbox (agent fingerprint matches)"
_REBUILDING = "fingerprint mismatch or unknown"


def _failed(output: str) -> OpenShellCommandResult:
    return OpenShellCommandResult(ok=False, command=[], output=output)


# --- preparation -----------------------------------------------------------------------------------------


def test_the_run_is_seeded_from_the_prepared_policy_and_uploads_through_the_prepared_sandbox(
    build_mission: BuildMission, tmp_path: Path
) -> None:
    inferred = tmp_path / "inferred-policy.yaml"
    harness = build_mission(policy_path=inferred, run_id=None, no_cleanup=True)

    harness.run()

    assert harness.session.storage.victim_policy_path == inferred
    (orchestrator,) = harness.orchestrator_kwargs
    assert orchestrator["openshell_uploader"] == ("shell", harness.lifecycle)
    assert orchestrator["relay_uploader"] == ("relay", harness.lifecycle)
    assert orchestrator["run_rounds"]["run_id"]


# --- host backends ---------------------------------------------------------------------------------------


def test_without_backends_nothing_is_started(build_mission: BuildMission) -> None:
    harness = build_mission(no_cleanup=True)

    harness.run()

    assert "backend setup: starting host backends" not in harness.lines


@pytest.mark.usefixtures("relay_probe")
def test_backends_are_announced_and_their_build_output_goes_to_the_run_log(build_mission: BuildMission) -> None:
    harness = build_mission(backends=("db",), no_cleanup=True)

    harness.run()

    assert "backend setup: starting host backends" in harness.lines
    assert "backends ready" in harness.lines
    assert "compose output" in harness.run_log


@pytest.mark.usefixtures("relay_probe")
def test_a_backend_that_fails_to_start_surfaces_the_error_and_stops_the_run(build_mission: BuildMission) -> None:
    harness = build_mission(backends=("db",), no_cleanup=True)
    harness.backends.up_result = _failed("port in use")

    with pytest.raises(MissionError, match="host backend startup failed"):
        harness.run()

    assert "port in use" in harness.lines
    assert harness.lifecycle.calls == []


# --- bringing the victim up ------------------------------------------------------------------------------


def test_a_fresh_start_builds_the_sandbox_once(build_mission: BuildMission) -> None:
    harness = build_mission(no_cleanup=True)

    harness.run()

    assert harness.lifecycle.calls == ["down", "up"]
    assert harness.statuses == ["Building and starting sandbox"]
    assert "sandbox ready" in harness.lines
    assert "built" in harness.run_log


def test_reuse_skips_the_rebuild_when_the_same_agent_was_built_before(build_mission: BuildMission) -> None:
    build_mission(no_cleanup=True).run()
    reused = build_mission(skip_initial_up=True, no_cleanup=True)

    reused.run()

    assert reused.lifecycle.calls == []
    assert _REUSING in reused.lines


def test_reuse_rebuilds_and_says_why_when_a_different_agent_was_built_before(build_mission: BuildMission) -> None:
    build_mission(start_command="run-old", no_cleanup=True).run()
    rebuilt = build_mission(start_command="run-new", skip_initial_up=True, no_cleanup=True)

    rebuilt.run()

    assert rebuilt.lifecycle.calls == ["down", "up"]
    assert _REBUILDING in rebuilt.lines[0]
    assert _REUSING not in rebuilt.lines
    # The rebuild records the new agent, so the next reuse is accepted.
    again = build_mission(start_command="run-new", skip_initial_up=True, no_cleanup=True)
    again.run()
    assert _REUSING in again.lines


def test_reuse_with_no_record_of_a_previous_build_rebuilds(build_mission: BuildMission) -> None:
    harness = build_mission(skip_initial_up=True, no_cleanup=True)

    harness.run()

    assert _REBUILDING in harness.lines[0]
    assert harness.lifecycle.calls == ["down", "up"]


def test_a_sandbox_that_fails_to_start_surfaces_its_output_and_is_never_trusted_for_reuse(
    build_mission: BuildMission,
) -> None:
    failed = build_mission(no_cleanup=True)
    failed.lifecycle.up_result = _failed("docker build failed")

    with pytest.raises(MissionError, match="OpenShell sandbox failed to start"):
        failed.run()

    assert "docker build failed" in failed.lines
    retried = build_mission(skip_initial_up=True, no_cleanup=True)
    retried.run()
    assert _REBUILDING in retried.lines[0]


def test_health_is_awaited_with_the_configured_timeout(build_mission: BuildMission) -> None:
    harness = build_mission(health_url="http://victim/health", no_cleanup=True)

    harness.run()

    assert harness.health_waits == [("http://victim/health", 5.0)]
    assert "victim health ready" in harness.lines
    assert "Waiting for victim health" in harness.statuses


def test_a_victim_that_never_turns_healthy_is_reported_as_unavailable(
    build_mission: BuildMission, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = build_mission(health_url="http://victim/health", no_cleanup=True)

    def never_healthy(_url: str, _timeout: float) -> None:
        raise TimeoutError

    monkeypatch.setattr(runner, "wait_for_health", never_healthy)

    with pytest.raises(VictimUnavailableError, match=r"did not become healthy at http://victim/health within 5\.0s"):
        harness.run()

    assert "victim health ready" not in harness.lines


# --- relay instrumentation -------------------------------------------------------------------------------


def test_the_relay_probe_is_skipped_for_a_plain_http_target(
    build_mission: BuildMission, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "check_relay_instrumentation", lambda **_k: pytest.fail("must not probe"))
    harness = build_mission(no_cleanup=True)

    harness.run()

    assert "Checking Relay instrumentation" not in harness.statuses


class _FakeHttpClient:
    posts: list[dict[str, Any]]

    def __init__(self, *, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> _FakeHttpClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str]) -> str:
        type(self).posts.append({"url": url, "json": json, "headers": headers})
        return "response"


@pytest.mark.usefixtures("relay_probe")
@pytest.mark.parametrize("quarantined", [False, True])
@pytest.mark.parametrize("relative", [True, False], ids=["relative-atof", "absolute-atof"])
def test_the_relay_probe_hits_the_victim_and_syncs_the_atof_log_from_the_sandbox(
    build_mission: BuildMission,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    quarantined: bool,
    relative: bool,
) -> None:
    atof = Path("relay/events.atof.jsonl") if relative else tmp_path / "abs" / "events.atof.jsonl"
    _FakeHttpClient.posts = []
    monkeypatch.setattr(runner.httpx, "AsyncClient", _FakeHttpClient)
    seen: dict[str, Any] = {}

    async def fake_check(*, probe: Any, atof_path: Path, sync: Any) -> Any:
        seen["response"] = await probe()
        seen["atof_path"] = atof_path
        sync()
        return SimpleNamespace(records_observed=4, quarantined=quarantined)

    monkeypatch.setattr(runner, "check_relay_instrumentation", fake_check)
    harness = build_mission(atof_path=atof, no_cleanup=True)
    host_atof = tmp_path / atof if relative else atof

    harness.run()

    assert seen == {"response": "response", "atof_path": host_atof}
    assert _FakeHttpClient.posts == [
        {"url": "http://victim/v1", "json": {"prompt": runner.RELAY_PROBE_PROMPT}, "headers": {"x": "y"}}
    ]
    assert harness.lifecycle.fetched == [(f"{runner.SANDBOX_ARTIFACTS_DIR}/events.atof.jsonl", host_atof)]
    assert "Checking Relay instrumentation" in harness.statuses
    assert "relay instrumentation confirmed (4 event(s))" in harness.lines
    assert any("dirty scope stack" in line for line in harness.lines) is quarantined


def test_reuse_adopts_the_pre_built_image_the_sandbox_was_created_from(build_mission: BuildMission) -> None:
    build_mission(image="reg/agent:1", no_cleanup=True).run()
    reused = build_mission(skip_initial_up=True, no_cleanup=True)

    reused.run()

    assert reused.lifecycle.calls == []
    assert reused.lifecycle.config.image == "reg/agent:1"
    assert "reusing the pre-built victim image recorded for this sandbox: reg/agent:1" in reused.lines
    assert _REUSING in reused.lines


def test_an_explicit_image_wins_over_the_recorded_one(build_mission: BuildMission) -> None:
    build_mission(image="reg/agent:1", no_cleanup=True).run()
    rebuilt = build_mission(image="reg/agent:2", skip_initial_up=True, no_cleanup=True)

    rebuilt.run()

    assert rebuilt.lifecycle.calls == ["down", "up"]
    assert rebuilt.lifecycle.config.image == "reg/agent:2"


def test_a_local_build_forgets_the_recorded_image(build_mission: BuildMission) -> None:
    build_mission(image="reg/agent:1", no_cleanup=True).run()
    build_mission(start_command="run-new", no_cleanup=True).run()
    reused = build_mission(start_command="run-new", skip_initial_up=True, no_cleanup=True)

    reused.run()

    assert reused.lifecycle.config.image is None
    assert _REUSING in reused.lines

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a finished mission leaves behind: JSON run artifacts and the hardened image bundle."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from agent_hardener.openshell.relay_victim import RELAY_PLUGINS_UPLOAD_DEST
from agent_hardener.runtime import runner

if TYPE_CHECKING:
    from conftest import BuildMission, MissionHarness

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def no_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither artifact is produced unless a test supplies a builder."""
    monkeypatch.setattr(runner, "build_mitigations", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "build_validation", lambda _reports: None)


# --- run artifacts ---------------------------------------------------------------------------------------


def test_run_artifacts_are_written_as_indented_json(
    build_mission: BuildMission, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "build_mitigations", lambda *_a, **_k: {"before": "a", "after": "b"})
    monkeypatch.setattr(runner, "build_validation", lambda _reports: {"attacks": []})
    harness = build_mission(no_cleanup=True)

    harness.run()

    assert harness.read_artifact("mitigations.json") == {"before": "a", "after": "b"}
    assert (harness.run_dir / "mitigations.json").read_text().startswith('{\n  "before"')
    assert harness.read_artifact("validation.json") == {"attacks": []}


@pytest.mark.parametrize("payload", [None, {}, []], ids=["none", "empty-dict", "empty-list"])
def test_falsy_run_artifacts_are_not_written(
    build_mission: BuildMission, monkeypatch: pytest.MonkeyPatch, payload: Any
) -> None:
    monkeypatch.setattr(runner, "build_validation", lambda _reports: payload)
    harness = build_mission(no_cleanup=True)

    harness.run()

    assert not (harness.run_dir / "validation.json").exists()


def test_a_failing_artifact_builder_never_breaks_the_run(
    build_mission: BuildMission, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(_reports: Any) -> dict[str, str]:
        raise ValueError("bad report")

    monkeypatch.setattr(runner, "build_validation", broken)
    harness = build_mission(no_cleanup=True)

    result = harness.run()

    assert result.success
    assert not (harness.run_dir / "validation.json").exists()


# --- hardened bundle -------------------------------------------------------------------------------------


@pytest.fixture
def written_bundles(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_bundle(**kwargs: Any) -> Path:
        calls.append(kwargs)
        return Path(kwargs["destination"])

    monkeypatch.setattr(runner, "write_hardened_bundle", fake_bundle)
    return calls


def _deploy_guardrails(harness: MissionHarness) -> Path:
    """Put a hardened plugins.toml where a defender would have left it for this run."""
    active_state = runner.victim_active_state_dir(harness.run_dir)
    active_state.mkdir(parents=True, exist_ok=True)
    plugins = active_state / RELAY_PLUGINS_UPLOAD_DEST.rsplit("/", 1)[-1]
    plugins.write_text("[[components]]\n")
    return plugins


@pytest.mark.usefixtures("relay_probe")
@pytest.mark.parametrize("with_relay", [False, True], ids=["no-relay-victim", "no-build-context"])
def test_bundle_is_skipped_without_a_relay_victim_or_build_context(
    build_mission: BuildMission, written_bundles: list[dict[str, Any]], tmp_path: Path, with_relay: bool
) -> None:
    harness = build_mission(
        atof_path=tmp_path / "events.atof.jsonl" if with_relay else None,
        build_context=None if with_relay else tmp_path / "image" / "Dockerfile",
        no_cleanup=True,
    )

    harness.run()

    assert written_bundles == []


@pytest.mark.usefixtures("relay_probe")
def test_bundle_is_skipped_when_no_guardrails_were_deployed(
    build_mission: BuildMission, written_bundles: list[dict[str, Any]], tmp_path: Path
) -> None:
    harness = build_mission(
        atof_path=tmp_path / "events.atof.jsonl", build_context=tmp_path / "image" / "Dockerfile", no_cleanup=True
    )

    harness.run()

    assert written_bundles == []
    assert not any("hardened image bundle" in line for line in harness.lines)


@pytest.mark.usefixtures("relay_probe")
@pytest.mark.parametrize("policy_name", [None, "policy.yaml"])
def test_bundle_ships_the_guardrails_and_the_hardened_policy(
    build_mission: BuildMission, written_bundles: list[dict[str, Any]], tmp_path: Path, policy_name: str | None
) -> None:
    harness = build_mission(
        atof_path=tmp_path / "events.atof.jsonl", build_context=tmp_path / "image" / "Dockerfile", no_cleanup=True
    )
    if policy_name:
        harness.session.storage.victim_policy_path = tmp_path / "configured" / policy_name
    plugins = _deploy_guardrails(harness)

    harness.run()

    (call,) = written_bundles
    active_state = plugins.parent
    assert call["build_root"] == tmp_path / "image"
    assert call["plugins_toml"] == plugins
    assert call["destination"] == harness.run_dir / "hardened-image"
    assert (call["agent"], call["run_id"]) == ("agent", "run-1")
    assert call["policy"] == (active_state / policy_name if policy_name else None)
    assert call["image"] is None
    assert f"  → hardened image bundle: {harness.run_dir / 'hardened-image'}" in harness.lines


@pytest.mark.usefixtures("relay_probe")
def test_bundle_names_the_pre_built_image_the_run_validated(
    build_mission: BuildMission, written_bundles: list[dict[str, Any]], tmp_path: Path
) -> None:
    harness = build_mission(
        atof_path=tmp_path / "events.atof.jsonl",
        build_context=tmp_path / "image" / "Dockerfile",
        image="reg/agent:1",
        no_cleanup=True,
    )
    _deploy_guardrails(harness)

    harness.run()

    (call,) = written_bundles
    assert call["image"] == "reg/agent:1"
    assert call["build_root"] == tmp_path / "image"


@pytest.mark.usefixtures("relay_probe")
def test_a_packaging_failure_is_logged_and_does_not_fail_the_run(
    build_mission: BuildMission, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def broken(**_kwargs: Any) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(runner, "write_hardened_bundle", broken)
    harness = build_mission(
        atof_path=tmp_path / "events.atof.jsonl", build_context=tmp_path / "image" / "Dockerfile", no_cleanup=True
    )
    _deploy_guardrails(harness)

    result = harness.run()

    assert result.success
    assert "could not assemble the hardened image bundle" in harness.run_log
    assert not any("hardened image bundle" in line for line in harness.lines)

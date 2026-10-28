# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the --reuse agent-fingerprint guard in run_mission."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from agent_hardener.runtime import runner
from agent_hardener.swarm_tracker import agent_fingerprint_path

if TYPE_CHECKING:
    from pathlib import Path


def _session(workflow: Path | None) -> SimpleNamespace:
    return SimpleNamespace(target=SimpleNamespace(agent_relay_plugins=workflow))


def _openshell(
    sandbox: str = "agent-hardener-finance", start: str = "/app/start.sh", image: str | None = None
) -> SimpleNamespace:
    return SimpleNamespace(sandbox=sandbox, start_command=start, build_context=None, image=image)


def test_fingerprint_is_deterministic(tmp_path: Path) -> None:
    wf = tmp_path / "workflow.yaml"
    wf.write_text("agent: finance\n", encoding="utf-8")
    fp1 = runner._agent_fingerprint(_session(wf), _openshell())
    fp2 = runner._agent_fingerprint(_session(wf), _openshell())
    assert fp1 == fp2


def test_fingerprint_changes_when_workflow_changes(tmp_path: Path) -> None:
    finance = tmp_path / "finance.yaml"
    finance.write_text("agent: finance — expense tools\n", encoding="utf-8")
    review = tmp_path / "review.yaml"
    review.write_text("agent: code-review — PR tools\n", encoding="utf-8")
    # Same sandbox name + start command, different workflow → different fingerprint.
    assert runner._agent_fingerprint(_session(finance), _openshell()) != runner._agent_fingerprint(
        _session(review), _openshell()
    )


def test_fingerprint_changes_with_start_command(tmp_path: Path) -> None:
    wf = tmp_path / "workflow.yaml"
    wf.write_text("agent: finance\n", encoding="utf-8")
    assert runner._agent_fingerprint(_session(wf), _openshell(start="/app/a.sh")) != runner._agent_fingerprint(
        _session(wf), _openshell(start="/app/b.sh")
    )


def test_fingerprint_changes_with_victim_image(tmp_path: Path) -> None:
    wf = tmp_path / "workflow.yaml"
    wf.write_text("agent: finance\n", encoding="utf-8")
    fingerprints = {
        runner._agent_fingerprint(_session(wf), _openshell(image=image))
        for image in (None, "registry.example.com/agent:1", "registry.example.com/agent:2")
    }
    assert len(fingerprints) == 3


def test_fingerprint_sidecar_roundtrip_and_mismatch(tmp_path: Path) -> None:
    wf = tmp_path / "workflow.yaml"
    wf.write_text("agent: finance\n", encoding="utf-8")
    fp = runner._agent_fingerprint(_session(wf), _openshell())
    path = agent_fingerprint_path(tmp_path, "agent-hardener-finance")

    assert runner._read_sidecar(path) is None  # no sidecar yet → cannot verify → must rebuild
    runner._write_sidecar(path, fp)
    assert runner._read_sidecar(path) == fp  # matches → reuse is safe

    wf.write_text("agent: code-review\n", encoding="utf-8")  # swap the agent behind the same sandbox
    assert runner._read_sidecar(path) != runner._agent_fingerprint(_session(wf), _openshell())

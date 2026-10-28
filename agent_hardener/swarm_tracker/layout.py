# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run-log directory layout primitives for Swarm Tracker.

This module is the single authority for where artifacts live. Names + scopes are declared once
(:class:`Scope`, :class:`ArtifactKind`, the catalog constants); :class:`RunLayout` resolves a kind to a
concrete directory for a given run. The composition root builds one ``RunLayout`` per run and injects the
*resolved* dir into each component — components never import the catalog.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

_RUN_LOGS = "run-logs"
_VICTIM_ACTIVE_STATE = "victim-active-state"
_VICTIM_STATE = "victim-state"
_INIT = "init"
_AGENT_HARDENER = ".agent-hardener"
_BUILDS = "builds"
_BENIGN_PROFILES = "benign_profiles"


class Scope(Enum):
    """Where an artifact kind lives, from broadest to narrowest."""

    GLOBAL = "global"  # cwd/.agent-hardener — cross-run (builds, patched policies, forward pid)
    TARGET = "target"  # root_dir/benign_profiles/<target> — per-target synth cache
    RUN = "run"  # run-logs/<run_id>
    ROUND = "round"  # run-logs/<run_id>/round_<n>
    ITERATION = "iteration"  # run-logs/<run_id>/round_<n>/iteration-<n>


@dataclass(frozen=True)
class ArtifactKind:
    """A named artifact and the scope it is written under."""

    name: str
    scope: Scope


# Catalog: the subprocess-output kinds a component must be handed a directory for.
OPENSHELL_LOGS = ArtifactKind("openshell-logs", Scope.RUN)
GARAK = ArtifactKind("garak", Scope.ROUND)  # attackers run once per round, so garak scans are round-scoped


@dataclass(frozen=True)
class RunLayout:
    """Per-run resolver: maps an :class:`ArtifactKind` to its concrete directory for this run.

    Built once at the composition root and injected — NOT a module global (so tests and a UI driving
    multiple/concurrent runs each hold their own).
    """

    root_dir: Path
    run_id: str

    @property
    def run_dir(self) -> Path:
        return create_run_dir(self.root_dir, self.run_id)

    def round_dir(self, round_number: int) -> Path:
        return create_round_dir(self.run_dir, round_number)

    def iteration_dir(self, round_number: int, iteration: int) -> Path:
        return create_iteration_dir(self.round_dir(round_number), iteration)

    def base(
        self,
        scope: Scope,
        *,
        round_n: int | None = None,
        iteration_n: int | None = None,
        target: str | None = None,
    ) -> Path:
        """Return the base directory for ``scope`` (before the kind name is appended)."""
        if scope is Scope.GLOBAL:
            return Path.cwd() / _AGENT_HARDENER
        if scope is Scope.TARGET:
            if target is None:
                raise ValueError("TARGET scope requires target=")
            return self.root_dir / _BENIGN_PROFILES / target
        if scope is Scope.RUN:
            return self.run_dir
        if scope is Scope.ROUND:
            if round_n is None:
                raise ValueError("ROUND scope requires round_n=")
            return self.round_dir(round_n)
        if round_n is None or iteration_n is None:  # Scope.ITERATION
            raise ValueError("ITERATION scope requires round_n= and iteration_n=")
        return self.iteration_dir(round_n, iteration_n)

    def dir_for(
        self,
        kind: ArtifactKind,
        *,
        round_n: int | None = None,
        iteration_n: int | None = None,
        target: str | None = None,
    ) -> Path:
        """Resolve, create, and return the directory a component writes this kind of artifact into."""
        artifact_dir = self.base(kind.scope, round_n=round_n, iteration_n=iteration_n, target=target) / kind.name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir


def create_run_dir(root_dir: Path, run_id: str) -> Path:
    """Create and return <root_dir>/run_logs/<run_id>/."""
    run_dir = root_dir / _RUN_LOGS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def create_round_dir(run_dir: Path, round_number: int) -> Path:
    """Create and return <run_dir>/round_<N>/."""
    round_dir = run_dir / f"round_{round_number}"
    round_dir.mkdir(parents=True, exist_ok=True)
    return round_dir


def create_iteration_dir(round_dir: Path, iteration: int) -> Path:
    """Create and return <round_dir>/iteration-<N>/."""
    iteration_dir = round_dir / f"iteration-{iteration}"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    return iteration_dir


def run_logs_root(root_dir: Path) -> Path:
    """The ``<root_dir>/run-logs`` directory that holds every run (not created here)."""
    return root_dir / _RUN_LOGS


def agent_fingerprint_path(cwd: Path, sandbox: str) -> Path:
    """Sidecar recording the fingerprint of the agent last built into ``sandbox`` (GLOBAL scope; not created here)."""
    return cwd / _AGENT_HARDENER / _BUILDS / sandbox / "agent_fingerprint.txt"


def victim_image_path(cwd: Path, sandbox: str) -> Path:
    """Sidecar recording the pre-built image ``sandbox`` was last created from (GLOBAL scope; not created here)."""
    return cwd / _AGENT_HARDENER / _BUILDS / sandbox / "victim_image.txt"


def read_victim_image(cwd: Path, sandbox: str) -> str | None:
    """The pre-built image ``sandbox`` was created from, or None after a local build or when unknown."""
    try:
        return victim_image_path(cwd, sandbox).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def write_victim_image(cwd: Path, sandbox: str, image: str | None) -> None:
    """Record (or, for a locally built sandbox, forget) the pre-built image ``sandbox`` was created from."""
    path = victim_image_path(cwd, sandbox)
    if image is None:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(image + "\n", encoding="utf-8")


def benign_profiles_dir(root_dir: Path, target: str) -> Path:
    """The per-target synth cache dir ``<root_dir>/benign_profiles/<target>`` (not created here)."""
    return root_dir / _BENIGN_PROFILES / target


def victim_active_state_dir(run_dir: Path) -> Path:
    """The ``<run_dir>/victim-active-state`` dir holding the deployed victim files (not created here)."""
    return run_dir / _VICTIM_ACTIVE_STATE


def init_dir(run_dir: Path) -> Path:
    """The ``<run_dir>/init`` dir holding the baseline victim files (not created here)."""
    return run_dir / _INIT


def write_init_files(run_dir: Path, policy_path: Path | None, workflow_path: Path | None) -> None:
    """Copy baseline victim agent files into <run_dir>/init/. Called once before loop 1."""
    init_dir = run_dir / _INIT
    init_dir.mkdir(parents=True, exist_ok=True)
    _copy_victim_files(init_dir, policy_path, workflow_path)


def update_victim_active_state(run_dir: Path, policy_path: Path | None, workflow_path: Path | None) -> None:
    """Overwrite <run_dir>/victim-active-state/ with the latest victim agent files."""
    active_dir = run_dir / _VICTIM_ACTIVE_STATE
    active_dir.mkdir(parents=True, exist_ok=True)
    _copy_victim_files(active_dir, policy_path, workflow_path)


def snapshot_victim_state(iteration_dir: Path, policy_path: Path | None, workflow_path: Path | None) -> None:
    """Copy victim agent files into <iteration_dir>/victim-state/ as a point-in-time snapshot."""
    state_dir = iteration_dir / _VICTIM_STATE
    state_dir.mkdir(parents=True, exist_ok=True)
    _copy_victim_files(state_dir, policy_path, workflow_path)


def _copy_victim_files(dest: Path, policy_path: Path | None, workflow_path: Path | None) -> None:
    if policy_path and policy_path.exists():
        dst = dest / policy_path.name
        if policy_path.resolve() != dst.resolve():
            shutil.copy2(policy_path, dst)
    if workflow_path and workflow_path.exists():
        dst = dest / workflow_path.name
        if workflow_path.resolve() != dst.resolve():
            shutil.copy2(workflow_path, dst)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Swarm Tracker — run-log directory layout, component writers, and run-level logging.

All path construction under run_logs/ lives here. Nothing outside this package
should build paths into run_logs/ directly.
"""

from agent_hardener.swarm_tracker.layout import (
    GARAK,
    OPENSHELL_LOGS,
    ArtifactKind,
    RunLayout,
    Scope,
    agent_fingerprint_path,
    benign_profiles_dir,
    create_iteration_dir,
    create_round_dir,
    create_run_dir,
    init_dir,
    run_logs_root,
    snapshot_victim_state,
    update_victim_active_state,
    victim_active_state_dir,
    victim_image_path,
    write_init_files,
)
from agent_hardener.swarm_tracker.writers import write_component_output, write_run_config

__all__ = [
    "GARAK",
    "OPENSHELL_LOGS",
    "ArtifactKind",
    "RunLayout",
    "Scope",
    "agent_fingerprint_path",
    "benign_profiles_dir",
    "create_iteration_dir",
    "create_round_dir",
    "create_run_dir",
    "init_dir",
    "run_logs_root",
    "snapshot_victim_state",
    "update_victim_active_state",
    "victim_active_state_dir",
    "victim_image_path",
    "write_component_output",
    "write_init_files",
    "write_run_config",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``agent-hardener`` command-line interface.

A single entrypoint for running Agent Hardener against a user's agent. ``run`` is the bash script
replacement: expand the manifest, bring up host backends, build + harden the agent in OpenShell, and
clean up (the garak agent_breaker attacker spawns the garak CLI itself). ``init`` scaffolds a manifest
plus a ``garak-scan.yaml`` config; ``up``/``down``/``status`` expose the sandbox lifecycle, and ``stage`` writes
the victim build context for hosts that build the image elsewhere and pass it back with ``--victim-image``.

One module per command lives beside this file; importing them here registers each on the shared ``app``.
The import order fixes the order commands appear in ``--help``.
"""

from __future__ import annotations

from agent_hardener.cli._app import app
from agent_hardener.cli import run, synth_benign, serve, lifecycle, setup, init, inspect

__all__ = ["app", "init", "inspect", "lifecycle", "run", "serve", "setup", "synth_benign"]

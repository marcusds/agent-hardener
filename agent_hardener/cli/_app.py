# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared Typer app and CLI-wide defaults.

Command modules import ``app`` from here and register themselves with ``@app.command(...)``; the package
``__init__`` imports those modules so registration happens on import. Kept separate from ``__init__`` to
avoid a circular import (``__init__`` → command modules → ``app``).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import typer

# The LLM/agent stack emits many cosmetic third-party warnings (Pydantic serializer warnings,
# ChatNVIDIA structured-output notices, deprecations). They're noise for CLI users, so silence
# all warnings for every command. (The synth/replay paths re-apply this inside a context manager
# too, since some dependencies reset warnings.filters mid-run.)
warnings.filterwarnings("ignore")

app = typer.Typer(
    help="Agent Hardener — run a security war-game against your NAT agent.",
    no_args_is_help=True,
    add_completion=False,  # hide the built-in --install-completion/--show-completion options
)

DEFAULT_MANIFEST = Path("agent-hardener.yaml")
DEFAULT_ENV_FILE = Path(".env")
VICTIM_IMAGE_HELP = (
    "Pre-built victim image to run instead of building locally with docker. Build it from the context "
    "`agent-hardener stage` writes."
)

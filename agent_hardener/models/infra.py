# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Victim-environment and external-tool infrastructure specs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from agent_hardener.models.base import AgentHardenerModel, VictimControlType


class VictimControlConfig(AgentHardenerModel):
    """Configuration for applying defender output to the victim environment."""

    type: VictimControlType = "file"
    config: dict[str, Any] = Field(default_factory=dict)


class BackendEndpoint(AgentHardenerModel):
    """A single endpoint the sandboxed agent is allowed to reach.

    Backends run on the host by default, reached via ``host.docker.internal``; ``host`` points at one
    running elsewhere. These entries are injected into the sandbox network policy.
    """

    host: str = "host.docker.internal"
    port: int = Field(gt=0, le=65535)
    # OpenShell's policy validator only accepts these protocols. ``tcp`` is L4-only (databases,
    # brokers): no ``access``/``enforcement`` is emitted for it.
    protocol: Literal["rest", "websocket", "graphql", "mcp", "json-rpc", "tcp"] = "rest"
    enforcement: str = "enforce"
    access: str = "full"
    allowed_ips: list[str] = Field(default_factory=list)


class BackendServiceSpec(AgentHardenerModel):
    """A host-side docker-compose backend that the NAT agent's tools call."""

    name: str = Field(min_length=1)
    compose_file: Path | None = None
    compose_project: str | None = None
    health_url: str | None = None
    allowlist: list[BackendEndpoint] = Field(default_factory=list)


class RelayVictimSpec(AgentHardenerModel):
    """How to build and serve the user's Relay-connected agent as the OpenShell victim.

    The user owns the image. Agent Hardener hardens the agent they actually ship, so it does not build
    one for them: ``dockerfile`` is required, and ``victim_binaries`` must scope egress because the
    image's venv layout is unknown to us.

    ``relay_artifacts`` is where the agent writes ``events.atof.jsonl``. Agent Hardener reads it to
    prove the victim is instrumented at all (:mod:`agent_hardener.preflight.relay`).
    """

    project_dir: Path
    dockerfile: Path
    #: Carried from the manifest so the build can stage harness-specific wiring (Hermes needs two
    #: extra Dockerfile lines) and the preflight can name what it could not enforce.
    harness: str | None = None
    relay_artifacts: Path = Path("artifacts/relay")
    victim_port: int = Field(default=8000, gt=0, le=65535)
    victim_binaries: list[str] = Field(min_length=1)
    agent_env: dict[str, str] = Field(default_factory=dict)
    #: The dotenv file the OpenShell provider reads; never copied into the staged build context.
    secrets_file: Path | None = None
    backends: list[BackendServiceSpec] = Field(default_factory=list)
    egress: list[str] = Field(default_factory=list)
    discover_egress: bool = True

    @property
    def atof_path(self) -> Path:
        """Where the victim's ATOF event log lands, relative to the sandbox working directory."""
        return self.relay_artifacts / "events.atof.jsonl"


class GarakSettings(AgentHardenerModel):
    """How agent-hardener drives garak's agent_breaker probe via the garak CLI.

    The probe is run as a subprocess against a dynamically rendered config; only the victim
    endpoint and reporting destination are run-specific. ``target_uri`` / ``target_port``
    override the manifest-derived victim endpoint; the model fields override the baked-in
    red-team / detector defaults (each also honours ``GARAK_<KEY>`` env vars).
    """

    config_path: str = "garak-scan.yaml"
    # Optional explicit override for garak's output dir. Unset (the default) means the attacker writes to
    # the ready ``artifact_dir`` the stage injects (its run-scoped garak dir); set to force a fixed path.
    report_dir: str | None = None
    report_prefix: str = "agent-breaker"
    target_uri: str | None = None
    target_port: int | None = None
    red_team_model_type: str | None = None
    red_team_model_name: str | None = None
    red_team_model_uri: str | None = None
    detector_model_type: str | None = None
    detector_model_name: str | None = None
    detector_model_uri: str | None = None
    max_attempts_per_tool: int | None = None
    #: Cap on a single attack turn, seconds. Raise for an agent whose turns are long —
    #: a sub-agent tool can push one turn past the default.
    request_timeout: int | None = None
    generations: int | None = None
    # Hard cap (seconds) on the attacker's own execution. Heavy agents (many tools / long LLM chains)
    # can need well over the default; raise this rather than let a scan be killed mid-probe (which yields
    # a spurious "0 hits"). Unset -> the default in ``_default_attackers``.
    timeout_s: int | None = None


class DefenderSettings(AgentHardenerModel):
    """Per-run overrides for the defenders, mirroring :class:`GarakSettings` for the attacker."""

    # Hard cap (seconds) on each defender's execution. The guardrails defender validates its draft
    # by invoking the victim once per augmented attack and per benign request, so a slow victim — a
    # Fabric agent spawns an MCP subprocess and makes an LLM call per tool — can exceed the default
    # mid-refinement, which discards a guardrail that was nearly ready. Unset -> the built-in default.
    timeout_s: int | None = None


class PreloadedAttackConfig(AgentHardenerModel):
    """Configuration for loading existing attack hit records instead of running an attacker."""

    name: str = Field(min_length=1)
    path: Path
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

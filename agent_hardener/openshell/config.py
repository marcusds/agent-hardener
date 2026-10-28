# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenShell lifecycle value types: the command-runner seam, config, and result.

Pure data + the subprocess-backed runner, with no dependency on the lifecycle orchestration —
so both :mod:`agent_hardener.openshell.command` (the executor) and :mod:`agent_hardener.openshell.lifecycle`
can build on them without a cycle.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from agent_hardener.models import RelayVictimSpec


class CommandRunner(Protocol):
    """Command execution contract for lifecycle operations."""

    def run(self, args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        """Run a command and return the completed process."""


class SubprocessCommandRunner:
    """Subprocess-backed command runner."""

    def run(self, args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        """Run a command and capture output."""
        return subprocess.run(  # noqa: S603 - command arguments are explicit lists built by lifecycle callers.
            args,
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )


#: The sandbox's canonical main process (see ``OpenShellConfig.create_command``).
DEFAULT_CREATE_COMMAND = ("sleep", "infinity")


@dataclass(frozen=True)
class OpenShellConfig:
    """OpenShell gateway/sandbox settings."""

    gateway: str
    sandbox: str
    policy_path: Path
    openshell_bin: str = "openshell"
    build_context: Path | None = None
    image: str | None = None
    provider: str | None = None
    provider_type: str = "generic"
    provider_env_file: Path | None = None
    provider_credentials: list[str] = field(default_factory=list)
    forward: str | None = None
    start_command: str | None = None
    # OpenShell makes this the sandbox's canonical main process: when it exits the sandbox goes
    # ``Completed`` and exec/upload stop working. It must therefore outlive the run; the victim itself
    # is started separately with ``sandbox exec`` once the uploads are in place.
    create_command: list[str] = field(default_factory=lambda: list(DEFAULT_CREATE_COMMAND))
    uploads: list[str] = field(default_factory=list)
    health_url: str | None = None
    health_timeout: float = 120.0
    sandbox_timeout: float = 180.0
    policy_wait_timeout: float = 60.0
    cwd: Path | None = None
    relay_victim: RelayVictimSpec | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> OpenShellConfig:
        """Build config from a YAML-friendly mapping."""
        from agent_hardener.models import RelayVictimSpec  # noqa: PLC0415 - avoid import cycle at module load.

        policy_path = Path(str(data["policy_path"]))
        build_context = Path(str(data["build_context"])) if data.get("build_context") else None
        cwd = Path(str(data["cwd"])) if data.get("cwd") else None
        uploads_value = data.get("uploads") or []
        uploads = [uploads_value] if isinstance(uploads_value, str) else list(uploads_value)
        relay_victim = RelayVictimSpec.model_validate(data["relay_victim"]) if data.get("relay_victim") else None
        return cls(
            gateway=str(data.get("gateway", "auto-defender")),
            sandbox=str(data.get("sandbox", "agents-lab-managed")),
            policy_path=policy_path,
            openshell_bin=str(data.get("openshell_bin", "openshell")),
            build_context=build_context,
            image=str(data["image"]) if data.get("image") else None,
            provider=str(data["provider"]) if data.get("provider") else None,
            provider_type=str(data.get("provider_type", "generic")),
            provider_env_file=Path(str(data["provider_env_file"])) if data.get("provider_env_file") else None,
            provider_credentials=[str(value) for value in data.get("provider_credentials", [])],
            forward=str(data["forward"]) if data.get("forward") else None,
            start_command=str(data["start_command"]) if data.get("start_command") else None,
            create_command=_create_command_from_value(data.get("create_command")),
            uploads=[str(value) for value in uploads],
            health_url=str(data["health_url"]) if data.get("health_url") else None,
            health_timeout=float(data.get("health_timeout", 120.0)),
            sandbox_timeout=float(data.get("sandbox_timeout", 180.0)),
            policy_wait_timeout=float(data.get("policy_wait_timeout", 60.0)),
            cwd=cwd,
            relay_victim=relay_victim,
        )


@dataclass(frozen=True)
class OpenShellCommandResult:
    """Result from an OpenShell lifecycle command."""

    ok: bool
    command: list[str]
    output: str

    @classmethod
    def from_completed(cls, command: list[str], completed: subprocess.CompletedProcess[str]) -> OpenShellCommandResult:
        """Adapt a finished subprocess into a result (non-zero exit → not ok; ``None`` stdout → empty string)."""
        return cls(ok=completed.returncode == 0, command=command, output=completed.stdout or "")


def _create_command_from_value(value: object) -> list[str]:
    if value is None or value == "":
        return list(DEFAULT_CREATE_COMMAND)
    if isinstance(value, str):
        command = shlex.split(value)
    elif isinstance(value, list):
        command = [str(part) for part in value]
    else:
        raise ValueError("openshell create_command must be a string or list")
    if not command:
        raise ValueError("openshell create_command must not be empty")
    return command

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``up``/``down``/``status``/``stage`` commands: the OpenShell sandbox lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer

from agent_hardener.cli._app import DEFAULT_ENV_FILE, DEFAULT_MANIFEST, VICTIM_IMAGE_OPTION, app
from agent_hardener.cli._errors import cli_error_boundary
from agent_hardener.display.console import emit, status_factory
from agent_hardener.errors import SandboxProvisioningError, VictimBuildError

if TYPE_CHECKING:
    from agent_hardener.openshell.config import OpenShellCommandResult


def _report(result: OpenShellCommandResult) -> None:
    """Emit a lifecycle result's output; raise a classified error when it failed (else exit 0).

    A lifecycle failure surfaces as ``ok=False`` data, not an exception — raise
    :class:`SandboxProvisioningError` so the surrounding :func:`cli_error_boundary` serializes it
    for the caller (rather than exiting with a bare non-zero code).
    """
    emit(result.output)
    if not result.ok:
        raise SandboxProvisioningError("sandbox lifecycle command failed; see the output above")
    raise typer.Exit(0)


def _simple_lifecycle(config: Path, verb: str) -> None:
    """Run a no-build lifecycle verb (``down``/``status``) against the configured sandbox."""
    from agent_hardener.config import load_config
    from agent_hardener.openshell.lifecycle import OpenShellLifecycle
    from agent_hardener.tools.openshell import openshell_config

    with cli_error_boundary():
        lifecycle = OpenShellLifecycle(openshell_config(load_config(str(config)).victim_control))
        _report(getattr(lifecycle, verb)())


@app.command()
def up(
    config: Path = typer.Option(DEFAULT_MANIFEST, "--config", "-c"),
    env_file: Path = typer.Option(DEFAULT_ENV_FILE, "--env-file"),
    victim_image: str | None = VICTIM_IMAGE_OPTION,
) -> None:
    """Build and start the OpenShell sandbox for the configured agent."""
    from agent_hardener.config import load_config
    from agent_hardener.openshell.lifecycle import OpenShellLifecycle
    from agent_hardener.tools.openshell import (
        configure_local_docker_host,
        load_local_env,
        openshell_config,
        prepare_relay_victim,
        with_victim_image,
    )

    with cli_error_boundary():
        configure_local_docker_host()
        load_local_env(env_file)
        victim_control = with_victim_image(load_config(str(config)).victim_control, victim_image)
        config_obj = prepare_relay_victim(openshell_config(victim_control))
        with status_factory("Building and starting sandbox"):
            result = OpenShellLifecycle(config_obj).up()
        _report(result)


@app.command()
def down(config: Path = typer.Option(DEFAULT_MANIFEST, "--config", "-c")) -> None:
    """Tear down the OpenShell sandbox."""
    _simple_lifecycle(config, "down")


@app.command()
def status(config: Path = typer.Option(DEFAULT_MANIFEST, "--config", "-c")) -> None:
    """Show OpenShell sandbox status."""
    _simple_lifecycle(config, "status")


@app.command()
def stage(config: Path = typer.Option(DEFAULT_MANIFEST, "--config", "-c")) -> None:
    """Write the victim's staged build context and print its directory.

    Build and push that directory with any image builder, then pass the image to ``run --victim-image``.
    """
    from agent_hardener.config import load_config
    from agent_hardener.tools.openshell import openshell_config, prepare_relay_victim

    with cli_error_boundary():
        prepared = prepare_relay_victim(openshell_config(load_config(str(config)).victim_control))
        if prepared.relay_victim is None or prepared.build_context is None:
            raise VictimBuildError(
                "the config declares no victim build to stage",
                remediation="Point --config at an agent-hardener.yaml manifest with an agent.dockerfile.",
            )
        typer.echo(str(prepared.build_context.parent))

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``synth-benign`` command: synthesize a smart-benign profile + request suite for the victim."""

from __future__ import annotations

from pathlib import Path

import typer

from agent_hardener.cli._app import DEFAULT_ENV_FILE, DEFAULT_MANIFEST, VICTIM_IMAGE_OPTION, app
from agent_hardener.display.console import fail, info, section, status_factory
from agent_hardener.display.interview import auto_answer_provider, tty_answer_provider


@app.command(name="synth-benign")
def synth_benign(
    config: Path = typer.Option(DEFAULT_MANIFEST, "--config", "-c", help="Manifest or full session config."),
    env_file: Path = typer.Option(DEFAULT_ENV_FILE, "--env-file", help="Dotenv file with secrets."),
    no_interactive: bool = typer.Option(
        False, "--no-interactive", help="Skip the interview entirely (leaner, rules-only suite; use in CI)."
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Run the interview but auto-accept each question's recommended default (no prompts).",
    ),
    validator: str | None = typer.Option(
        None, "--validator", help="benign_validator entry to use when the config declares multiple."
    ),
    upload_dataset: bool = typer.Option(
        False, "--upload-dataset", help="Upload generated requests to a Langfuse dataset (requires LANGFUSE_ENABLED=1)."
    ),
    reuse: bool = typer.Option(
        False, "--reuse", help="Reuse the running sandbox if it's already Ready (skip rebuild)."
    ),
    no_cleanup: bool = typer.Option(False, "--no-cleanup", help="Leave the sandbox running after synthesis."),
    victim_image: str | None = VICTIM_IMAGE_OPTION,
) -> None:
    """Bring up the victim sandbox, synthesize the smart benign profile + request suite, then tear it down.

    Self-contained: synthesis probes the live victim, so this builds the sandbox first and cleans it up
    after. Use ``--reuse`` to reuse an already-Ready sandbox and ``--no-cleanup`` to leave it running.
    Writes profile.json + requests.csv under <storage.root_dir>/benign_profiles/<target>/. The same
    synthesis runs automatically as a pre-flight phase of ``agent-hardener run``; use this to generate,
    iterate on, or inspect a suite standalone.
    """
    from contextlib import ExitStack

    from agent_hardener.agents.validators.smart_benign.wizard import run_synth_wizard
    from agent_hardener.cli._errors import cli_error_boundary
    from agent_hardener.config import load_config
    from agent_hardener.model_check import preflight_configured_models
    from agent_hardener.tools.openshell import (
        configure_local_docker_host,
        load_local_env,
        managed_victim_sandbox,
        with_victim_image,
    )

    if yes and no_interactive:
        fail("--yes and --no-interactive are mutually exclusive")
        raise typer.Exit(1)

    with cli_error_boundary():
        section("Agent Hardener benign-suite synthesis")
        configure_local_docker_host()
        load_local_env(env_file)
        # Synth uses the analysis LLM; fail fast on a missing credential before the (slow) sandbox build.
        preflight_configured_models()
        session = load_config(str(config))
        victim_control = with_victim_image(session.victim_control, victim_image)
        answer_provider = auto_answer_provider if yes else tty_answer_provider
        # The sandbox build (docker) can take minutes; show a spinner so the command isn't silent.
        with ExitStack() as sandbox:
            with status_factory("Building and starting victim sandbox"):
                sandbox.enter_context(
                    managed_victim_sandbox(
                        victim_control, reuse=reuse, no_cleanup=no_cleanup, base_url=session.target.base_url
                    )
                )
            info("Sandbox ready — starting synthesis")
            code = run_synth_wizard(
                config,
                answer_provider=answer_provider,
                no_interactive=no_interactive,
                yes=yes,
                validator=validator,
                upload_dataset=upload_dataset,
            )
        raise typer.Exit(code)

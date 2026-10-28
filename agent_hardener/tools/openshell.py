# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenShell lifecycle CLI for Agent Hardener."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import shlex
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import yaml  # type: ignore[import-untyped]

from agent_hardener.agents.defenders.openshell_defender_v2.policy.schema import parse_policy
from agent_hardener.config import load_config
from agent_hardener.openshell.egress import (
    CodeEgressSource,
    DockerfileEnvEgressSource,
    EgressDiscoverer,
    EgressSource,
    ManualEgressSource,
)
from agent_hardener.openshell.lifecycle import OpenShellConfig, OpenShellLifecycle, read_env_file, wait_for_http_health
from agent_hardener.openshell.policy_egress import (
    apply_victim_binaries,
    bind_provider_credentials,
    inject_backend_egress,
    inject_discovered_egress,
    normalize_host_gateway_allowed_ips,
    reconcile_shared_endpoint_settings,
)
from agent_hardener.openshell.relay_victim import dockerfile_env, stage_relay_victim_build
from agent_hardener.preflight.relay import RELAY_PROBE_PROMPT, RELAY_PROBE_TIMEOUT_SECONDS, check_relay_instrumentation
from agent_hardener.relay_plugin.config import judge_endpoint
from agent_hardener.relay_plugin.victim import ARTIFACTS_ENV, SANDBOX_ARTIFACTS_DIR

# Commands that build/launch the sandbox and therefore need the victim build staged.
_BUILD_COMMANDS = {"up", "restart", "cycle-reset"}

if TYPE_CHECKING:
    from collections.abc import Iterator

    from agent_hardener.models import RelayVictimSpec, VictimControlConfig


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Manage the configured OpenShell victim.")
    parser.add_argument(
        "command",
        choices=["up", "down", "restart", "status", "ensure-provider", "apply-policy", "cycle-reset"],
    )
    parser.add_argument("--config", required=True, help="Agent Hardener YAML/JSON config")
    parser.add_argument("--policy", help="Policy path for apply-policy")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the OpenShell lifecycle CLI."""
    args = build_arg_parser().parse_args(argv)
    session_config = load_config(args.config)
    base_config = openshell_config(session_config.victim_control)
    if args.command in _BUILD_COMMANDS:
        base_config = prepare_relay_victim(base_config)
    lifecycle = OpenShellLifecycle(base_config)

    if args.command == "up":
        result = lifecycle.up()
    elif args.command == "down":
        result = lifecycle.down()
    elif args.command in {"restart", "cycle-reset"}:
        result = lifecycle.restart()
    elif args.command == "status":
        result = lifecycle.status()
    elif args.command == "ensure-provider":
        result = lifecycle.ensure_provider()
    else:
        if not args.policy:
            parser = build_arg_parser()
            parser.error("--policy is required for apply-policy")
        result = lifecycle.apply_policy(Path(args.policy))

    print(result.output)
    return 0 if result.ok else 1


def configure_local_docker_host() -> None:
    """Use the common Colima Docker socket when DOCKER_HOST is unset."""
    if os.environ.get("DOCKER_HOST"):
        return
    colima_socket = Path.home() / ".colima/default/docker.sock"
    if colima_socket.exists():
        os.environ["DOCKER_HOST"] = f"unix://{colima_socket}"


def load_local_env(path: Path | str | None) -> None:
    """Load dotenv values into this process without printing secrets."""
    env_path = Path(path) if path else None
    if env_path is None or not env_path.exists():
        return
    for key, value in read_env_file(env_path).items():
        os.environ.setdefault(key, value)


def target_health_url(base_url: str | None) -> str | None:
    """Convert a target base URL to the NAT health endpoint."""
    if not base_url:
        return None
    parts = urlsplit(base_url)
    return urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))


def wait_for_health(url: str, timeout_seconds: float) -> None:
    """Wait until the victim health endpoint responds."""
    wait_for_http_health(url, timeout_seconds)


def openshell_config(config: VictimControlConfig) -> OpenShellConfig:
    """Extract OpenShell config from victim-control config."""
    if config.type != "openshell":
        raise SystemExit("config victim_control.type must be 'openshell'")
    return OpenShellConfig.from_mapping(config.config)


def with_victim_image(config: VictimControlConfig, image: str | None) -> VictimControlConfig:
    """Point the sandbox at a pre-built victim image (one built from the ``agent-hardener stage`` context)."""
    if image is None:
        return config
    return config.model_copy(update={"config": {**config.config, "image": image}})


def sandbox_is_ready(config: OpenShellConfig) -> bool:
    """Return True when the configured sandbox already exists and is ``Ready``.

    The ``--reuse`` fast path uses this to skip the loop-1 rebuild when a usable sandbox is already
    running; otherwise the run builds fresh. Mirrors the cycle script's ``sandbox get`` phase check.
    """
    status = OpenShellLifecycle(config).status()
    return status.ok and "Ready" in status.output


def prepare_relay_victim(config: OpenShellConfig) -> OpenShellConfig:
    """Stage the victim build and patch the policy with backend + auto-discovered egress.

    Returns ``config`` unchanged when no ``relay_victim`` spec is present. Otherwise stages the build
    context under ``<cwd>/.agent-hardener/builds/<sandbox>/`` and, when backends declare an allowlist or
    egress is discovered from the agent's workflow/code, writes a policy with the
    ``relay_victim_backends`` and/or ``relay_victim_discovered_egress`` blocks injected.
    """
    spec = config.relay_victim
    if spec is None:
        return config
    cwd = config.cwd or Path.cwd()
    build_root = cwd / ".agent-hardener" / "builds" / config.sandbox
    staged_dockerfile = stage_relay_victim_build(spec, cwd, build_root)

    project_dir = spec.project_dir if spec.project_dir.is_absolute() else cwd / spec.project_dir
    project_dir = project_dir.resolve()
    # Agent Hardener's own injected dependency, allow-listed unconditionally: the guardrail plugin runs
    # a safety judge *inside* the victim, against the same analysis endpoint the rest of the run
    # uses. Discovery cannot find it — it is in no file the user wrote — and the sandbox is
    # default-deny, so without this the judge's call is dropped, the guardrail fails open, and the
    # run reports an attack landing against a guardrail that was installed and never consulted.
    sources: list[EgressSource] = [
        ManualEgressSource([*spec.egress, judge_endpoint()]),
        # The composed Dockerfile carries the author's ENV plus the manifest's agent.env, so it is
        # where a backend URL the agent was configured with actually appears.
        DockerfileEnvEgressSource(staged_dockerfile),
    ]
    if spec.discover_egress:
        sources.insert(0, CodeEgressSource(project_dir))
    discovered = EgressDiscoverer(sources).policy_endpoints()

    backend_endpoints = [endpoint for backend in spec.backends for endpoint in backend.allowlist]
    dest = cwd / ".agent-hardener" / "policies" / f"{config.sandbox}.yaml"
    inject_backend_egress(config.policy_path, backend_endpoints, spec.victim_binaries, dest)
    inject_discovered_egress(dest, discovered, spec.victim_binaries, dest)
    policy = yaml.safe_load(dest.read_text(encoding="utf-8")) or {}
    # The template's own blocks name the NAT-era interpreter path, which grants a Fabric victim
    # nothing; retarget them at this victim before anything is uploaded.
    apply_victim_binaries(policy, spec.victim_binaries)
    # host.docker.internal endpoints (declared backends reach the host via it) carry a
    # Docker-Desktop-specific allowed_ips default; rewrite it to this daemon's actual bridge IP.
    normalize_host_gateway_allowed_ips(policy)
    reconcile_shared_endpoint_settings(policy)
    if config.provider:
        bind_provider_credentials(policy, config.provider)
    # Fail here, naming the field, rather than at `sandbox create`: OpenShell rejects unknown
    # fields and inconsistent endpoints outright, and this is the policy every defender starts from.
    parse_policy(policy)
    dest.write_text(yaml.safe_dump(policy, sort_keys=False, default_flow_style=False), encoding="utf-8")
    policy_path = dest

    # BYO images get the proxy shim and the guardrail plugin via COPY, but `openshell exec` drops
    # image ENV — so the launch command must set both itself.
    #
    # The victim writes ATOF to a fixed path under /home/sandbox, not to the manifest's
    # `agent.relay_artifacts`. Two reasons: every policy template already grants that path
    # read-write, and the sandbox shares no filesystem with the host anyway — the manifest path is
    # where Agent Hardener *lands* the file after pulling it out (see OpenShellLifecycle.fetch_file), so
    # the two are different sides of a copy rather than one shared location.
    start_command = config.start_command
    if spec.dockerfile is not None and start_command:
        start_command = f"env {_launch_env(spec, dockerfile_env(staged_dockerfile))} {start_command}"

    return replace(config, build_context=staged_dockerfile, policy_path=policy_path, start_command=start_command)


#: Prepended by Agent Hardener rather than replaced, because the image may set them too.
_ADDITIVE_ENV = {"PATH", "PYTHONPATH"}


def _launch_env(spec: RelayVictimSpec, image_env: dict[str, str]) -> str:
    """The environment the victim is started with, as an ``env`` prefix.

    OpenShell carries most of the image's ``ENV`` into the sandbox but **replaces ``PATH``** with its
    own default (verified by inspecting a sandbox container: ``AGENT_CONFIG_PATH``, ``PORT`` and
    ``VIRTUAL_ENV`` survive; ``PATH`` does not). So a victim runs with everything its author declared
    except the one variable that decides which interpreter a bare ``python`` resolves to.

    That one is enough to break things quietly. NeMo Fabric spawns its adapter host as a subprocess
    and resolves the interpreter by name, so without the venv on ``PATH`` it finds the system Python
    and dies with "No module named 'nemo_fabric_adapters'" — while the agent process itself looks
    perfectly healthy, answering ``/health`` and failing every actual request.

    Re-exporting the rest is defensive rather than required: it costs nothing and removes the need to
    track exactly which variables OpenShell chooses to keep.

    Layering, last wins:

    1. the image's ``ENV`` — the environment the agent was built to run in, which by then includes
       the manifest's ``agent.env``;
    2. Agent Hardener's own (the shim on ``PYTHONPATH``, the ATOF directory), prepended to any image
       value rather than replacing it.
    The manifest's ``agent.env`` needs no separate layer: it is baked into the composed Dockerfile,
    so it arrives as part of (1).
    """
    env = dict(image_env)
    env["PYTHONPATH"] = _prepend("/app/openshell-shims", image_env.get("PYTHONPATH"), "${PYTHONPATH:-}")
    bin_dirs = _victim_bin_dirs(spec.victim_binaries)
    if bin_dirs or "PATH" in image_env:
        fallback = "${PATH:-/usr/local/bin:/usr/bin:/bin}"
        env["PATH"] = _prepend(":".join(bin_dirs), image_env.get("PATH"), fallback)
    env[ARTIFACTS_ENV] = SANDBOX_ARTIFACTS_DIR
    # PATH/PYTHONPATH carry shell expansions on purpose; everything else is a literal value.
    return " ".join(
        f"{key}={value if key in _ADDITIVE_ENV else shlex.quote(value)}" for key, value in env.items() if value
    )


def _prepend(ours: str, from_image: str | None, fallback: str) -> str:
    """``ours`` in front of the image's value, or of *fallback* when the image declares none.

    Skipped when the image already leads with it — the Dockerfile carries our own additions (the shim
    on ``PYTHONPATH``, the venv on ``PATH``), so prepending unconditionally duplicates every entry.

    Falling back to the live variable rather than dropping it keeps whatever the sandbox itself
    provides — losing it would leave a victim with no ``PATH`` at all.
    """
    tail = from_image or fallback
    if not ours or tail == ours or tail.startswith(f"{ours}:"):
        return tail
    return f"{ours}:{tail}"


def _victim_bin_dirs(binaries: list[str]) -> list[str]:
    """The directories behind the victim's ``binaries`` globs, in order, de-duplicated.

    ``/workspace/.venv/bin/**`` describes which processes may egress; the directory it names is also
    where that victim's interpreter lives, so it is the right thing to put on PATH. System paths are
    skipped — they are already there, and prepending them would shadow the venv.
    """
    seen: list[str] = []
    for glob in binaries:
        directory = glob.split("*", 1)[0].rstrip("/")
        if directory and not directory.startswith("/usr/") and directory not in seen:
            seen.append(directory)
    return seen


def verify_relay_instrumentation(prepared: OpenShellConfig, *, base_url: str | None) -> None:
    """Send one benign probe and require the victim to emit ATOF, or fail.

    Shared by ``synth-benign`` and ``run`` so both bring-ups apply the same bar. ``base_url`` of
    ``None`` skips it: the caller has no endpoint to probe, which is the standalone-lifecycle case
    rather than an uninstrumented victim.
    """
    if base_url is None:
        return
    spec = prepared.relay_victim
    if spec is None:
        return
    cwd = prepared.cwd or Path.cwd()
    atof_path = cwd / spec.atof_path if not spec.atof_path.is_absolute() else spec.atof_path
    lifecycle = OpenShellLifecycle(prepared)

    async def probe() -> Any:
        async with httpx.AsyncClient(timeout=RELAY_PROBE_TIMEOUT_SECONDS) as client:
            return await client.post(
                base_url, json={"model": "victim", "messages": [{"role": "user", "content": RELAY_PROBE_PROMPT}]}
            )

    asyncio.run(
        check_relay_instrumentation(
            probe=probe,
            atof_path=atof_path,
            sync=lambda: lifecycle.fetch_file(f"{SANDBOX_ARTIFACTS_DIR}/{atof_path.name}", atof_path),
        )
    )


@contextlib.contextmanager
def managed_victim_sandbox(
    config: VictimControlConfig, *, reuse: bool, no_cleanup: bool, base_url: str | None = None
) -> Iterator[OpenShellConfig]:
    """Bring the victim sandbox up, verify its Relay instrumentation, yield, then tear it down.

    Mirrors ``run``'s lifecycle semantics: ``reuse`` skips the rebuild when a Ready sandbox is already
    running, and only a sandbox this call brought up is torn down on exit (``no_cleanup`` leaves it
    running). Lets standalone commands (e.g. ``synth-benign``) probe the live victim without the caller
    hand-orchestrating ``up`` / ``down``.

    The instrumentation check runs here rather than only in ``run`` because this is the *first* and
    slowest bring-up: failing at ``synth-benign`` costs one probe, while failing at ``run`` costs a
    docker build plus the whole benign interview first.
    """
    openshell = openshell_config(config)
    reused = reuse and sandbox_is_ready(openshell) and OpenShellLifecycle(openshell).runs_configured_image()
    prepared = prepare_relay_victim(openshell)
    if not reused:
        result = OpenShellLifecycle(prepared).up()
        if not result.ok:
            raise RuntimeError(result.output)
    try:
        verify_relay_instrumentation(prepared, base_url=base_url)
        yield openshell
    finally:
        if not no_cleanup and not reused:  # only tear down what we brought up
            OpenShellLifecycle(openshell).down()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

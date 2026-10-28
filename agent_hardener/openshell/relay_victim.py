# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage a build context for the user's Relay-connected victim image.

OpenShell builds the sandbox image from ``sandbox create --from <Dockerfile>`` using the
Dockerfile's parent directory as the build context, and cannot forward ``--build-arg``. So we stage
a build dir: copy the user's project into it and place their Dockerfile at its root.

Agent Hardener does not render a Dockerfile of its own. Tier 1 hardens the agent the user actually
ships, so building a *different* image would undercut the result — the image is theirs, and the only
thing we add is the egress shim the sandbox needs to route their traffic through its proxy.
"""

from __future__ import annotations

import re
import shlex
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from agent_hardener.errors import VictimBuildError

#: The guardrail package as it is named *inside the victim*. Copied in rather than pip-installed:
#: installing would put Agent Hardener's whole dependency set into the image we are supposed to be
#: hardening, and would need private package-index credentials inside the user's Docker build. The package
#: uses relative imports so it runs unchanged under this name.
GUARDRAIL_PACKAGE = "agent_hardener_guardrail"
_PLUGIN_SOURCE = Path(__file__).resolve().parent.parent / "relay_plugin"

if TYPE_CHECKING:
    from agent_hardener.models import RelayVictimSpec

# Excluded when copying the project into the staged build context (avoids huge/irrelevant trees).
# Dotenv files hold the agent's secrets, which reach the sandbox via the OpenShell provider instead; the
# staged context may be uploaded and built elsewhere (see ``agent-hardener stage``).
_COPY_IGNORE = shutil.ignore_patterns(".git", ".venv", "__pycache__", "*.pyc", ".agent-hardener", ".env", ".env.*")

# System-policy layer Relay discovers plugin config from. Agent Hardener uploads the guardrails the
# defenders produce here; the user/project layers are deliberately not used, because NeMo Fabric
# hard-rejects plugin config inherited from them (nemo_fabric_adapters/common/utils.py:286,312).
RELAY_PLUGINS_UPLOAD_DEST = "/etc/nemo-relay/plugins.toml"

# aiohttp egress shim, written directly into BYO build contexts (the generic Dockerfile bakes its own
# printf'd copy). Makes aiohttp honor OpenShell's injected HTTP(S)_PROXY so egress/DNS go through the
# policy-enforcing proxy. Auto-imported via `sitecustomize` on PYTHONPATH at interpreter startup.
_SHIM_PY = """\
# --- Agent Hardener: everything the war-game needs, with no code in the victim ---------------
try:
    import agent_hardener_guardrail.autopatch as _autopatch
    _autopatch.install()
except Exception:
    pass

# --- OpenShell: make aiohttp honour the sandbox proxy ------------------------------------
try:
    import functools, aiohttp
    _orig = aiohttp.ClientSession.__init__
    if not getattr(_orig, "_openshell_trust_env", False):
        @functools.wraps(_orig)
        def _init(self, *a, **kw):
            kw.setdefault("trust_env", True)
            _orig(self, *a, **kw)
        _init._openshell_trust_env = True
        aiohttp.ClientSession.__init__ = _init
except Exception:
    pass
"""


_CACHE_MOUNT = re.compile(r"--mount=type=cache,[^\s\\]*\s*")


def _strip_buildkit_only_syntax(staged_dockerfile: Path) -> None:
    """Remove BuildKit-only ``RUN --mount`` flags from the staged Dockerfile.

    OpenShell builds through the Docker Engine API's classic builder, which rejects them outright
    ("the --mount option requires BuildKit"). Nothing on our side can switch that builder on:
    DOCKER_BUILDKIT and an installed buildx affect the docker *CLI*, and OpenShell never invokes it.

    Safe to drop because a cache mount is a build-speed optimisation — the same layer builds without
    it, just slower. Done here rather than asked of the user because the Dockerfile is generated for
    them (``nemo agents package`` emits these mounts), so there is nothing for them to edit.
    """
    text = staged_dockerfile.read_text(encoding="utf-8")
    stripped = _CACHE_MOUNT.sub("", text)
    if stripped != text:
        staged_dockerfile.write_text(stripped, encoding="utf-8")


def dockerfile_env(dockerfile: Path) -> dict[str, str]:
    """The ``ENV`` declarations of a staged Dockerfile, in file order.

    Read from the Dockerfile rather than from the built image because it is available before the
    image exists — and because it is the author's own statement of how their agent expects to run,
    which is exactly what ``openshell sandbox exec`` discards.

    Values are returned verbatim, including references like ``$PATH``: they are re-exported through a
    shell, so the expansion the author wrote still happens. Malformed lines are skipped rather than
    raised on — a Dockerfile Docker itself accepts must not fail the run here.
    """
    env: dict[str, str] = {}
    for raw in _logical_lines(dockerfile.read_text(encoding="utf-8")):
        if not raw.upper().startswith("ENV "):
            continue
        try:
            tokens = shlex.split(raw[4:], posix=True)
        except ValueError:
            continue
        if not tokens:
            continue
        if "=" not in tokens[0]:
            # Legacy `ENV KEY value with spaces` — one variable, everything after it is the value.
            env[tokens[0]] = " ".join(tokens[1:])
            continue
        for token in tokens:
            key, _, value = token.partition("=")
            if key:
                env[key] = value
    return env


def _logical_lines(text: str) -> list[str]:
    """Dockerfile instructions with backslash continuations joined and comments dropped."""
    lines: list[str] = []
    buffer = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not buffer and (not stripped or stripped.startswith("#")):
            continue
        if stripped.endswith("\\"):
            buffer += stripped[:-1].rstrip() + " "
            continue
        lines.append((buffer + stripped).strip())
        buffer = ""
    if buffer:
        lines.append(buffer.strip())
    return lines


def _inject_agent_env(staged_dockerfile: Path, agent_env: dict[str, str]) -> None:
    """Bake the manifest's ``agent.env`` into the image as ``ENV`` lines.

    The composed Dockerfile then states the victim's whole runtime environment in one place: the
    author's own declarations plus whatever the manifest adds. Everything downstream reads it from
    there — the launcher, which re-exports it because OpenShell replaces ``PATH`` with its own, and
    egress discovery, which learns from it which hosts the agent will call.

    Secrets are not written here. They are named in ``agent.secrets`` and delivered by OpenShell's
    provider mechanism; this file is staged to disk and is part of the artifact a run hands back.
    """
    if not agent_env:
        return
    lines = "".join(f"ENV {key}={shlex.quote(value)}\n" for key, value in sorted(agent_env.items()))
    with staged_dockerfile.open("a", encoding="utf-8") as dockerfile:
        dockerfile.write("\n# --- Agent Hardener: the manifest's agent.env ---\n" + lines)


def _inject_shim_dir(build_root: Path, staged_dockerfile: Path) -> None:
    """Write ``sitecustomize.py`` into the context and COPY the shim directory into the image.

    One edit carries both shims — the guardrail registration and the egress proxy default — because
    they share a delivery mechanism: a ``sitecustomize`` on ``PYTHONPATH``, which CPython imports at
    interpreter startup before any user module.

    Uses COPY (not RUN) so it works even if the user's Dockerfile ends with a non-root ``USER`` —
    COPY runs as the build daemon. The launcher still prepends ``PYTHONPATH`` (see prepare_relay_victim),
    since OpenShell replaces the image's ``PATH`` with its own.
    """
    shim_dir = build_root / "openshell-shims"
    shim_dir.mkdir(exist_ok=True)
    (shim_dir / "sitecustomize.py").write_text(_SHIM_PY, encoding="utf-8")
    with staged_dockerfile.open("a", encoding="utf-8") as dockerfile:
        dockerfile.write(
            "\n# --- Agent Hardener: two shims, both auto-imported by sitecustomize on PYTHONPATH ---\n"
            "#   agent_hardener_guardrail/  the guardrail plugin. sitecustomize registers its kind with\n"
            "#                          Relay before the agent calls initialize() — the one part of\n"
            "#                          the war-game that cannot be delivered as configuration.\n"
            "#   aiohttp proxy default  ClientSession(trust_env=True), so the sandbox's egress proxy\n"
            "#                          is honoured. aiohttp ignores HTTP_PROXY otherwise, and the\n"
            "#                          victim's traffic would bypass the policy that scopes it.\n"
            "COPY openshell-shims/ /app/openshell-shims/\n"
            "ENV PYTHONPATH=/app/openshell-shims:${PYTHONPATH}\n"
        )


def _stage_guardrail_plugin(build_root: Path) -> None:
    """Copy the guardrail plugin into the shim directory the Dockerfile already COPYs and PYTHONPATHs.

    Riding the shim directory's ``COPY openshell-shims/`` means no second Dockerfile edit and no install
    step: the victim gets four Python files, not Agent Hardener's dependency tree. The victim imports it
    as ``agent_hardener_guardrail`` and calls ``connect_victim()``.
    """
    destination = build_root / "openshell-shims" / GUARDRAIL_PACKAGE
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(_PLUGIN_SOURCE, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


#: Hermes reads Relay plugin config from this env var and nowhere else — not from the ``/etc`` path
#: Relay itself discovers — so the upload destination has to be named to it explicitly. And its Relay
#: integration is an opt-in plugin: without ``plugins enable`` the env var is read by nobody.
_HERMES_WIRING = """
# --- Agent Hardener: Hermes reads plugin config only from this variable, and Relay is opt-in ---
ENV HERMES_NEMO_RELAY_PLUGINS_TOML={destination}
RUN hermes plugins enable observability/nemo_relay
"""


def _inject_hermes_wiring(staged_dockerfile: Path) -> None:
    """Point Hermes at the uploaded plugin config and turn its Relay plugin on.

    Only for ``harness: hermes``. Every other harness lets Relay discover ``/etc/nemo-relay`` on its
    own; Hermes does not look there, so without these two lines the guardrails are uploaded, parsed
    by nobody, and the victim answers every attack while the run reports a clean install.

    The agent config must also omit its ``telemetry:`` block — the Fabric adapter pops and overwrites
    this variable whenever Relay telemetry is declared. That half cannot be enforced from here; the
    manifest's ``harness`` is what lets the preflight say so.
    """
    with staged_dockerfile.open("a", encoding="utf-8") as dockerfile:
        dockerfile.write(_HERMES_WIRING.format(destination=RELAY_PLUGINS_UPLOAD_DEST))


def stage_relay_victim_build(spec: RelayVictimSpec, cwd: Path, build_root: Path) -> Path:
    """Stage the build context and return the path to the staged ``Dockerfile``.

    Copies ``spec.project_dir`` into ``build_root`` and writes a ``Dockerfile`` at its root:
    the rendered template (generic mode) or a copy of the project's own Dockerfile (BYO mode).

    Args:
        spec: The NAT victim specification.
        cwd: Base directory that relative paths in ``spec`` resolve against.
        build_root: Directory to stage into (created/replaced).

    Returns:
        Path to the staged ``Dockerfile`` (suitable for ``--from``).
    """
    project_dir = spec.project_dir if spec.project_dir.is_absolute() else cwd / spec.project_dir
    project_dir = project_dir.resolve()
    if not project_dir.is_dir():
        raise VictimBuildError(f"relay_victim.project_dir does not exist: {project_dir}")

    if build_root.exists():
        shutil.rmtree(build_root)
    shutil.copytree(project_dir, build_root, ignore=_COPY_IGNORE, symlinks=True)

    staged_dockerfile = build_root / "Dockerfile"
    user_dockerfile = spec.dockerfile if spec.dockerfile.is_absolute() else project_dir / spec.dockerfile
    if not user_dockerfile.is_file():
        raise VictimBuildError(f"relay_victim.dockerfile does not exist: {user_dockerfile}")
    shutil.copyfile(user_dockerfile, staged_dockerfile)
    _strip_buildkit_only_syntax(staged_dockerfile)
    _inject_agent_env(staged_dockerfile, spec.agent_env)
    _inject_shim_dir(build_root, staged_dockerfile)
    _stage_guardrail_plugin(build_root)
    if spec.harness == "hermes":
        _inject_hermes_wiring(staged_dockerfile)

    (build_root / ".dockerignore").write_text(".git\n.venv\n__pycache__\n*.pyc\n.env\n.env.*\n", encoding="utf-8")
    return staged_dockerfile

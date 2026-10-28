# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble a run's hardened image bundle — the artifact a user can actually deploy.

A run's guardrails are a ``plugins.toml``, and that file alone is not adoptable: applying it to an
agent that lacks the plugin implementing its kind does not warn, it raises
(``plugin component 'agent_hardener.pre_tool_verifier' is not registered``), so the agent will not start.

Adoption needs three things — the plugin code, a registration that runs before Relay's
``initialize()``, and the config — and a run already produces all three. They are simply in two
places: the staged build context (your Dockerfile plus the appended ``COPY``/``ENV`` and the shim)
and the run's final ``plugins.toml``. This module puts them together, so what leaves a war-game is a
buildable image rather than evidence.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

#: Where the bundle's Dockerfile bakes the guardrails. Matches the path Relay discovers as system
#: policy, so the image needs no runtime upload — unlike the war-game, where the file is delivered
#: per round.
BUNDLE_PLUGINS_DEST = "/etc/nemo-relay/plugins.toml"

#: The sandbox half of the staged shim. Excluded from the bundle: it defaults ``aiohttp`` sessions to
#: ``trust_env=True`` so OpenShell's egress proxy is honoured, which is a silent behaviour change in
#: an image that is no longer running inside OpenShell.
_SANDBOX_SHIM_MARKER = "# --- OpenShell:"

_DOCKERFILE_SUFFIX = """
# --- Agent Hardener: the guardrails this war-game produced -----------------------------------
# Baked rather than uploaded: the war-game delivers this file per round, but a shipped image
# should carry its own policy. Relay discovers this path as system-layer config.
COPY plugins.toml {destination}
"""

_STAGED_SOURCE = """\
Built from your own image plus the guardrails the war-game produced. Nothing here was invented: the
Dockerfile is yours with two lines appended, and `plugins.toml` is the config the run validated.
"""

_PREBUILT_SOURCE = """\
Built from `{image}`, the pre-built image this run validated, plus the guardrails the war-game
produced. The Dockerfile only layers `plugins.toml` onto that image, so the result is exactly what
was tested. The image still carries the OpenShell proxy shim it was staged with.
"""

_README = """\
# Hardened image — {agent}

{source}

```bash
docker build -t {agent}-hardened:{run_id} .
```

## Deploying it

**NeMo Platform** — same agent config, hardened image:

```bash
nemo agents deploy --agent {agent} --mode docker --image {agent}-hardened:{run_id}
```

`--mode docker` is required: `--image` is rejected in the default `subprocess` mode.

**Anywhere else** — run the image. The guardrails are baked in at `{destination}`.

## What it needs at runtime

Each guarded tool call is one LLM round trip to the safety judge, so the container needs
`INFERENCE_API_KEY` (or whatever `api_key_env` names in `plugins.toml`) and network reach to that
endpoint. **Without it the judge errors and every call is allowed** — the guardrail fails open, by
design, and says so only in the log.

## The two halves of the hardening

`plugins.toml` is the guardrails: which tools are guarded, and what counts as an attack. It travels
in the image and takes effect wherever the image runs.

`openshell-policy.yaml`, if present, is the other half — the sandbox policy the run hardened:
filesystem scope, and which processes may reach which hosts. It is **not** applied by `docker run`;
it is an OpenShell artifact, and running the image without it gives you a guarded agent in an
unconstrained sandbox. Apply it with `openshell sandbox create --policy openshell-policy.yaml`, or
port its allow-lists to whatever confines your production runtime.

## Editing the policy

`plugins.toml` is data. Change a threshold or an instruction, rebuild, redeploy — no code change.
Removing the `agent_hardener.pre_tool_verifier` component removes the guardrails; removing the plugin
package from the image while leaving the component in place will stop the agent from starting.

## One assumption this image makes

The guardrail registers through `PYTHONPATH`, declared as a Dockerfile `ENV`. Any runtime that
starts the container *without* honouring image `ENV` — as OpenShell's `sandbox exec` does — will
start this agent unguarded, and nothing will say so. If yours does that, export
`PYTHONPATH=/app/openshell-shims` explicitly.
"""


def write_hardened_bundle(
    *,
    build_root: Path | None,
    plugins_toml: Path,
    destination: Path,
    agent: str,
    run_id: str,
    policy: Path | None = None,
    image: str | None = None,
) -> Path:
    """Assemble the deployable bundle and return its directory.

    Args:
        build_root: The run's staged build context — the user's project plus the appended Dockerfile
            and ``openshell-shims/``. Ignored when ``image`` is given: that tree was re-staged for the
            run's policy but never built, so shipping it would package something the run did not test.
        plugins_toml: The run's final guardrail config.
        destination: Directory to write (replaced if it exists).
        agent: Agent name, for the image tag in the README.
        run_id: The run this came from, so a hardened image is traceable to the evidence for it.
        policy: The run's final OpenShell sandbox policy, when there is one. Shipped beside the
            image because a war-game hardens two things, not one — the guardrails *and* the policy —
            and handing back only the half that fits in the image would understate the result.
        image: The pre-built image the run validated (``--victim-image``), when there was one. The
            bundle's Dockerfile then starts ``FROM`` it instead of rebuilding the staged context.

    Returns:
        The bundle directory.
    """
    if destination.exists():
        shutil.rmtree(destination)
    if image is None:
        if build_root is None:
            raise ValueError("a hardened bundle needs a staged build context or a pre-built image")
        shutil.copytree(build_root, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        _strip_sandbox_shim(destination / "openshell-shims" / "sitecustomize.py")
        source = _STAGED_SOURCE
    else:
        destination.mkdir(parents=True)
        (destination / "Dockerfile").write_text(f"FROM {image}\n", encoding="utf-8")
        source = _PREBUILT_SOURCE.format(image=image)
    shutil.copyfile(plugins_toml, destination / "plugins.toml")

    if policy is not None and policy.is_file():
        shutil.copyfile(policy, destination / "openshell-policy.yaml")

    with (destination / "Dockerfile").open("a", encoding="utf-8") as dockerfile:
        dockerfile.write(_DOCKERFILE_SUFFIX.format(destination=BUNDLE_PLUGINS_DEST))
    (destination / "README.md").write_text(
        _README.format(agent=agent, run_id=run_id, destination=BUNDLE_PLUGINS_DEST, source=source),
        encoding="utf-8",
    )
    return destination


def _strip_sandbox_shim(sitecustomize: Path) -> None:
    """Drop the OpenShell half of the shim, keeping only the guardrail registration.

    The war-game image needs both; a shipped image needs one. Leaving the proxy shim in would change
    how the user's ``aiohttp`` sessions resolve proxies, in production, for no reason they asked for.
    """
    if not sitecustomize.is_file():
        return
    text = sitecustomize.read_text(encoding="utf-8")
    marker = text.find(_SANDBOX_SHIM_MARKER)
    if marker != -1:
        sitecustomize.write_text(text[:marker].rstrip() + "\n", encoding="utf-8")

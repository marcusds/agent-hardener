# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the hardened image bundle — the artifact a user deploys.

The bundle is the only run output that is *executable* rather than evidence, so what matters here is
that it carries the three things adoption needs (plugin, registration, config) and drops the one it
must not: the sandbox proxy shim, which would change how a production agent resolves proxies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agent_hardener.bundle import BUNDLE_PLUGINS_DEST, write_hardened_bundle

if TYPE_CHECKING:
    from pathlib import Path

_SITECUSTOMIZE = """\
# --- Agent Hardener: everything the war-game needs, with no code in the victim ---------------
try:
    import agent_hardener_guardrail.autopatch as _autopatch
    _autopatch.install()
except Exception:
    pass

# --- OpenShell: make aiohttp honour the sandbox proxy ------------------------------------
try:
    import aiohttp
except Exception:
    pass
"""


def _build_root(tmp_path: Path) -> Path:
    root = tmp_path / "build"
    (root / "openshell-shims" / "agent_hardener_guardrail").mkdir(parents=True, exist_ok=True)
    (root / "Dockerfile").write_text("FROM python:3.12-slim\nUSER agent\n", encoding="utf-8")
    (root / "openshell-shims" / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")
    (root / "openshell-shims" / "agent_hardener_guardrail" / "plugin.py").write_text("x = 1", encoding="utf-8")
    return root


def _bundle(tmp_path: Path, image: str | None = None) -> Path:
    plugins = tmp_path / "plugins.toml"
    plugins.write_text('version = 1\n[[components]]\nkind = "agent_hardener.pre_tool_verifier"\n', encoding="utf-8")
    policy = tmp_path / "openshell-policy.yaml"
    policy.write_text("version: 1\nfilesystem_policy:\n  read_only: [/usr]\n", encoding="utf-8")
    return write_hardened_bundle(
        build_root=_build_root(tmp_path),
        plugins_toml=plugins,
        destination=tmp_path / "hardened-image",
        agent="ledger",
        run_id="20260830T120000Z-abc",
        policy=policy,
        image=image,
    )


def test_bundle_carries_everything_adoption_needs(tmp_path: Path) -> None:
    """Plugin code, the registration that runs before initialize(), and the config it drives."""
    bundle = _bundle(tmp_path)

    assert (bundle / "openshell-shims" / "agent_hardener_guardrail" / "plugin.py").is_file()
    assert "_autopatch.install()" in (bundle / "openshell-shims" / "sitecustomize.py").read_text()
    assert "agent_hardener.pre_tool_verifier" in (bundle / "plugins.toml").read_text()


def test_the_guardrails_are_baked_not_uploaded(tmp_path: Path) -> None:
    """A shipped image must carry its own policy; only the war-game delivers it per round."""
    dockerfile = (_bundle(tmp_path) / "Dockerfile").read_text()

    assert dockerfile.startswith("FROM python:3.12-slim")  # the user's file, appended to
    assert f"COPY plugins.toml {BUNDLE_PLUGINS_DEST}" in dockerfile


def test_the_sandbox_proxy_shim_is_stripped(tmp_path: Path) -> None:
    """It is OpenShell-specific. Shipping it would silently change proxy resolution in production."""
    shim = (_bundle(tmp_path) / "openshell-shims" / "sitecustomize.py").read_text()

    assert "aiohttp" not in shim
    assert "_autopatch.install()" in shim  # the half that must survive


def test_rebuilding_replaces_a_previous_bundle(tmp_path: Path) -> None:
    """A stale file from an earlier run would ship guardrails the report never validated."""
    bundle = _bundle(tmp_path)
    (bundle / "stale.txt").write_text("from an earlier run", encoding="utf-8")

    assert not (_bundle(tmp_path) / "stale.txt").exists()


def test_the_sandbox_policy_ships_beside_the_image(tmp_path: Path) -> None:
    """A war-game hardens two things. Shipping only the half that fits in the image understates it."""
    bundle = _bundle(tmp_path)

    assert "filesystem_policy" in (bundle / "openshell-policy.yaml").read_text()
    # ...and the README has to say docker will not apply it.
    assert "not** applied by `docker run`" in (bundle / "README.md").read_text()


def test_a_pre_built_image_is_layered_not_rebuilt(tmp_path: Path) -> None:
    """The staged tree was never built when --victim-image ran; shipping it would package an untested image."""
    bundle = _bundle(tmp_path, image="registry.example.com/ledger:1")

    assert (bundle / "Dockerfile").read_text().startswith("FROM registry.example.com/ledger:1\n")
    assert f"COPY plugins.toml {BUNDLE_PLUGINS_DEST}" in (bundle / "Dockerfile").read_text()
    assert not (bundle / "openshell-shims").exists()
    assert "agent_hardener.pre_tool_verifier" in (bundle / "plugins.toml").read_text()
    assert "filesystem_policy" in (bundle / "openshell-policy.yaml").read_text()
    assert "Built from `registry.example.com/ledger:1`" in (bundle / "README.md").read_text()


def test_a_bundle_needs_a_build_root_or_an_image(tmp_path: Path) -> None:
    plugins = tmp_path / "plugins.toml"
    plugins.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="staged build context or a pre-built image"):
        write_hardened_bundle(
            build_root=None, plugins_toml=plugins, destination=tmp_path / "out", agent="ledger", run_id="r"
        )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the thin agent manifest and its expansion to a full SessionConfig."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_hardener.config import load_config
from agent_hardener.manifest import (
    DEFAULT_DEFENDER_TIMEOUT_SECONDS,
    AgentManifest,
    expand,
    load_manifest,
    with_session_reclaim,
)
from agent_hardener.models import SessionConfig
from agent_hardener.openshell.naming import sandbox_name
from agent_hardener.openshell.relay_victim import RELAY_PLUGINS_UPLOAD_DEST
from agent_hardener.tools.openshell import openshell_config

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]

AGENT = {
    "agent": {
        "name": "finance",
        "project_dir": "../agents-lab",
        "dockerfile": "deploy/finance/Dockerfile",
        "start_command": "/app/.venv/bin/python -m finance.serve",
        "binaries": ["/app/.venv/bin/**"],
        "port": 8000,
        "secrets": ["INFERENCE_API_KEY"],
        "env": {"FINANCE_BACKEND_URL": "http://host.docker.internal:8086"},
    },
    "backends": [
        {
            "name": "finance",
            "compose_file": "../agents-lab/services/finance_backend/docker-compose.yaml",
            "health_url": "http://127.0.0.1:8086/health",
            "ports": [8086],
        }
    ],
}


def test_expand_produces_valid_session_config() -> None:
    config = expand(AgentManifest.model_validate(AGENT), base_dir=REPO_ROOT)
    assert isinstance(config, SessionConfig)
    assert config.target.base_url == "http://127.0.0.1:8000/v1/chat/completions"
    assert [a.name for a in config.attackers] == ["garak-agent-breaker"]
    assert config.victim.name == "openshell-victim"
    assert config.attack_validators
    assert config.benign_validators


def test_default_attacker_timeout() -> None:
    config = expand(AgentManifest.model_validate(AGENT), base_dir=REPO_ROOT)
    attacker = config.attackers[0]
    assert attacker.config["timeout_s"] == 7200  # 2h default cap: the probe scales with tool count
    assert attacker.timeout_seconds == 7500  # wrapper sits above the attacker's own cap


def test_attacker_timeout_overridable_via_garak() -> None:
    manifest = {**AGENT, "garak": {"timeout_s": 600}}
    config = expand(AgentManifest.model_validate(manifest), base_dir=REPO_ROOT)
    attacker = config.attackers[0]
    assert attacker.config["timeout_s"] == 600
    assert attacker.timeout_seconds == 900


def test_expand_sets_swarm_tracker_storage() -> None:
    config = expand(AgentManifest.model_validate(AGENT), base_dir=REPO_ROOT)
    # run-logs land under <root_dir>/run-logs/; victim paths seed init/ + victim-active-state/.
    assert config.storage.root_dir == REPO_ROOT / ".agent-hardener"
    assert config.storage.victim_policy_path is not None
    assert config.storage.victim_policy_path.name == "openshell-policy-permissive.yaml"
    assert config.storage.victim_relay_plugins_path is not None
    assert config.storage.victim_relay_plugins_path.name == "plugins.toml"


def test_expand_relay_plugins_policy_overrides(tmp_path: Path) -> None:
    plugins = tmp_path / "hardened-plugins.toml"
    plugins.write_text("version = 1\n", encoding="utf-8")
    policy = tmp_path / "hardened-policy.yaml"
    policy.write_text("version: 1\n", encoding="utf-8")

    config = expand(AgentManifest.model_validate(AGENT), base_dir=REPO_ROOT, relay_plugins=plugins, policy=policy)

    # The override replaces the initial guardrail set / policy at every host-facing derived key.
    assert config.storage.victim_relay_plugins_path == plugins.resolve()
    assert config.target.agent_relay_plugins == plugins.resolve()
    assert config.storage.victim_policy_path == policy.resolve()
    assert config.victim_control.config["policy_path"] == str(policy.resolve())
    assert config.victim_control.config["uploads"] == [f"{plugins.resolve()}:{RELAY_PLUGINS_UPLOAD_DEST}"]


def test_guardrails_are_uploaded_to_the_one_layer_fabric_accepts() -> None:
    """NeMo Fabric hard-rejects plugin config inherited from the user or project layers."""
    config = expand(AgentManifest.model_validate(AGENT), base_dir=REPO_ROOT)
    (upload,) = config.victim_control.config["uploads"]
    assert upload.endswith(":/etc/nemo-relay/plugins.toml")


def test_expand_without_overrides_uses_manifest_defaults() -> None:
    config = expand(AgentManifest.model_validate(AGENT), base_dir=REPO_ROOT)
    assert config.storage.victim_policy_path is not None
    assert config.storage.victim_policy_path.name == "openshell-policy-permissive.yaml"


def test_expand_builds_relay_victim_and_backends() -> None:
    config = expand(AgentManifest.model_validate(AGENT), base_dir=REPO_ROOT)
    osc = openshell_config(config.victim_control)
    assert osc.sandbox == sandbox_name("finance")
    assert osc.provider_credentials == ["INFERENCE_API_KEY"]
    assert osc.forward == "0.0.0.0:8000"
    assert osc.relay_victim is not None
    assert osc.relay_victim.agent_env == {"FINANCE_BACKEND_URL": "http://host.docker.internal:8086"}
    assert [b.name for b in osc.relay_victim.backends] == ["finance"]
    assert osc.relay_victim.backends[0].allowlist[0].port == 8086


def test_expand_points_a_backend_at_a_non_host_address() -> None:
    backend = {"name": "finance", "ports": [8086], "host": "finance.ns.svc", "allowed_ips": ["10.96.0.0/12"]}
    config = expand(AgentManifest.model_validate({**AGENT, "backends": [backend]}), base_dir=REPO_ROOT)
    relay_victim = openshell_config(config.victim_control).relay_victim
    assert relay_victim is not None
    (endpoint,) = relay_victim.backends[0].allowlist
    assert (endpoint.host, endpoint.port, endpoint.allowed_ips) == ("finance.ns.svc", 8086, ["10.96.0.0/12"])


def test_expand_carries_the_secrets_file_so_staging_can_exclude_it() -> None:
    config = expand(AgentManifest.model_validate(AGENT), base_dir=REPO_ROOT)
    relay_victim = openshell_config(config.victim_control).relay_victim
    assert relay_victim is not None
    assert relay_victim.secrets_file == Path(".env")


@pytest.mark.parametrize(
    ("backend", "message"),
    [
        ({"name": "f", "ports": [8086], "host": "http://finance.ns.svc"}, "bare hostname"),
        ({"name": "f", "ports": [8086], "host": "finance.ns.svc:8086"}, "bare hostname"),
        ({"name": "f", "host": "finance.ns.svc"}, "no ports"),
        ({"name": "f", "allowed_ips": ["10.0.0.5/32"]}, "no ports"),
        (
            {"name": "f", "ports": [8086], "host": "finance.ns.svc", "compose_file": "x.yaml"},
            "both host and compose_file",
        ),
    ],
)
def test_backend_rejects_routes_that_would_silently_do_nothing(backend: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        AgentManifest.model_validate({**AGENT, "backends": [backend]})


def test_a_blank_backend_host_means_the_default() -> None:
    manifest = AgentManifest.model_validate({**AGENT, "backends": [{"name": "f", "ports": [8086], "host": "  "}]})
    assert manifest.backends[0].host is None


def test_expand_carries_the_users_image_and_binaries() -> None:
    raw = {
        "agent": {
            "name": "finance",
            "project_dir": "../agents-lab",
            "dockerfile": "deploy/finance/Dockerfile",
            "start_command": "/app/.venv/bin/python -m finance.serve",
            "binaries": ["/app/.venv/bin/**"],
            "secrets": ["INFERENCE_API_KEY"],
        }
    }
    config = expand(AgentManifest.model_validate(raw), base_dir=REPO_ROOT)
    osc = openshell_config(config.victim_control)
    assert osc.relay_victim is not None
    assert osc.relay_victim.dockerfile == Path("deploy/finance/Dockerfile")
    assert osc.relay_victim.victim_binaries == ["/app/.venv/bin/**"]
    assert osc.start_command is not None
    assert osc.start_command.endswith("-m finance.serve")


def test_byo_start_command_override() -> None:
    raw = {
        "agent": {
            "name": "x",
            "project_dir": "p",
            "dockerfile": "D",
            "binaries": ["/a/**"],
            "start_command": "my custom serve",
            "secrets": ["K"],
        }
    }
    osc = openshell_config(expand(AgentManifest.model_validate(raw), base_dir=REPO_ROOT).victim_control)
    assert osc.start_command == "my custom serve"


def test_a_manifest_without_a_start_command_is_rejected() -> None:
    """Agent Hardener cannot guess how an agent it did not build is launched."""
    raw = {"agent": {"name": "x", "project_dir": "p", "dockerfile": "D", "binaries": ["/a/**"], "secrets": ["K"]}}
    with pytest.raises(ValidationError):
        AgentManifest.model_validate(raw)


def test_the_start_command_is_passed_through_verbatim() -> None:
    """The launch command is passed through unmodified.

    The guardrail is Relay plugin config the agent reads on restart, so the launch command and the
    hardening no longer have to agree about a shared config path.
    """
    config = expand(AgentManifest.model_validate(AGENT), base_dir=REPO_ROOT)
    osc = openshell_config(config.victim_control)
    assert osc.start_command == "/app/.venv/bin/python -m finance.serve"


def test_expand_without_base_dir_has_no_cwd() -> None:
    osc = openshell_config(expand(AgentManifest.model_validate(AGENT)).victim_control)
    assert osc.cwd is None


def test_expand_guardrails_opt_in() -> None:
    base = dict(AGENT)
    base["agent"] = {**AGENT["agent"], "guardrails_tool": "transfer_funds"}
    config = expand(AgentManifest.model_validate(base), base_dir=REPO_ROOT)
    assert [d.name for d in config.defenders] == ["openshell-policy-defender", "defender-guardrails"]
    assert config.defenders[1].config["target_tool"] == "transfer_funds"


def test_overrides_deep_merge() -> None:
    base = {**AGENT, "overrides": {"run": {"rounds": 5}}}
    config = expand(AgentManifest.model_validate(base), base_dir=REPO_ROOT)
    assert config.run.rounds == 5


def test_a_manifest_without_an_image_is_rejected() -> None:
    """Agent Hardener hardens the agent the user ships; it never builds the image itself."""
    raw = {"agent": {"name": "x", "project_dir": "p", "secrets": ["K"]}}
    with pytest.raises((ValidationError, ValueError)):
        AgentManifest.model_validate(raw)


def test_load_manifest_and_load_config_detect_manifest(tmp_path: Path) -> None:
    manifest_file = tmp_path / "agent-hardener.yaml"
    manifest_file.write_text(
        "agent:\n"
        "  name: demo\n"
        "  project_dir: ../agents-lab\n"
        "  dockerfile: Dockerfile\n"
        "  start_command: /app/serve\n"
        "  binaries: ['/app/**']\n"
        "  secrets: [INFERENCE_API_KEY]\n",
        encoding="utf-8",
    )
    manifest = load_manifest(manifest_file)
    assert manifest.agent.name == "demo"
    # load_config auto-detects the manifest (top-level `agent:`) and expands it.
    config = load_config(str(manifest_file))
    assert isinstance(config, SessionConfig)
    assert config.target.name == "demo"


def test_load_config_still_parses_full_session_config() -> None:
    config = load_config(str(REPO_ROOT / "tests/fixtures/pipeline_e2e/pipeline.yaml"))
    assert isinstance(config, SessionConfig)
    assert config.target.name == "pipeline-e2e-target"


def test_load_config_rejects_overrides_for_full_session_config() -> None:
    full_config = REPO_ROOT / "tests/fixtures/pipeline_e2e/pipeline.yaml"
    with pytest.raises(ValueError, match="require a manifest"):
        load_config(str(full_config), policy=Path("/tmp/whatever.yaml"))


def test_example_manifest_round_trips() -> None:
    config = load_config(str(REPO_ROOT / "examples/agent-hardener.yaml"))
    osc = openshell_config(config.victim_control)
    assert osc.relay_victim is not None
    assert [b.name for b in osc.relay_victim.backends] == ["finance"]


# --- garak settings -----------------------------------------------------------------------


def test_expand_default_garak_attacker_config() -> None:
    config = expand(AgentManifest.model_validate(AGENT), base_dir=REPO_ROOT)
    assert config.garak is not None
    attacker_cfg = config.attackers[0].config
    # Defaults flow into the attacker config; overrides are absent until set.
    assert attacker_cfg["config_path"] == "garak-scan.yaml"
    # report_dir is unset by default so garak artifacts land under the run's iteration dir.
    assert "report_dir" not in attacker_cfg
    assert "target_port" not in attacker_cfg


def test_expand_garak_overrides_thread_into_attacker() -> None:
    raw = {
        **AGENT,
        "garak": {"target_port": 9100, "red_team_model_name": "custom-model"},
    }
    config = expand(AgentManifest.model_validate(raw), base_dir=REPO_ROOT)
    attacker_cfg = config.attackers[0].config
    assert attacker_cfg["target_port"] == 9100
    assert attacker_cfg["red_team_model_name"] == "custom-model"


def test_hardening_survives_the_between_round_restart() -> None:
    """The two facts that make round 2+ attack a *hardened* agent rather than the original.

    The guardrails defender must run, and its output must have somewhere to land that the victim
    re-reads on restart. Under NAT this also required the start command to serve the uploaded path;
    that third coupling is gone, because Relay reads its plugin config independently of how the
    agent was launched.
    """
    config = expand(AgentManifest.model_validate(AGENT), base_dir=REPO_ROOT)
    assert "defender-guardrails" in [d.name for d in config.defenders]
    (upload,) = config.victim_control.config["uploads"]
    assert upload.endswith(f":{RELAY_PLUGINS_UPLOAD_DEST}")


def test_defender_timeout_is_overridable_per_run() -> None:
    """A slow victim must not cost a guardrail that was nearly ready.

    The guardrails defender validates its draft by invoking the victim once per augmented attack and
    per benign request, so the default cap can fire mid-refinement and discard the work.
    """
    base = {
        "agent": {
            "name": "a",
            "project_dir": ".",
            "dockerfile": "Dockerfile",
            "start_command": "run",
            "binaries": ["/app/.venv/bin/**"],
        }
    }

    default = expand(AgentManifest.model_validate(base))
    raised = expand(AgentManifest.model_validate({**base, "defenders": {"timeout_s": 1800}}))

    assert {d.timeout_seconds for d in default.defenders} == {DEFAULT_DEFENDER_TIMEOUT_SECONDS}
    assert {d.timeout_seconds for d in raised.defenders} == {1800}


def test_session_reclaim_flags_are_added_to_a_fabric_start_command():
    """A Fabric victim must reclaim sessions no one will ask for again.

    The server opens a durable session per request when the caller sends no session id, and holds it
    for 30 minutes. A war-game never reuses one, and each session carries a rebuilt harness — measured
    at ~160 MB — so the default window exhausts the victim and the run reports the OOM as unblocked
    attacks.
    """
    command = with_session_reclaim("/workspace/.venv/bin/python -m nemo_agents_plugin.fabric.server --port 8000")
    assert "--idle-session-timeout-seconds 20" in command
    assert "--session-cleanup-interval-seconds 10" in command


def test_session_reclaim_leaves_an_unrecognised_command_alone():
    """Appending flags to an author's own launcher would be guessing at its argument parser."""
    own = "/app/bin/serve --listen 0.0.0.0:8000"
    assert with_session_reclaim(own) == own


def test_session_reclaim_is_not_applied_twice():
    """The flags can already be there — from the author's Dockerfile, or a re-rendered manifest."""
    once = with_session_reclaim("python -m nemo_agents_plugin.fabric.server --port 8000")
    assert with_session_reclaim(once) == once


def test_a_hand_written_fabric_manifest_gets_the_flags_too():
    """The gap this closes: nothing generates the command for a CLI user, so nothing else would."""
    manifest = AgentManifest.model_validate(
        {
            "agent": {
                "name": "byo",
                "project_dir": ".",
                "dockerfile": "Dockerfile",
                "start_command": "python -m nemo_agents_plugin.fabric.server --port 8000",
                "binaries": ["/app/.venv/bin/**"],
            }
        }
    )
    config = expand(manifest, base_dir=REPO_ROOT)
    assert "--idle-session-timeout-seconds 20" in config.victim_control.config["start_command"]

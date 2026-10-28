# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the agent-hardener Typer CLI (no Docker/OpenShell required)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import yaml
from typer.testing import CliRunner

from agent_hardener.cli import app
from agent_hardener.config import load_config
from agent_hardener.display.interview import auto_answer_provider, tty_answer_provider
from agent_hardener.manifest import build_backend, build_manifest, render_manifest_yaml
from agent_hardener.models import SessionConfig, VictimControlConfig
from agent_hardener.openshell.lifecycle import OpenShellCommandResult
from agent_hardener.project import (
    append_env,
    default_agent_name,
    discover_dockerfiles,
    discover_egress_entries,
    discover_env_files,
    find_project_root,
    missing_secrets,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

runner = CliRunner()


@pytest.fixture(autouse=True)
def _inference_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The run/synth/serve commands preflight-require a credential; default it for CLI tests."""
    monkeypatch.setenv("INFERENCE_API_KEY", "test-key")


def _ok(output: str = "ok") -> OpenShellCommandResult:
    return OpenShellCommandResult(ok=True, command=[], output=output)


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("run", "up", "down", "status", "init"):
        assert command in result.output


def test_run_help_lists_flags() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    for flag in ("--config", "--env-file", "--rounds", "--no-cleanup"):
        assert flag in result.output


def _replay_session_config(root_dir: Path) -> SessionConfig:
    return SessionConfig(
        storage={"root_dir": root_dir},
        target={"name": "victim"},
        victim={"name": "victim", "role": "victim", "implementation": "pkg:run"},
    )


def _patch_run_command(monkeypatch: pytest.MonkeyPatch, session_config: SessionConfig) -> list[SessionConfig]:
    """Stub the run command's side effects; capture the SessionConfig handed to run_mission."""
    captured: list[SessionConfig] = []

    def fake_run_mission(config: SessionConfig, **_kwargs: object) -> SimpleNamespace:
        captured.append(config)
        return SimpleNamespace(success=True)

    monkeypatch.setattr("agent_hardener.tools.openshell.configure_local_docker_host", lambda: None)
    monkeypatch.setattr("agent_hardener.tools.openshell.load_local_env", lambda _p: None)
    monkeypatch.setattr("agent_hardener.tools.openshell.sandbox_is_ready", lambda _c: False)
    monkeypatch.setattr("agent_hardener.config.load_config", lambda _p, **_kw: session_config)
    monkeypatch.setattr("agent_hardener.runtime.runner.run_mission", fake_run_mission)
    return captured


def test_run_replay_resolves_latest_hitlog_from_run_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hitlog = tmp_path / "run-logs" / "run-abc" / "round_1" / "garak" / "agent-breaker.uuid.hitlog.jsonl"
    hitlog.parent.mkdir(parents=True)
    hitlog.write_text("{}\n", encoding="utf-8")
    captured = _patch_run_command(monkeypatch, _replay_session_config(tmp_path))

    result = runner.invoke(app, ["run", "--replay"])

    assert result.exit_code == 0
    (config,) = captured
    assert config.attackers == []
    assert [attack.path for attack in config.preloaded_attacks] == [hitlog.resolve()]


def test_run_victim_image_reaches_the_sandbox_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session_config = _replay_session_config(tmp_path).model_copy(
        update={"victim_control": VictimControlConfig(type="openshell", config={"sandbox": "sb"})}
    )
    captured = _patch_run_command(monkeypatch, session_config)

    result = runner.invoke(app, ["run", "--victim-image", "registry.example.com/agent:1"])

    assert result.exit_code == 0
    (config,) = captured
    assert config.victim_control.config == {"sandbox": "sb", "image": "registry.example.com/agent:1"}


def test_run_replay_fails_without_hitlog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run_command(monkeypatch, _replay_session_config(tmp_path))

    result = runner.invoke(app, ["run", "--replay"])

    assert result.exit_code == 1
    assert "No hitlog found under" in result.output


def test_run_replay_with_explicit_hitlog_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hitlog = tmp_path / "uploaded.hitlog.jsonl"
    hitlog.write_text("{}\n", encoding="utf-8")
    captured = _patch_run_command(monkeypatch, _replay_session_config(tmp_path))

    result = runner.invoke(app, ["run", "--replay", str(hitlog)])

    assert result.exit_code == 0
    (config,) = captured
    assert config.attackers == []
    assert [attack.path for attack in config.preloaded_attacks] == [hitlog.resolve()]


def test_run_replay_explicit_path_missing_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run_command(monkeypatch, _replay_session_config(tmp_path))

    result = runner.invoke(app, ["run", "--replay", str(tmp_path / "nope.jsonl")])

    assert result.exit_code == 1
    assert "Replay hitlog not found" in result.output


def test_run_benign_suite_sets_session_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    suite = tmp_path / "suite.csv"
    suite.write_text("tool,payload,label,rationale,persona\nclock,now,benign,,\n", encoding="utf-8")
    captured = _patch_run_command(monkeypatch, _replay_session_config(tmp_path))

    result = runner.invoke(app, ["run", "--benign-suite", str(suite)])

    assert result.exit_code == 0
    (config,) = captured
    assert config.benign_suite_path == suite.resolve()


def test_run_benign_suite_missing_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run_command(monkeypatch, _replay_session_config(tmp_path))

    result = runner.invoke(app, ["run", "--benign-suite", str(tmp_path / "nope.csv")])

    assert result.exit_code == 1
    assert "Benign suite not found" in result.output


def test_run_rejects_removed_synth_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Generation moved to `synth-benign`; `run` is a pure consumer and no longer accepts these.
    _patch_run_command(monkeypatch, _replay_session_config(tmp_path))
    for flag in ("--reuse-benign", "--stop-after-synth"):
        result = runner.invoke(app, ["run", flag])
        assert result.exit_code == 2, f"{flag} should be rejected as an unknown option"


def _make_workflows(root: Path, *rel_paths: str) -> None:
    for rel in rel_paths:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("_type: react_agent\n", encoding="utf-8")


def test_default_agent_name_from_workflow_folder() -> None:
    assert default_agent_name("../agents-lab", "agents_lab/agents/codereview/workflow.yaml") == "codereview"


def test_default_agent_name_includes_variant() -> None:
    assert (
        default_agent_name("../agents-lab", "agents_lab/agents/codereview/workflow.hardened.yaml")
        == "codereview-hardened"
    )


def test_default_agent_name_falls_back_to_project_dir() -> None:
    assert default_agent_name("../agents-lab", None) == "agents-lab"


def test_find_project_root_walks_up(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    agent_dir = tmp_path / "agents_lab" / "agents" / "finance"
    agent_dir.mkdir(parents=True)
    # Pointing at the agent subfolder resolves up to the project root.
    assert find_project_root(agent_dir) == tmp_path
    assert find_project_root(tmp_path) == tmp_path


def test_find_project_root_none_when_no_marker(tmp_path: Path) -> None:
    sub = tmp_path / "loose" / "dir"
    sub.mkdir(parents=True)
    # No installable marker in this tree or its (tmp) ancestors.
    assert find_project_root(sub) is None


def test_init_resolves_project_root_from_agent_subfolder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A real NAT layout: root has pyproject; the agent + workflow live in a subfolder.
    root = tmp_path / "myproject"
    agent_dir = root / "agents_lab" / "agents" / "finance"
    agent_dir.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='myproject'\n", encoding="utf-8")
    (agent_dir / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "agent-hardener.yaml"

    # Point init at the agent subfolder — it should rewrite to the project root + relative workflow.
    result = runner.invoke(
        app,
        [
            "init",
            "--project-dir",
            str(agent_dir),
            "--name",
            "finance",
            "--secrets-file",
            "agent.env",
            "--secrets",
            "K",
            "--binary",
            "/app/**",
            "--start-command",
            "/app/serve",
            "--yes",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert data["agent"]["project_dir"] == "myproject"


def _scaffold_project(root: Path) -> Path:
    """A minimal NAT layout: install marker at root, workflow + .env in an agent subfolder."""
    agent_dir = root / "agents_lab" / "agents" / "research"
    agent_dir.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='myproject'\n", encoding="utf-8")
    (agent_dir / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    (agent_dir / "workflow.hardened.yaml").write_text("_type: react_agent\n", encoding="utf-8")
    (root / ".env").write_text("INFERENCE_API_KEY=x\nEXTRA_TOKEN=y\n", encoding="utf-8")
    return agent_dir


def test_inspect_json_detects_project_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "myproject"
    agent_dir = _scaffold_project(root)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["inspect", "--project-dir", str(agent_dir), "--json"])
    assert result.exit_code == 0, result.output
    detected = json.loads(result.output)

    assert detected["project_dir"] == "myproject"
    # Workflows are project-root-relative so they can be passed straight back to `init --workflow`.
    assert detected["default_agent_name"] == "myproject"
    assert detected["default_port"] == 8000
    assert detected["secrets_file"] == ".env"
    assert detected["secret_names"] == ["INFERENCE_API_KEY", "EXTRA_TOKEN"]


def test_init_yes_writes_egress_from_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='proj'\n", encoding="utf-8")
    (root / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "agent-hardener.yaml"

    result = runner.invoke(
        app,
        [
            "init",
            "--project-dir",
            str(root),
            "--name",
            "proj",
            "--secrets-file",
            "agent.env",
            "--secrets",
            "K",
            "--egress",
            "host.docker.internal:8086",
            "--egress",
            "integrate.api.nvidia.com",
            "--binary",
            "/app/**",
            "--start-command",
            "/app/serve",
            "--yes",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert data["agent"]["egress"] == ["host.docker.internal:8086", "integrate.api.nvidia.com"]


def test_init_yes_writes_route_only_backend_from_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='proj'\n", encoding="utf-8")
    (root / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "agent-hardener.yaml"

    result = runner.invoke(
        app,
        [
            "init",
            "--project-dir",
            str(root),
            "--name",
            "proj",
            "--secrets-file",
            "agent.env",
            "--secrets",
            "K",
            "--backend",
            "finance:8086",
            "--backend",
            "cache:6379,6380",
            "--binary",
            "/app/**",
            "--start-command",
            "/app/serve",
            "--yes",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert data["backends"] == [{"name": "finance", "ports": [8086]}, {"name": "cache", "ports": [6379, 6380]}]


def test_init_backend_flag_rejects_bad_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='proj'\n", encoding="utf-8")
    (root / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "init",
            "--project-dir",
            str(root),
            "--name",
            "p",
            "--secrets",
            "K",
            "--backend",
            "finance",
            "--binary",
            "/app/**",
            "--start-command",
            "/app/serve",
            "--yes",
            "--output",
            str(tmp_path / "o.yaml"),
        ],
    )
    assert result.exit_code != 0
    assert "NAME:PORT" in result.output


def test_inspect_detects_backend_ports_from_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='proj'\n", encoding="utf-8")
    (root / "workflow.yaml").write_text(
        "functions:\n  t:\n    base_url: http://localhost:8086\n  u:\n    base_url: http://127.0.0.1:5432\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["inspect", "--project-dir", str(root), "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["backend_ports"] == [5432, 8086]


def test_build_manifest_records_the_users_image_and_command() -> None:
    manifest = build_manifest(
        name="finance",
        project_dir="../agents-lab",
        dockerfile="deploy/Dockerfile",
        start_command="/app/serve",
        binaries=["/app/**"],
        port=8000,
        secrets=["INFERENCE_API_KEY"],
    )
    agent = manifest["agent"]
    assert agent["dockerfile"] == "deploy/Dockerfile"
    assert agent["start_command"] == "/app/serve"
    assert agent["binaries"] == ["/app/**"]
    assert agent["secrets"] == ["INFERENCE_API_KEY"]


def test_render_manifest_yaml_is_valid_and_contains_stubs() -> None:
    manifest = build_manifest(
        name="fin",
        project_dir="../agents-lab",
        dockerfile="deploy/Dockerfile",
        start_command="/app/serve",
        binaries=["/app/**"],
        port=8000,
        secrets=["INFERENCE_API_KEY"],
    )
    rendered = render_manifest_yaml(manifest)
    data = yaml.safe_load(rendered)
    assert data["agent"]["name"] == "fin"
    assert data["backends"] == []
    assert "# egress:" in rendered
    assert "# discover_egress:" in rendered
    assert "# garak:" in rendered
    assert "# overrides:" in rendered
    assert "# relay_artifacts:" in rendered
    assert "NemoRelayMiddleware" in rendered  # the relay contract is stated where users will read it
    assert data["agent"]["dockerfile"] == "deploy/Dockerfile"
    # Backend example stub is present when backends list is empty.
    assert "# - name: my-service" in rendered


def test_build_manifest_writes_egress_and_render_drops_stub() -> None:
    manifest = build_manifest(
        name="fin",
        project_dir="p",
        dockerfile="deploy/Dockerfile",
        start_command="/app/serve",
        binaries=["/app/**"],
        port=8000,
        secrets=["K"],
        egress=["api.example.com", "svc:9000"],
    )
    assert manifest["agent"]["egress"] == ["api.example.com", "svc:9000"]
    rendered = render_manifest_yaml(manifest)
    assert "# egress:" not in rendered  # real egress key present, so the commented stub is dropped


def test_discover_env_files_finds_dotenv_and_variants(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("A=1\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text("B=2\n", encoding="utf-8")
    found = discover_env_files(tmp_path)
    assert found == [tmp_path / ".env", tmp_path / ".env.local"]  # plain .env sorts before variants
    assert discover_env_files(tmp_path / "missing") == []


def test_discover_dockerfiles_matches_variants(tmp_path: Path) -> None:
    (tmp_path / "deploy").mkdir()
    (tmp_path / "deploy" / "Dockerfile").write_text("FROM python\n", encoding="utf-8")
    (tmp_path / "Dockerfile.prod").write_text("FROM python\n", encoding="utf-8")
    (tmp_path / "agent.Dockerfile").write_text("FROM python\n", encoding="utf-8")
    names = {p.name for p in discover_dockerfiles(tmp_path)}
    assert names == {"Dockerfile", "Dockerfile.prod", "agent.Dockerfile"}
    assert discover_dockerfiles(tmp_path / "missing") == []


def test_init_byo_uses_detected_dockerfile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "proj"
    (project / "deploy").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='p'\n", encoding="utf-8")  # no workflow
    (project / "deploy" / "Dockerfile").write_text("FROM python\n", encoding="utf-8")  # → BYO detected
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.env").write_text("INFERENCE_API_KEY=present\n", encoding="utf-8")
    output = tmp_path / "agent-hardener.yaml"

    # Only a Dockerfile is present, so the image prompt defaults to BYO (2); accept the detected
    # Dockerfile + default binaries, blank workflow (none to find), default port, no egress,
    # decline backend. No start-command prompt any more — it is generated.
    result = runner.invoke(
        app,
        [
            "init",
            "--project-dir",
            str(project),
            "--name",
            "byo",
            "--secrets-file",
            "agent.env",
            "--output",
            str(output),
        ],
        input="\n\n/app/serve\n\n\n\n\n\n\nn\n",
    )
    assert result.exit_code == 0, result.output
    agent = yaml.safe_load(output.read_text(encoding="utf-8"))["agent"]
    assert agent["dockerfile"] == "deploy/Dockerfile"  # the detected default was accepted
    assert agent["start_command"] == "/app/serve"


def test_init_regenerates_stale_garak_with_chosen_port(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "proj"
    (project / "a").mkdir(parents=True)
    (project / "a" / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    (tmp_path / "agent.env").write_text("INFERENCE_API_KEY=present\n", encoding="utf-8")
    garak = tmp_path / "garak-scan.yaml"
    garak.write_text("stale: true\n", encoding="utf-8")  # pre-existing with the wrong port/content
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "agent-hardener.yaml"

    result = runner.invoke(
        app,
        [
            "init",
            "--project-dir",
            str(project),
            "--name",
            "fin",
            "--secrets-file",
            "agent.env",
            "--output",
            str(output),
        ],
        input="\n\n/app/serve\n9001\n\nn\n",  # port 9001, no extra egress, decline backend
    )
    assert result.exit_code == 0, result.output
    text = garak.read_text(encoding="utf-8")
    assert "stale" not in text  # old file was overwritten
    assert "127.0.0.1:9001" in text  # regenerated with the chosen port


def test_discover_egress_entries_from_project_config(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("base_url: https://api.tavily.com/search\n", encoding="utf-8")
    entries = discover_egress_entries(tmp_path)
    assert "api.tavily.com" in entries  # discovered host (port 443 → bare host)


def test_init_writes_valid_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "proj"
    (project / "agents" / "fin").mkdir(parents=True)
    (project / "agents" / "fin" / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "agent-hardener.yaml"

    result = runner.invoke(
        app,
        [
            "init",
            "--project-dir",
            str(project),
            "--name",
            "fin",
            "--secrets",
            "INFERENCE_API_KEY",
            "--secrets-file",
            "agent.env",
            "--output",
            str(output),
            "--binary",
            "/app/**",
            "--start-command",
            "/app/serve",
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert data["agent"]["name"] == "fin"
    assert data["agent"]["secrets_file"] == "agent.env"
    # The scaffolded manifest expands into a valid session config.
    assert isinstance(load_config(str(output)), SessionConfig)
    # The summary lists the generated artifacts (paths shown; not asserted exactly since Rich wraps).
    assert "Generated:" in result.output
    assert "garak-scan.yaml" in result.output


def test_build_backend() -> None:
    assert build_backend("finance", "c.yaml", [8086], "http://h/health") == {
        "name": "finance",
        "compose_file": "c.yaml",
        "ports": [8086],
        "health_url": "http://h/health",
    }
    assert "health_url" not in build_backend("x", "c.yaml", [1])  # omitted when blank


def test_build_backend_without_compose_file() -> None:
    # An already-running host service: no compose file, just the route (ports) + optional health.
    backend = build_backend("ext-db", None, [5432], "http://127.0.0.1:5432/health")
    assert backend == {"name": "ext-db", "ports": [5432], "health_url": "http://127.0.0.1:5432/health"}
    assert "compose_file" not in backend


def test_init_backend_without_compose_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "proj"
    (project / "a").mkdir(parents=True)
    (project / "a" / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.env").write_text("INFERENCE_API_KEY=present\n", encoding="utf-8")
    output = tmp_path / "agent-hardener.yaml"

    # port; no egress; backend yes; name; blank compose (already running); ports; blank health; no more.
    result = runner.invoke(
        app,
        [
            "init",
            "--project-dir",
            str(project),
            "--name",
            "fin",
            "--secrets-file",
            "agent.env",
            "--output",
            str(output),
        ],
        input="\n\n/app/serve\n8000\n\ny\n\next\n5432\n\nn\n",  # backend: confirm, blank compose, name, ports, blank health, no-more
    )
    assert result.exit_code == 0, result.output
    backends = yaml.safe_load(output.read_text(encoding="utf-8"))["backends"]
    assert backends == [{"name": "ext", "ports": [5432]}]  # compose_file omitted
    assert isinstance(load_config(str(output)), SessionConfig)  # compose-less backend still expands


def test_init_prompts_for_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "proj"
    (project / "a").mkdir(parents=True)
    (project / "a" / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.env").write_text("INFERENCE_API_KEY=present\n", encoding="utf-8")
    output = tmp_path / "agent-hardener.yaml"

    # Dockerfile + binaries (accept defaults), start command, port, egress (none → blank), then
    # backend: confirm, compose file, name, ports, health, no-more.
    backend_input = (
        "\n\n/app/serve\n"
        "8000\n\ny\n../agents-lab/services/finance_backend/docker-compose.yaml\nfinance\n8086\n"
        "http://127.0.0.1:8086/health\nn\n"
    )
    result = runner.invoke(
        app,
        [
            "init",
            "--project-dir",
            str(project),
            "--name",
            "fin",
            "--secrets-file",
            "agent.env",
            "--output",
            str(output),
        ],
        input=backend_input,
    )
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert data["backends"] == [
        {
            "name": "finance",
            "compose_file": "../agents-lab/services/finance_backend/docker-compose.yaml",
            "ports": [8086],
            "health_url": "http://127.0.0.1:8086/health",
        }
    ]


def test_init_skips_backend_when_declined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "proj"
    (project / "a").mkdir(parents=True)
    (project / "a" / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.env").write_text("INFERENCE_API_KEY=present\n", encoding="utf-8")
    output = tmp_path / "agent-hardener.yaml"

    result = runner.invoke(
        app,
        [
            "init",
            "--project-dir",
            str(project),
            "--name",
            "fin",
            "--secrets-file",
            "agent.env",
            "--output",
            str(output),
        ],
        input="\n\n/app/serve\n8000\n\nn\n",  # default port, no extra egress, then decline the backend prompt
    )
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(output.read_text(encoding="utf-8"))["backends"] == []


def test_init_prompts_for_the_image_and_the_launch_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """There is no longer a "build it for me" branch — the image and its command are the user's."""
    project = tmp_path / "proj"
    (project / "a").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='p'\n", encoding="utf-8")
    (project / "a" / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.env").write_text("INFERENCE_API_KEY=present\n", encoding="utf-8")
    output = tmp_path / "agent-hardener.yaml"

    # dockerfile → binaries (accept default) → start command → port → no extra egress → no backend
    result = runner.invoke(
        app,
        [
            "init",
            "--project-dir",
            str(project),
            "--name",
            "byo",
            "--secrets-file",
            "agent.env",
            "--output",
            str(output),
        ],
        input="deploy/Dockerfile\n\n/app/serve\n8000\n\nn\n",
    )
    assert result.exit_code == 0, result.output
    agent = yaml.safe_load(output.read_text(encoding="utf-8"))["agent"]
    assert agent["dockerfile"] == "deploy/Dockerfile"
    assert agent["start_command"] == "/app/serve"


def test_init_dockerfile_flag_is_scriptable_under_yes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--dockerfile composes with --yes; without it BYO was reachable only through a TTY prompt."""
    project = tmp_path / "proj"
    (project / "a").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='p'\n", encoding="utf-8")
    (project / "a" / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.env").write_text("INFERENCE_API_KEY=present\n", encoding="utf-8")
    output = tmp_path / "agent-hardener.yaml"

    result = runner.invoke(
        app,
        [
            "init",
            "--start-command",
            "/app/serve",
            "--yes",
            "--project-dir",
            str(project),
            "--name",
            "byo",
            "--secrets-file",
            "agent.env",
            "--output",
            str(output),
            "--dockerfile",
            "deploy/Dockerfile",
            "--binary",
            "/app/.venv/bin/**",
        ],
    )
    assert result.exit_code == 0, result.output
    agent = yaml.safe_load(output.read_text(encoding="utf-8"))["agent"]
    assert agent["dockerfile"] == "deploy/Dockerfile"
    assert agent["binaries"] == ["/app/.venv/bin/**"]  # auto-discovered, not dropped


def test_init_dockerfile_without_binary_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RelayVictimSpec rejects a BYO image with no binary globs; fail at init with the reason."""
    project = tmp_path / "proj"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='p'\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "agent-hardener.yaml"

    result = runner.invoke(
        app,
        [
            "init",
            "--yes",
            "--start-command",
            "/app/serve",
            "--project-dir",
            str(project),
            "--output",
            str(output),
            "--dockerfile",
            "D",
        ],
    )
    assert result.exit_code == 1
    assert "--binary" in result.output
    assert not output.exists()


def test_init_egress_discovery_approval_and_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "proj"
    (project / "a").mkdir(parents=True)
    (project / "a" / "config.yaml").write_text("base_url: https://api.tavily.com/search\n", encoding="utf-8")
    (project / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.env").write_text("INFERENCE_API_KEY=present\n", encoding="utf-8")
    output = tmp_path / "agent-hardener.yaml"

    # dockerfile + binaries (accept defaults); start command; port; egress "y" (allow
    # discovered) + add "extra.example.com"; decline backend.
    result = runner.invoke(
        app,
        [
            "init",
            "--project-dir",
            str(project),
            "--name",
            "fin",
            "--secrets-file",
            "agent.env",
            "--output",
            str(output),
        ],
        input="\n\n/app/serve\n8000\ny\nextra.example.com\nn\n",
    )
    assert result.exit_code == 0, result.output
    egress = yaml.safe_load(output.read_text(encoding="utf-8"))["agent"]["egress"]
    assert "api.tavily.com" in egress  # discovered + approved
    assert "extra.example.com" in egress  # user-added


def test_init_detects_env_in_agent_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "proj"
    (project / "a").mkdir(parents=True)
    (project / "a" / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    (project / ".env").write_text("INFERENCE_API_KEY=present\n", encoding="utf-8")  # lives in the agent root
    (tmp_path / ".env").write_text("CWD_KEY=should-not-be-picked\n", encoding="utf-8")  # Agent Hardener cwd — ignored
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "agent-hardener.yaml"

    # No --secrets-file: the prompt should default to the agent-root .env, not the cwd one. Accept it.
    result = runner.invoke(
        app,
        [
            "init",
            "--project-dir",
            str(project),
            "--name",
            "fin",
            "--output",
            str(output),
        ],
        input="\n\n/app/serve\n\n\n\nn\n",  # port, secrets-file (accept detected), egress add (none), decline backend
    )
    assert result.exit_code == 0, result.output
    agent = yaml.safe_load(output.read_text(encoding="utf-8"))["agent"]
    assert agent["secrets_file"] == "proj/.env"  # detected in the agent root, expressed relative to cwd
    assert agent["secrets"] == ["INFERENCE_API_KEY"]  # seeded from that file, not the cwd .env


def test_init_question_mark_shows_help_then_reprompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "proj"
    (project / "a").mkdir(parents=True)
    (project / "a" / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.env").write_text("INFERENCE_API_KEY=present\n", encoding="utf-8")
    output = tmp_path / "agent-hardener.yaml"

    # Answer the port prompt with "?" first: help text prints, then it re-asks and accepts 8000.
    result = runner.invoke(
        app,
        [
            "init",
            "--project-dir",
            str(project),
            "--name",
            "fin",
            "--secrets-file",
            "agent.env",
            "--output",
            str(output),
        ],
        input="\n\n/app/serve\n?\n8000\n\nn\n",
    )
    assert result.exit_code == 0, result.output
    assert "garak attacks" in result.output  # the port help blurb was shown
    assert yaml.safe_load(output.read_text(encoding="utf-8"))["agent"]["port"] == 8000  # re-prompt accepted


def test_init_seeds_secrets_from_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "proj"
    (project / "a").mkdir(parents=True)
    (project / "a" / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.env").write_text("AGENT_KEY=v1\nOTHER_KEY=v2\n", encoding="utf-8")
    output = tmp_path / "agent-hardener.yaml"

    result = runner.invoke(
        app,
        [
            "init",
            "--project-dir",
            str(project),
            "--name",
            "fin",
            "--secrets-file",
            "agent.env",
            "--output",
            str(output),
            "--binary",
            "/app/**",
            "--start-command",
            "/app/serve",
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(output.read_text(encoding="utf-8"))
    # secret names default to the keys already present in the agent's env file.
    assert data["agent"]["secrets"] == ["AGENT_KEY", "OTHER_KEY"]


def test_missing_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FROM_ENV", raising=False)
    monkeypatch.setenv("FROM_ENV", "set")
    env_path = tmp_path / ".env"
    env_path.write_text("IN_FILE=value\n", encoding="utf-8")
    assert missing_secrets(["FROM_ENV", "IN_FILE", "ABSENT"], env_path) == ["ABSENT"]


def test_append_env_creates_and_appends(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    append_env(env_path, {"A": "1"})
    assert env_path.read_text(encoding="utf-8") == "A=1\n"
    append_env(env_path, {"B": "2"})
    assert env_path.read_text(encoding="utf-8") == "A=1\nB=2\n"
    append_env(env_path, {})  # no-op
    assert env_path.read_text(encoding="utf-8") == "A=1\nB=2\n"


def test_init_prompts_for_missing_secret_and_writes_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "proj"
    (project / "a").mkdir(parents=True)
    (project / "a" / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)

    result = runner.invoke(
        app,
        [
            "init",
            "--project-dir",
            str(project),
            "--name",
            "fin",
            "--secrets-file",
            "agent.env",
        ],
        input="\n\n/app/serve\n8000\n\nn\nnvapi-secret\n",  # default port, no extra egress, decline backend, then the INFERENCE_API_KEY prompt
    )
    assert result.exit_code == 0, result.output
    env_text = (tmp_path / "agent.env").read_text(encoding="utf-8")
    assert "INFERENCE_API_KEY=nvapi-secret" in env_text


def test_init_yes_does_not_prompt_or_write_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "proj"
    (project / "a").mkdir(parents=True)
    (project / "a" / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)

    result = runner.invoke(
        app,
        [
            "init",
            "--project-dir",
            str(project),
            "--name",
            "fin",
            "--start-command",
            "/app/serve",
            "--binary",
            "/app/**",
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    assert not (tmp_path / ".env").exists()  # --yes never prompts/writes secrets


# --- lifecycle commands (mocked OpenShell) ------------------------------------------------


def _patch_common(monkeypatch: pytest.MonkeyPatch, base_config: object) -> None:
    monkeypatch.setattr("agent_hardener.tools.openshell.configure_local_docker_host", lambda: None)
    monkeypatch.setattr("agent_hardener.tools.openshell.load_local_env", lambda _p: None)
    monkeypatch.setattr("agent_hardener.tools.openshell.openshell_config", lambda _vc: base_config)
    monkeypatch.setattr("agent_hardener.tools.openshell.prepare_relay_victim", lambda cfg: cfg)
    monkeypatch.setattr(
        "agent_hardener.config.load_config",
        lambda _p: SimpleNamespace(victim_control="vc", target=SimpleNamespace(base_url=None)),
    )


def test_down_command(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_common(monkeypatch, base_config="cfg")

    class FakeLifecycle:
        def __init__(self, _cfg: object) -> None:
            pass

        def down(self) -> OpenShellCommandResult:
            return _ok("torn-down")

    monkeypatch.setattr("agent_hardener.openshell.lifecycle.OpenShellLifecycle", FakeLifecycle)
    result = runner.invoke(app, ["down", "--config", "x.yaml"])
    assert result.exit_code == 0
    assert "torn-down" in result.output


def test_status_command(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_common(monkeypatch, base_config="cfg")

    class FakeLifecycle:
        def __init__(self, _cfg: object) -> None:
            pass

        def status(self) -> OpenShellCommandResult:
            return _ok("status-ready")

    monkeypatch.setattr("agent_hardener.openshell.lifecycle.OpenShellLifecycle", FakeLifecycle)
    result = runner.invoke(app, ["status", "--config", "x.yaml"])
    assert result.exit_code == 0
    assert "status-ready" in result.output


def test_up_command(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_common(monkeypatch, base_config="cfg")

    class FakeLifecycle:
        def __init__(self, _cfg: object) -> None:
            pass

        def up(self) -> OpenShellCommandResult:
            return _ok("brought-up")

    monkeypatch.setattr("agent_hardener.openshell.lifecycle.OpenShellLifecycle", FakeLifecycle)
    result = runner.invoke(app, ["up", "--config", "x.yaml"])
    assert result.exit_code == 0
    assert "brought-up" in result.output


def test_stage_prints_the_staged_build_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_common(monkeypatch, base_config=SimpleNamespace(relay_victim="spec", build_context=tmp_path / "Dockerfile"))
    result = runner.invoke(app, ["stage", "--config", "x.yaml"])
    assert result.exit_code == 0
    assert result.output.strip() == str(tmp_path)


def test_stage_fails_without_a_victim_build(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_common(monkeypatch, base_config=SimpleNamespace(relay_victim=None, build_context=None))
    result = runner.invoke(app, ["stage", "--config", "x.yaml"])
    assert result.exit_code == 1
    assert "no victim build to stage" in result.output


def _patch_synth(monkeypatch: pytest.MonkeyPatch, events: list[str], *, up_ok: bool = True) -> None:
    """Wire synth-benign against a recording sandbox lifecycle + a stubbed wizard.

    ``managed_victim_sandbox`` uses the ``OpenShellLifecycle`` bound in ``agent_hardener.tools.openshell``,
    so patch it there (not in ``openshell.lifecycle``).
    """
    _patch_common(monkeypatch, base_config="cfg")

    class FakeLifecycle:
        def __init__(self, _cfg: object) -> None:
            pass

        def up(self) -> OpenShellCommandResult:
            events.append("up")
            return _ok("brought-up") if up_ok else OpenShellCommandResult(ok=False, command=[], output="build failed")

        def down(self) -> OpenShellCommandResult:
            events.append("down")
            return _ok("torn-down")

    monkeypatch.setattr("agent_hardener.tools.openshell.OpenShellLifecycle", FakeLifecycle)
    monkeypatch.setattr(
        "agent_hardener.agents.validators.smart_benign.wizard.run_synth_wizard",
        lambda *_a, **_k: events.append("synth") or 0,
    )


def test_synth_benign_manages_sandbox_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    _patch_synth(monkeypatch, events)
    result = runner.invoke(app, ["synth-benign", "--config", "x.yaml"])
    assert result.exit_code == 0
    assert events == ["up", "synth", "down"]  # builds, synthesizes, tears down


def test_synth_benign_reuse_skips_build_and_teardown(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    _patch_synth(monkeypatch, events)
    monkeypatch.setattr("agent_hardener.tools.openshell.sandbox_is_ready", lambda _cfg: True)
    result = runner.invoke(app, ["synth-benign", "--reuse"])
    assert result.exit_code == 0
    assert events == ["synth"]  # ready sandbox reused: no up, no down


def test_synth_benign_no_cleanup_leaves_sandbox_up(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    _patch_synth(monkeypatch, events)
    result = runner.invoke(app, ["synth-benign", "--no-cleanup"])
    assert result.exit_code == 0
    assert events == ["up", "synth"]  # brought up but left running


def test_synth_benign_propagates_wizard_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    _patch_synth(monkeypatch, events)
    monkeypatch.setattr(
        "agent_hardener.agents.validators.smart_benign.wizard.run_synth_wizard",
        lambda *_a, **_k: events.append("synth") or 3,
    )
    result = runner.invoke(app, ["synth-benign"])
    assert result.exit_code == 3
    assert events == ["up", "synth", "down"]  # non-zero synth still tears the sandbox down


def test_synth_benign_surfaces_sandbox_build_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    _patch_synth(monkeypatch, events, up_ok=False)
    result = runner.invoke(app, ["synth-benign"])
    assert result.exit_code != 0
    assert events == ["up"]  # failed to build; wizard never ran, nothing to tear down


def test_synth_benign_yes_and_no_interactive_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    _patch_synth(monkeypatch, events)
    result = runner.invoke(app, ["synth-benign", "--yes", "--no-interactive"])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output
    assert events == []  # guarded before touching the sandbox


def test_synth_benign_yes_selects_auto_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    _patch_synth(monkeypatch, [])
    monkeypatch.setattr(
        "agent_hardener.agents.validators.smart_benign.wizard.run_synth_wizard",
        lambda *_a, **kw: captured.update(kw) or 0,
    )

    assert runner.invoke(app, ["synth-benign", "--yes"]).exit_code == 0
    assert captured["yes"] is True
    assert captured["answer_provider"] is auto_answer_provider

    captured.clear()
    assert runner.invoke(app, ["synth-benign"]).exit_code == 0
    assert captured["yes"] is False
    assert captured["answer_provider"] is tty_answer_provider


def _fake_run_session_config() -> SimpleNamespace:
    return SimpleNamespace(
        garak=None,
        victim_control="vc",
        target=SimpleNamespace(base_url="http://127.0.0.1:8000/v1/chat/completions"),
    )


def test_run_command_wires_run_mission(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr("agent_hardener.tools.openshell.configure_local_docker_host", lambda: None)
    monkeypatch.setattr("agent_hardener.tools.openshell.load_local_env", lambda _p: None)
    monkeypatch.setattr("agent_hardener.tools.openshell.sandbox_is_ready", lambda _c: False)
    monkeypatch.setattr("agent_hardener.tools.openshell.openshell_config", lambda _vc: "cfg")
    monkeypatch.setattr("agent_hardener.config.load_config", lambda _p, **_kw: _fake_run_session_config())

    def fake_run_mission(session_config, **kwargs):
        captured["session_config"] = session_config
        captured["kwargs"] = kwargs
        return SimpleNamespace(reports=[], success=True)

    monkeypatch.setattr("agent_hardener.runtime.runner.run_mission", fake_run_mission)
    result = runner.invoke(app, ["run", "--config", "x.yaml", "--rounds", "2"])
    assert result.exit_code == 0
    assert captured["kwargs"]["rounds"] == 2  # type: ignore[index]


def test_run_command_mission_error_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_hardener.runtime.runner import MissionError  # noqa: PLC0415

    monkeypatch.setattr("agent_hardener.tools.openshell.configure_local_docker_host", lambda: None)
    monkeypatch.setattr("agent_hardener.tools.openshell.load_local_env", lambda _p: None)
    monkeypatch.setattr("agent_hardener.config.load_config", lambda _p, **_kw: _fake_run_session_config())

    def boom(*_a: object, **_k: object) -> None:
        raise MissionError("sandbox down")

    monkeypatch.setattr("agent_hardener.runtime.runner.run_mission", boom)
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 1
    assert "sandbox down" in result.output

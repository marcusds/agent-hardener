# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the NAT-victim path (config models, staging, egress, backends, prepare)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from agent_hardener.config import load_config
from agent_hardener.errors import VictimBuildError
from agent_hardener.models import BackendEndpoint, BackendServiceSpec, RelayVictimSpec
from agent_hardener.openshell.egress import DockerfileEnvEgressSource
from agent_hardener.openshell.lifecycle import OpenShellCommandResult, OpenShellConfig
from agent_hardener.openshell.policy_egress import (
    apply_victim_binaries,
    bind_provider_credentials,
    build_backend_egress_block,
    inject_backend_egress,
    reconcile_shared_endpoint_settings,
)
from agent_hardener.openshell.relay_victim import GUARDRAIL_PACKAGE, dockerfile_env, stage_relay_victim_build
from agent_hardener.providers import backends
from agent_hardener.providers.backends import BackendManager
from agent_hardener.relay_plugin.victim import SANDBOX_ARTIFACTS_DIR
from agent_hardener.tools import openshell as tool
from agent_hardener.tools.openshell import _launch_env, openshell_config, prepare_relay_victim, sandbox_is_ready

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


class RecordingRunner:
    def __init__(self, returncode: int = 0) -> None:
        self.commands: list[list[str]] = []
        self.returncode = returncode

    def run(self, args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        self.commands.append(args)
        return subprocess.CompletedProcess(args=args, returncode=self.returncode, stdout="ok")


@pytest.mark.parametrize(
    ("ok", "output", "ready"),
    [(True, "Phase: Ready", True), (True, "Phase: Pending", False), (False, "Phase: Ready", False)],
)
def test_sandbox_is_ready(monkeypatch: pytest.MonkeyPatch, *, ok: bool, output: str, ready: bool) -> None:
    class FakeLifecycle:
        def __init__(self, _config: object) -> None:
            pass

        def status(self) -> OpenShellCommandResult:
            return OpenShellCommandResult(ok=ok, command=[], output=output)

    monkeypatch.setattr(tool, "OpenShellLifecycle", FakeLifecycle)
    assert sandbox_is_ready(object()) is ready  # type: ignore[arg-type]


def _make_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    (project / "agents_lab" / "agents" / "finance").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (project / "uv.lock").write_text("", encoding="utf-8")
    (project / "agents_lab" / "agents" / "finance" / "workflow.yaml").write_text(
        "_type: react_agent\n", encoding="utf-8"
    )
    (project / "Dockerfile.byo").write_text('FROM python:3.11-slim\nCMD ["true"]\n', encoding="utf-8")
    (project / ".git").mkdir()
    return project


# --- config model -------------------------------------------------------------------------


def test_spec_uses_supplied_binaries() -> None:
    spec = RelayVictimSpec(
        project_dir="../agents-lab", dockerfile="deploy/finance/Dockerfile", victim_binaries=["/app/.venv/bin/**"]
    )
    assert spec.victim_binaries == ["/app/.venv/bin/**"]


def test_spec_requires_a_dockerfile() -> None:
    """Agent Hardener hardens the agent the user ships, so it never builds the image itself."""
    with pytest.raises(ValidationError):
        RelayVictimSpec(project_dir="../agents-lab", victim_binaries=["/x/**"])


def test_atof_path_is_derived_from_relay_artifacts() -> None:
    spec = RelayVictimSpec(
        project_dir="p", dockerfile="Dockerfile", victim_binaries=["/x/**"], relay_artifacts="out/relay"
    )
    assert str(spec.atof_path) == "out/relay/events.atof.jsonl"


def test_spec_requires_victim_binaries() -> None:
    """Binary globs scope which processes may egress; an unknown image layout cannot imply them."""
    with pytest.raises(ValidationError):
        RelayVictimSpec(project_dir="../agents-lab", dockerfile="deploy/finance/Dockerfile")


def test_legacy_config_has_no_relay_victim() -> None:
    config = openshell_config(load_config(str(REPO_ROOT / "tests/fixtures/pipeline_e2e/pipeline.yaml")).victim_control)
    assert config.relay_victim is None


def test_stage_copies_project_dockerfile_and_injects_shim(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    spec = RelayVictimSpec(project_dir=project, dockerfile="Dockerfile.byo", victim_binaries=["/app/.venv/bin/**"])
    build = tmp_path / "build"
    staged = stage_relay_victim_build(spec, tmp_path, build)

    staged_text = staged.read_text(encoding="utf-8")
    # User's Dockerfile is preserved, with the proxy shim appended.
    assert staged_text.startswith((project / "Dockerfile.byo").read_text(encoding="utf-8"))
    assert "COPY openshell-shims/ /app/openshell-shims/" in staged_text
    shim = (build / "openshell-shims" / "sitecustomize.py").read_text(encoding="utf-8")
    assert 'kw.setdefault("trust_env", True)' in shim


def test_stage_leaves_dotenv_and_the_secrets_file_out_of_the_build_context(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    (project / ".env").write_text("INFERENCE_API_KEY=secret\n", encoding="utf-8")
    (project / "config").mkdir()
    (project / "config" / "prod.env").write_text("GITHUB_TOKEN=secret\n", encoding="utf-8")
    (project / ".env.example").write_text("INFERENCE_API_KEY=\n", encoding="utf-8")
    spec = RelayVictimSpec(
        project_dir=project,
        dockerfile="Dockerfile.byo",
        victim_binaries=["/app/.venv/bin/**"],
        secrets_file=project / "config" / "prod.env",
    )
    build = tmp_path / "build"
    stage_relay_victim_build(spec, tmp_path, build)

    assert not (build / ".env").exists()
    assert not (build / "config" / "prod.env").exists()
    assert (build / ".env.example").is_file()
    dockerignore = (build / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in dockerignore
    assert "config/prod.env" in dockerignore


def test_stage_extends_the_projects_dockerignore_instead_of_replacing_it(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    (project / ".dockerignore").write_text("*.pem\nsecrets/\n.git", encoding="utf-8")
    spec = RelayVictimSpec(project_dir=project, dockerfile="Dockerfile.byo", victim_binaries=["/app/.venv/bin/**"])
    stage_relay_victim_build(spec, tmp_path, tmp_path / "build")

    lines = (tmp_path / "build" / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert lines[:3] == ["*.pem", "secrets/", ".git"]
    assert lines.count(".git") == 1
    assert {".venv", "__pycache__", "*.pyc", ".agent-hardener", ".env"} <= set(lines)


def test_stage_ignores_a_secrets_file_outside_the_project(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    spec = RelayVictimSpec(
        project_dir=project, dockerfile="Dockerfile.byo", victim_binaries=["/app/.venv/bin/**"], secrets_file=".env"
    )
    (tmp_path / ".env").write_text("INFERENCE_API_KEY=secret\n", encoding="utf-8")
    stage_relay_victim_build(spec, tmp_path, tmp_path / "build")

    assert ".env" in (tmp_path / "build" / ".dockerignore").read_text(encoding="utf-8").splitlines()


def test_stage_ships_the_guardrail_plugin_into_the_image(tmp_path: Path) -> None:
    """The guardrail runs in the victim's process, so its code has to be in the victim's image.

    Copied beside the proxy shim rather than pip-installed: an install would pull Agent Hardener's whole
    dependency tree into the agent we are hardening, and would need private package-index credentials inside
    the user's Docker build.
    """
    project = _make_project(tmp_path)
    spec = RelayVictimSpec(project_dir=project, dockerfile="Dockerfile.byo", victim_binaries=["/app/.venv/bin/**"])
    build = tmp_path / "build"
    stage_relay_victim_build(spec, tmp_path, build)

    staged_package = build / "openshell-shims" / GUARDRAIL_PACKAGE
    assert (staged_package / "plugin.py").is_file()
    assert (staged_package / "policy.py").is_file()
    assert (staged_package / "victim.py").is_file()
    # Relative imports are what let the same source run under a different top-level name.
    assert "from .config import" in (staged_package / "plugin.py").read_text(encoding="utf-8")


def test_stage_is_repeatable(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    spec = RelayVictimSpec(project_dir=project, dockerfile="Dockerfile.byo", victim_binaries=["/app/.venv/bin/**"])
    build = tmp_path / "build"
    stage_relay_victim_build(spec, tmp_path, build)
    stage_relay_victim_build(spec, tmp_path, build)  # build_root already exists -> replaced cleanly
    assert (build / "Dockerfile").exists()


def test_stage_missing_project_dir_raises(tmp_path: Path) -> None:
    spec = RelayVictimSpec(project_dir="nope", dockerfile="Dockerfile", victim_binaries=["/x/**"])
    with pytest.raises(VictimBuildError, match="project_dir"):
        stage_relay_victim_build(spec, tmp_path, tmp_path / "build")


def test_stage_missing_dockerfile_raises(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    spec = RelayVictimSpec(project_dir=project, dockerfile="missing/Dockerfile", victim_binaries=["/x/**"])
    with pytest.raises(VictimBuildError, match="dockerfile"):
        stage_relay_victim_build(spec, tmp_path, tmp_path / "build")


# --- policy egress ------------------------------------------------------------------------


def _policy(tmp_path: Path) -> Path:
    src = tmp_path / "policy.yaml"
    src.write_text(yaml.safe_dump({"version": 1, "network_policies": {"existing": {"name": "x"}}}), encoding="utf-8")
    return src


def test_inject_backend_egress_adds_block(tmp_path: Path) -> None:
    dest = tmp_path / "out.yaml"
    endpoints = [BackendEndpoint(port=8086, allowed_ips=["172.17.0.1/32"])]
    inject_backend_egress(_policy(tmp_path), endpoints, ["/app/project/.venv/bin/**"], dest)

    policy = yaml.safe_load(dest.read_text(encoding="utf-8"))
    assert "existing" in policy["network_policies"]
    block = policy["network_policies"]["relay_victim_backends"]
    assert block["endpoints"][0]["host"] == "host.docker.internal"
    assert block["endpoints"][0]["port"] == 8086
    assert block["endpoints"][0]["allowed_ips"] == ["172.17.0.1/32"]
    assert block["binaries"] == [{"path": "/app/project/.venv/bin/**"}]


def test_inject_backend_egress_is_idempotent(tmp_path: Path) -> None:
    dest = tmp_path / "out.yaml"
    endpoints = [BackendEndpoint(port=8086)]
    inject_backend_egress(_policy(tmp_path), endpoints, ["/x/**"], dest)
    inject_backend_egress(dest, endpoints, ["/x/**"], dest)

    policy = yaml.safe_load(dest.read_text(encoding="utf-8"))
    assert len(policy["network_policies"]["relay_victim_backends"]["endpoints"]) == 1


def test_build_backend_egress_block_omits_empty_allowed_ips() -> None:
    block = build_backend_egress_block([BackendEndpoint(port=9000)], ["/x/**"])
    assert "allowed_ips" not in block["endpoints"][0]


def test_inject_backend_egress_no_endpoints_copies_through(tmp_path: Path) -> None:
    dest = tmp_path / "out.yaml"
    inject_backend_egress(_policy(tmp_path), [], ["/x/**"], dest)
    policy = yaml.safe_load(dest.read_text(encoding="utf-8"))
    assert "relay_victim_backends" not in policy["network_policies"]


# --- backend manager ----------------------------------------------------------------------


def test_backend_manager_up_builds_and_downs(tmp_path: Path) -> None:
    runner = RecordingRunner()
    manager = BackendManager(runner=runner, compose_command=["docker", "compose"])
    specs = [BackendServiceSpec(name="finance", compose_file="svc/docker-compose.yaml", compose_project="fin")]

    up = manager.up(specs, tmp_path)
    assert up.ok
    assert runner.commands[0][:7] == [
        "docker",
        "compose",
        "-f",
        str(tmp_path / "svc/docker-compose.yaml"),
        "-p",
        "fin",
        "up",
    ]

    manager.down(specs, tmp_path)
    assert runner.commands[-1][-2:] == ["down", "-v"]


def test_backend_manager_up_aborts_on_failure(tmp_path: Path) -> None:
    runner = RecordingRunner(returncode=1)
    manager = BackendManager(runner=runner, compose_command=["docker", "compose"])
    specs = [
        BackendServiceSpec(name="a", compose_file="a/docker-compose.yaml"),
        BackendServiceSpec(name="b", compose_file="b/docker-compose.yaml"),
    ]
    result = manager.up(specs, tmp_path)
    assert not result.ok
    assert len(runner.commands) == 1  # second spec never attempted


def test_backend_manager_noop_without_compose_file(tmp_path: Path) -> None:
    runner = RecordingRunner()
    manager = BackendManager(runner=runner, compose_command=["docker", "compose"])
    result = manager.up([BackendServiceSpec(name="x")], tmp_path)
    assert result.ok
    assert runner.commands == []
    assert manager.down([BackendServiceSpec(name="x")], tmp_path).ok  # down skips specs without compose


def test_backend_manager_waits_for_health(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, float]] = []

    def _record(url: str, timeout_seconds: float) -> None:
        calls.append((url, timeout_seconds))

    monkeypatch.setattr("agent_hardener.providers.backends.wait_for_http_health", _record)
    manager = BackendManager(runner=RecordingRunner(), compose_command=["docker", "compose"])
    spec = BackendServiceSpec(name="f", compose_file="c.yaml", health_url="http://127.0.0.1:8086/health")
    assert manager.up([spec], tmp_path).ok
    assert calls == [("http://127.0.0.1:8086/health", 120.0)]


def test_backend_manager_health_timeout_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(url: str, timeout_seconds: float) -> None:
        raise TimeoutError("nope")

    monkeypatch.setattr("agent_hardener.providers.backends.wait_for_http_health", _raise)
    manager = BackendManager(runner=RecordingRunner(), compose_command=["docker", "compose"])
    spec = BackendServiceSpec(name="f", compose_file="c.yaml", health_url="http://127.0.0.1:8086/health")
    assert not manager.up([spec], tmp_path).ok


def test_compose_args_requires_compose_file(tmp_path: Path) -> None:
    manager = BackendManager(runner=RecordingRunner(), compose_command=["docker", "compose"])
    with pytest.raises(ValueError, match="no compose_file"):
        manager._compose_args(BackendServiceSpec(name="x"), tmp_path)


def test_resolve_compose_command(monkeypatch: pytest.MonkeyPatch) -> None:
    # docker present AND `docker compose` works -> prefer the v2 plugin.
    monkeypatch.setattr(backends.shutil, "which", lambda name: "/usr/bin/docker" if name == "docker" else None)
    monkeypatch.setattr(backends, "_docker_compose_works", lambda: True)
    assert backends.resolve_compose_command() == ["docker", "compose"]

    # docker present but `docker compose` broken, standalone docker-compose available -> use v1.
    monkeypatch.setattr(backends.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(backends, "_docker_compose_works", lambda: False)
    assert backends.resolve_compose_command() == ["docker-compose"]

    # nothing available -> fallback.
    monkeypatch.setattr(backends.shutil, "which", lambda _name: None)
    monkeypatch.setattr(backends, "_docker_compose_works", lambda: False)
    assert backends.resolve_compose_command() == ["docker", "compose"]


def test_backend_manager_handles_none_stdout(tmp_path: Path) -> None:
    class NoneStdoutRunner:
        def run(self, args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout=None)

    manager = BackendManager(runner=NoneStdoutRunner(), compose_command=["docker", "compose"])
    result = manager.up([BackendServiceSpec(name="f", compose_file="c.yaml")], tmp_path)
    assert result.ok
    assert result.output == ""


# --- prepare_relay_victim -------------------------------------------------------------------


def test_prepare_relay_victim_passthrough_without_spec() -> None:
    config = openshell_config(load_config(str(REPO_ROOT / "tests/fixtures/pipeline_e2e/pipeline.yaml")).victim_control)
    assert prepare_relay_victim(config) is config


def test_prepare_relay_victim_stages_and_patches_policy(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    policy = _policy(tmp_path)
    spec = RelayVictimSpec(
        project_dir=project,
        dockerfile="Dockerfile.byo",
        victim_binaries=["/app/.venv/bin/**"],
        backends=[BackendServiceSpec(name="finance", compose_file="c.yaml", allowlist=[BackendEndpoint(port=8086)])],
    )
    base = OpenShellConfig(gateway="g", sandbox="sbx", policy_path=policy, cwd=tmp_path, relay_victim=spec)
    prepared = prepare_relay_victim(base)

    assert prepared.build_context == tmp_path / ".agent-hardener" / "builds" / "sbx" / "Dockerfile"
    assert prepared.build_context.exists()
    patched = yaml.safe_load(prepared.policy_path.read_text(encoding="utf-8"))
    assert "relay_victim_backends" in patched["network_policies"]


def test_the_launch_command_pins_the_in_sandbox_atof_path(tmp_path: Path) -> None:
    """A sandboxed victim writes to a fixed path, not to the manifest's host-side one.

    The two are the ends of a copy, not one shared location: the sandbox shares no filesystem with
    the host. The in-sandbox path lives under /home/sandbox because every policy template already
    grants that read-write, so telemetry needs no Landlock exception of its own — pointing it at the
    manifest path instead makes Relay fail to open the sink and takes the victim down at startup.
    """
    project = _make_project(tmp_path)
    spec = RelayVictimSpec(
        project_dir=project,
        dockerfile="Dockerfile.byo",
        victim_binaries=["/app/.venv/bin/**"],
        relay_artifacts="telemetry/relay out",
    )
    base = OpenShellConfig(
        gateway="g",
        sandbox="sbx",
        policy_path=_policy(tmp_path),
        cwd=tmp_path,
        relay_victim=spec,
        start_command="python -m app",
    )

    prepared = prepare_relay_victim(base)

    assert f"AGENT_HARDENER_RELAY_ARTIFACTS={SANDBOX_ARTIFACTS_DIR}" in prepared.start_command
    # The manifest path is where the run *lands* the file after pulling it, not where the victim writes.
    assert "telemetry/relay out" not in prepared.start_command
    assert prepared.start_command.endswith("python -m app")


def test_the_judges_endpoint_is_always_allowed_out(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    policy = _policy(tmp_path)
    spec = RelayVictimSpec(project_dir=project, dockerfile="Dockerfile.byo", victim_binaries=["/app/.venv/bin/**"])
    base = OpenShellConfig(gateway="g", sandbox="sbx", policy_path=policy, cwd=tmp_path, relay_victim=spec)
    prepared = prepare_relay_victim(base)
    # Even with nothing declared or discovered, the guardrail plugin's safety judge runs *inside*
    # the victim and must reach its endpoint. Discovery cannot find it — it is in no file the user
    # wrote — and the sandbox is default-deny, so without this the judge's call is dropped, the
    # guardrail fails open, and the run reports an attack landing against a guardrail that was
    # installed and never consulted.
    assert prepared.policy_path != policy
    patched = yaml.safe_load(prepared.policy_path.read_text(encoding="utf-8"))
    hosts = {
        endpoint["host"] for block in patched["network_policies"].values() for endpoint in block.get("endpoints", [])
    }
    assert any("nvidia.com" in host for host in hosts)
    assert prepared.build_context.exists()


def test_prepare_relay_victim_injects_discovered_egress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep discovery offline: no real DNS (allowed_ips just stays empty here).
    monkeypatch.setattr(
        "agent_hardener.openshell.egress.discoverer.socket.getaddrinfo",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("offline")),
    )
    project = _make_project(tmp_path)
    # Give the workflow a real LLM URL so discovery has something to find.
    (project / "agents_lab" / "agents" / "finance" / "workflow.yaml").write_text(
        "llms:\n  nim:\n    base_url: https://integrate.api.nvidia.com/v1/\n", encoding="utf-8"
    )
    policy = _policy(tmp_path)
    spec = RelayVictimSpec(
        project_dir=project,
        dockerfile="Dockerfile.byo",
        victim_binaries=["/app/.venv/bin/**"],
        egress=["https://api.tavily.com"],
    )
    base = OpenShellConfig(gateway="g", sandbox="sbx", policy_path=policy, cwd=tmp_path, relay_victim=spec)
    prepared = prepare_relay_victim(base)

    assert prepared.policy_path != policy  # discovery -> patched policy written
    block = yaml.safe_load(prepared.policy_path.read_text(encoding="utf-8"))["network_policies"][
        "relay_victim_discovered_egress"
    ]
    hosts = {ep["host"] for ep in block["endpoints"]}
    assert "*.api.nvidia.com" in hosts  # from the workflow
    assert "*.tavily.com" in hosts  # from manual agent.egress


def test_prepare_relay_victim_byo_prepends_shim_pythonpath(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    policy = _policy(tmp_path)
    spec = RelayVictimSpec(project_dir=project, dockerfile="Dockerfile.byo", victim_binaries=["/app/.venv/bin/**"])
    base = OpenShellConfig(
        gateway="g",
        sandbox="sbx",
        policy_path=policy,
        cwd=tmp_path,
        relay_victim=spec,
        start_command="nat serve --config_file w.yaml --host 0.0.0.0 --port 8000",
    )
    prepared = prepare_relay_victim(base)
    assert prepared.start_command.startswith("env PYTHONPATH=/app/openshell-shims:")
    assert "nat serve" in prepared.start_command


def test_prepare_relay_victim_wraps_start_command_with_shim_path(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    policy = _policy(tmp_path)
    spec = RelayVictimSpec(project_dir=project, dockerfile="Dockerfile.byo", victim_binaries=["/app/.venv/bin/**"])
    base = OpenShellConfig(
        gateway="g",
        sandbox="sbx",
        policy_path=policy,
        cwd=tmp_path,
        relay_victim=spec,
        start_command="/app/serve.sh",
    )
    prepared = prepare_relay_victim(base)
    # `openshell sandbox exec` drops image ENV, so the shim path is prepended to the user's own
    # command instead of relying on the Dockerfile's ENV PYTHONPATH.
    assert "PYTHONPATH=/app/openshell-shims" in prepared.start_command
    assert prepared.start_command.endswith("/app/serve.sh")


def test_prepare_relay_victim_discover_egress_disabled(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    (project / "agents_lab" / "agents" / "finance" / "workflow.yaml").write_text(
        "base_url: https://integrate.api.nvidia.com/v1/\n", encoding="utf-8"
    )
    policy = _policy(tmp_path)
    spec = RelayVictimSpec(
        project_dir=project,
        dockerfile="Dockerfile.byo",
        victim_binaries=["/app/.venv/bin/**"],
        discover_egress=False,
    )
    base = OpenShellConfig(gateway="g", sandbox="sbx", policy_path=policy, cwd=tmp_path, relay_victim=spec)
    prepared = prepare_relay_victim(base)
    # Discovery is off, but the judge endpoint is Agent Hardener's own dependency, not the user's.
    assert prepared.policy_path != policy


def test_stage_strips_buildkit_only_cache_mounts(tmp_path: Path) -> None:
    """OpenShell builds via the Docker API's classic builder, which rejects `RUN --mount`.

    `nemo agents package` generates those mounts, so the user has no Dockerfile to edit — and neither
    DOCKER_BUILDKIT nor an installed buildx helps, because OpenShell never invokes the docker CLI.
    A cache mount only makes the build faster, so dropping it changes nothing but the build time.
    """
    project = _make_project(tmp_path)
    (project / "Dockerfile.byo").write_text(
        "FROM python:3.12-slim\n"
        "RUN --mount=type=cache,id=uv_cache,target=/root/.cache/uv,sharing=locked \\\n"
        "    uv pip install .\n",
        encoding="utf-8",
    )
    spec = RelayVictimSpec(project_dir=project, dockerfile="Dockerfile.byo", victim_binaries=["/app/**"])

    staged = stage_relay_victim_build(spec, tmp_path, tmp_path / "build")

    text = staged.read_text(encoding="utf-8")
    assert "--mount=type=cache" not in text
    assert "uv pip install ." in text  # the command itself survives


def test_hermes_gets_its_extra_wiring_and_others_do_not(tmp_path: Path) -> None:
    """Hermes reads plugin config only from its own env var, and its Relay plugin is opt-in.

    Every other harness lets Relay discover /etc/nemo-relay on its own, so staging these lines for
    them would enable a plugin nobody asked for.
    """
    project = tmp_path / "agent"
    project.mkdir()
    (project / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")

    def staged(harness: str | None) -> str:
        spec = RelayVictimSpec(
            project_dir=project, dockerfile=Path("Dockerfile"), victim_binaries=["/app/.venv/bin/**"], harness=harness
        )
        return stage_relay_victim_build(spec, tmp_path, tmp_path / f"build-{harness}").read_text(encoding="utf-8")

    hermes = staged("hermes")
    assert "HERMES_NEMO_RELAY_PLUGINS_TOML=/etc/nemo-relay/plugins.toml" in hermes
    assert "hermes plugins enable observability/nemo_relay" in hermes

    assert "HERMES_NEMO_RELAY" not in staged("deepagents")
    assert "HERMES_NEMO_RELAY" not in staged(None)


def test_the_launcher_exports_what_the_image_declares(tmp_path: Path) -> None:
    """`openshell sandbox exec` drops the image's ENV, so anything the agent needs is exported here.

    On main this was substituted into Agent Hardener's own Dockerfile template, which only existed for
    generated images — a BYO victim, which is now the only shape, never received it.
    """
    spec = RelayVictimSpec(
        project_dir=tmp_path, dockerfile=Path("Dockerfile"), victim_binaries=["/workspace/.venv/bin/**"]
    )
    rendered = _launch_env(spec, {"BACKEND_URL": "http://host.docker.internal:8084"})

    assert "PATH=/workspace/.venv/bin:" in rendered  # the venv, so a spawned helper is not the system python
    assert "BACKEND_URL=http://host.docker.internal:8084" in rendered
    assert "AGENT_HARDENER_RELAY_ARTIFACTS=" in rendered


def test_the_images_path_is_kept_behind_the_victims_own_bin(tmp_path: Path) -> None:
    """The venv goes first so a helper spawned by name is the agent's interpreter, not the system's.

    Everything the image declared stays behind it, rather than being replaced.
    """
    spec = RelayVictimSpec(
        project_dir=tmp_path, dockerfile=Path("Dockerfile"), victim_binaries=["/workspace/.venv/bin/**"]
    )

    rendered = _launch_env(spec, {"PATH": "/opt/custom/bin"})

    assert "PATH=/workspace/.venv/bin:/opt/custom/bin" in rendered


def test_the_images_own_env_is_carried_into_the_sandbox(tmp_path: Path) -> None:
    """The victim must run in the environment its author described, not one Agent Hardener invented.

    `openshell sandbox exec` discards image ENV. Fabric relies on it: the adapter host is spawned by
    name, so without the venv on PATH it resolves to the system interpreter and cannot import itself.
    """
    spec = RelayVictimSpec(
        project_dir=tmp_path, dockerfile=Path("Dockerfile"), victim_binaries=["/workspace/.venv/bin/**"]
    )
    image_env = {"PATH": "/workspace/.venv/bin:$PATH", "AGENT_CONFIG_PATH": "/workspace/agent.yaml", "PORT": "8000"}

    rendered = _launch_env(spec, image_env)

    assert "AGENT_CONFIG_PATH=/workspace/agent.yaml" in rendered
    assert "PORT=8000" in rendered
    assert "/workspace/.venv/bin:$PATH" in rendered  # the image's own value, kept rather than replaced


def test_dockerfile_env_reads_both_forms(tmp_path: Path) -> None:
    """Docker accepts `ENV K=V` and the legacy `ENV K value with spaces`; so must we."""
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "FROM python:3.12-slim\n"
        "# a comment\n"
        'ENV PATH="/workspace/.venv/bin:$PATH"\n'
        "ENV A=1 B=2\n"
        "ENV LEGACY some value\n"
        "ENV MULTI=x \\\n"
        "    Y=z\n",
        encoding="utf-8",
    )

    assert dockerfile_env(dockerfile) == {
        "PATH": "/workspace/.venv/bin:$PATH",
        "A": "1",
        "B": "2",
        "LEGACY": "some value",
        "MULTI": "x",
        "Y": "z",
    }


def test_the_manifest_env_is_baked_into_the_composed_dockerfile(tmp_path: Path) -> None:
    """One source of truth for the victim's environment: the image states it, everything reads it."""
    project = tmp_path / "agent"
    project.mkdir()
    (project / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")
    spec = RelayVictimSpec(
        project_dir=project,
        dockerfile=Path("Dockerfile"),
        victim_binaries=["/app/.venv/bin/**"],
        agent_env={"BACKEND_URL": "https://ledger.internal", "MODE": "prod"},
    )

    staged = stage_relay_victim_build(spec, tmp_path, tmp_path / "build")

    assert dockerfile_env(staged)["BACKEND_URL"] == "https://ledger.internal"
    assert dockerfile_env(staged)["MODE"] == "prod"


def test_egress_is_discovered_from_the_images_env(tmp_path: Path) -> None:
    """A backend URL the agent was configured with is in no file it wrote — only in its environment.

    Without this the agent receives the variable and is then refused the connection by its own
    sandbox policy.
    """
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "FROM python:3.12-slim\nENV BACKEND_URL=https://ledger.internal:8443\nENV MODE=prod\n", encoding="utf-8"
    )

    found = DockerfileEnvEgressSource(dockerfile).discover()

    assert {(e.host, e.port) for e in found} == {("ledger.internal", 8443)}
    assert DockerfileEnvEgressSource(tmp_path / "absent").discover() == set()


def test_the_launcher_does_not_duplicate_what_the_image_already_leads_with(tmp_path: Path) -> None:
    """The Dockerfile carries our own additions, so prepending blindly duplicates every entry."""
    spec = RelayVictimSpec(
        project_dir=tmp_path, dockerfile=Path("Dockerfile"), victim_binaries=["/workspace/.venv/bin/**"]
    )
    image_env = {"PATH": "/workspace/.venv/bin:/usr/bin", "PYTHONPATH": "/app/openshell-shims:"}

    rendered = _launch_env(spec, image_env)

    assert "PATH=/workspace/.venv/bin:/usr/bin" in rendered
    assert "/app/openshell-shims:/app/openshell-shims" not in rendered


def test_tcp_backend_endpoint_carries_no_request_level_fields() -> None:
    block = build_backend_egress_block([BackendEndpoint(port=5432, protocol="tcp")], ["/usr/local/bin/python*"])
    assert block["endpoints"] == [{"host": "host.docker.internal", "port": 5432, "protocol": "tcp"}]


def test_reconcile_unions_allowed_ips_across_blocks_on_one_host_port() -> None:
    policy = {
        "network_policies": {
            "a": {"endpoints": [{"host": "*.nvidia.com", "port": 443}]},
            "b": {"endpoints": [{"host": "*.nvidia.com", "port": 443, "allowed_ips": ["10.0.0.1"]}]},
            "c": {"endpoints": [{"host": "*.nvidia.com", "port": 80}]},
        }
    }
    reconcile_shared_endpoint_settings(policy)
    blocks = policy["network_policies"]
    assert blocks["a"]["endpoints"][0]["allowed_ips"] == ["10.0.0.1"]
    assert "allowed_ips" not in blocks["c"]["endpoints"][0]


def test_apply_victim_binaries_deduplicates_into_fresh_entries() -> None:
    policy = {"network_policies": {"a": {"binaries": [{"path": "/usr/local/bin/python*"}]}}}
    apply_victim_binaries(policy, ["/usr/local/bin/python*"])
    assert policy["network_policies"]["a"]["binaries"] == [
        {"path": "/usr/local/bin/python*"},
        {"path": "/usr/bin/python*"},
    ]
    assert "&id" not in yaml.safe_dump(policy)


def test_bind_provider_credentials_binds_only_inspected_endpoints() -> None:
    endpoints = [
        {"host": "a.example.com", "port": 443, "protocol": "rest"},
        {"host": "b.example.com", "port": 443, "protocol": "rest", "tls": "skip"},
        {"host": "db.example.com", "port": 5432, "protocol": "tcp"},
        {"host": "host.docker.internal", "port": 4318},
    ]
    policy = {"network_policies": {"a": {"endpoints": endpoints}}}

    bind_provider_credentials(policy, "secrets")

    assert [e.get("credential_binding") for e in endpoints] == [{"provider": "secrets"}, None, None, None]


def test_backend_manager_health_gates_a_backend_it_does_not_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, float]] = []
    monkeypatch.setattr(
        "agent_hardener.providers.backends.wait_for_http_health",
        lambda url, timeout_seconds: calls.append((url, timeout_seconds)),
    )
    runner = RecordingRunner()
    manager = BackendManager(runner=runner, compose_command=["docker", "compose"])
    spec = BackendServiceSpec(name="finance", health_url="http://finance.ns.svc:8086/health")

    assert manager.up([spec], tmp_path).ok
    assert runner.commands == []
    assert calls == [("http://finance.ns.svc:8086/health", 120.0)]

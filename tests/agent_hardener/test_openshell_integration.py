# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from agent_hardener.agents.victims.openshell_victim import build_payload
from agent_hardener.config import parse_config_data
from agent_hardener.models import (
    AgentConfig,
    AgentRunInput,
    TargetInput,
)
from agent_hardener.openshell.lifecycle import (
    OpenShellConfig,
    OpenShellLifecycle,
    is_transient_transport_failure,
    read_env_file,
)
from agent_hardener.runtime.adapters import _policy_digest
from agent_hardener.swarm_tracker import read_victim_image, write_victim_image
from agent_hardener.tools.openshell import main as openshell_tool_main

REPO_ROOT = Path(__file__).resolve().parents[2]


_NOT_FOUND = "code: 'Some requested entity was not found', message: \"sandbox not found\""


class FakeRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.deleting = False

    def run(self, args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        self.commands.append(args)
        if args[1:3] == ["sandbox", "delete"]:
            self.deleting = True
        elif self.deleting and args[1:3] == ["sandbox", "get"]:
            # The first status poll after an (async) delete sees the sandbox gone.
            self.deleting = False
            return subprocess.CompletedProcess(args=args, returncode=1, stdout=_NOT_FOUND)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok")


class SequenceRunner:
    def __init__(self, responses: list[tuple[int, str]]) -> None:
        self.commands: list[list[str]] = []
        self.responses = responses

    def run(self, args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        self.commands.append(args)
        returncode, stdout = self.responses.pop(0)
        return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout)


def _request() -> AgentRunInput:
    return AgentRunInput(round_id="round-0001", target=TargetInput(name="target"))


def test_config_accepts_top_level_openshell_victim_control(tmp_path: Path) -> None:
    config = parse_config_data(
        {
            "storage": {"root_dir": str(tmp_path)},
            "victim_control": {
                "type": "openshell",
                "config": {"gateway": "gw", "sandbox": "sb", "policy_path": str(tmp_path / "policy.yaml")},
            },
            "target": {"name": "target"},
            "victim": {"name": "victim"},
        },
    )

    assert config.victim_control.type == "openshell"
    assert config.victim_control.config["gateway"] == "gw"


def test_openshell_config_from_mapping_preserves_uploads_and_create_command(tmp_path: Path) -> None:
    upload = f"{tmp_path / 'workflow.yaml'}:/tmp/research_agent_workflow.yaml"

    config = OpenShellConfig.from_mapping(
        {
            "gateway": "gw",
            "sandbox": "sb",
            "policy_path": str(tmp_path / "policy.yaml"),
            "uploads": [upload],
            "create_command": ["/bin/sh", "-lc", "while true; do sleep 3600; done"],
            "policy_wait_timeout": 90,
        }
    )

    assert config.uploads == [upload]
    assert config.create_command == ["/bin/sh", "-lc", "while true; do sleep 3600; done"]
    assert config.policy_wait_timeout == 90


def test_openshell_config_from_mapping_defaults_create_command(tmp_path: Path) -> None:
    config = OpenShellConfig.from_mapping(
        {
            "gateway": "gw",
            "sandbox": "sb",
            "policy_path": str(tmp_path / "policy.yaml"),
        }
    )

    assert config.create_command == ["sleep", "infinity"]


def test_lifecycle_builds_policy_apply_and_restart_commands(tmp_path: Path) -> None:
    runner = FakeRunner()
    lifecycle = OpenShellLifecycle(
        OpenShellConfig(gateway="gw", sandbox="sb", policy_path=tmp_path / "policy.yaml", openshell_bin="openshell"),
        runner=runner,
    )

    assert lifecycle.apply_policy(tmp_path / "candidate.yaml").ok is True
    assert lifecycle.active_policy().ok is True
    assert lifecycle.restart().ok is True

    assert runner.commands[0] == [
        "openshell",
        "policy",
        "set",
        "sb",
        "--gateway",
        "gw",
        "--policy",
        str(tmp_path / "candidate.yaml"),
        "--wait",
        "--timeout",
        "60",
    ]
    assert runner.commands[1] == ["openshell", "policy", "get", "sb", "--gateway", "gw", "--base"]
    assert runner.commands[2][:4] == ["openshell", "sandbox", "delete", "sb"]
    assert runner.commands[3] == ["openshell", "sandbox", "get", "sb", "--gateway", "gw"]  # delete wait


def test_lifecycle_uploads_file_and_restarts_victim_in_place(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_hardener.openshell._polling.wait_for_http_health", lambda _url, _timeout: None)
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text("workflow: {}\n", encoding="utf-8")
    runner = FakeRunner()
    lifecycle = OpenShellLifecycle(
        OpenShellConfig(
            gateway="gw",
            sandbox="sb",
            policy_path=tmp_path / "policy.yaml",
            openshell_bin="openshell",
            start_command="/app/start-agents-lab.sh >/tmp/agents-lab.log 2>&1",
            health_url="http://127.0.0.1:8000/health",
        ),
        runner=runner,
    )

    assert lifecycle.upload_file(workflow, "/tmp/research_agent_workflow.yaml").ok is True
    assert lifecycle.restart_victim().ok is True

    assert runner.commands[0] == [
        "openshell",
        "sandbox",
        "upload",
        "--gateway",
        "gw",
        "--no-git-ignore",
        "sb",
        str(workflow),
        "/tmp/research_agent_workflow.yaml",
    ]
    assert runner.commands[1][1:4] == ["sandbox", "exec", "--gateway"]
    assert "victim pid file missing" in runner.commands[1][runner.commands[1].index("-lc") + 1]
    assert runner.commands[2][1:4] == ["sandbox", "exec", "--gateway"]
    assert "exec $1" in runner.commands[2][runner.commands[2].index("-lc") + 1]
    assert "/tmp/agent.pid" in runner.commands[2][runner.commands[2].index("-lc") + 1]


def test_lifecycle_restart_tolerates_missing_sandbox_delete(tmp_path: Path) -> None:
    runner = SequenceRunner(
        [
            (1, 'status: NotFound, message: "sandbox not found"'),
            (0, "gateway ok"),
            (1, 'status: NotFound, message: "sandbox not found"'),
            (0, "image built"),
            (0, "sandbox created"),
            (0, "Phase: Ready"),
        ]
    )
    lifecycle = OpenShellLifecycle(
        OpenShellConfig(
            gateway="gw",
            sandbox="sb",
            policy_path=tmp_path / "policy.yaml",
            openshell_bin="openshell",
            build_context=tmp_path,
        ),
        runner=runner,
    )

    assert lifecycle.restart(wait_for_health=False).ok is True


def test_lifecycle_reports_missing_openshell_binary(tmp_path: Path) -> None:
    class MissingBinaryRunner:
        def run(self, args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
            raise FileNotFoundError("openshell")

    lifecycle = OpenShellLifecycle(
        OpenShellConfig(gateway="gw", sandbox="sb", policy_path=tmp_path / "policy.yaml", openshell_bin="openshell"),
        runner=MissingBinaryRunner(),
    )

    result = lifecycle.status()

    assert result.ok is False
    assert result.command == ["openshell", "sandbox", "get", "sb", "--gateway", "gw"]
    assert "openshell" in result.output


@pytest.mark.unit
def test_ensure_gateway_started_only_checks_the_service_gateway(tmp_path: Path) -> None:
    """The gateway is a system service: a down gateway is reported, never "started"."""
    runner = SequenceRunner([(1, "gateway not connected")])
    lifecycle = OpenShellLifecycle(
        OpenShellConfig(gateway="gw", sandbox="sb", policy_path=tmp_path / "policy.yaml", openshell_bin="openshell"),
        runner=runner,
    )

    results = lifecycle._ensure_gateway_started()

    assert runner.commands == [["openshell", "gateway", "info", "--gateway", "gw"]]
    assert results[-1].ok is False


def test_read_env_file_parses_dotenv_values(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
# local secrets
NIM_API_KEY="nim-key"
INFERENCE_API_KEY='inference-key'
EMPTY=
IGNORED
""",
        encoding="utf-8",
    )

    assert read_env_file(env_file) == {
        "NIM_API_KEY": "nim-key",
        "INFERENCE_API_KEY": "inference-key",
        "EMPTY": "",
    }


def test_lifecycle_creates_provider_from_env_file_before_sandbox_create(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "NIM_API_KEY=nim-key\nINFERENCE_API_KEY=inference-key\nVLLM_API_KEY=dummy\nGITHUB_TOKEN=github-key\n",
        encoding="utf-8",
    )
    runner = SequenceRunner(
        [
            (0, "gateway ok"),
            (1, "sandbox missing"),
            (1, "provider missing"),
            (0, "profile imported"),
            (0, "provider created"),
            (0, "image built"),
            (0, "sandbox created"),
            (0, "Phase: Ready"),
        ]
    )
    lifecycle = OpenShellLifecycle(
        OpenShellConfig(
            gateway="gw",
            sandbox="sb",
            policy_path=tmp_path / "policy.yaml",
            openshell_bin="openshell",
            build_context=tmp_path,
            provider="agents-lab-secrets",
            provider_env_file=env_file,
            provider_credentials=["NIM_API_KEY", "INFERENCE_API_KEY", "VLLM_API_KEY", "GITHUB_TOKEN"],
        ),
        runner=runner,
    )

    assert lifecycle.up().ok is True

    assert runner.commands[1][1:4] == ["sandbox", "get", "sb"]
    assert runner.commands[2] == ["openshell", "provider", "get", "agents-lab-secrets", "--gateway", "gw"]
    # The "generic" provider type is backed by a profile generated for exactly these secrets.
    assert runner.commands[3][1:5] == ["provider", "profile", "import", "--file"]
    profile = yaml.safe_load(Path(runner.commands[3][5]).read_text(encoding="utf-8"))
    assert profile["id"] == "agents-lab-secrets"
    assert [c["env_vars"] for c in profile["credentials"]] == [
        ["NIM_API_KEY"],
        ["INFERENCE_API_KEY"],
        ["VLLM_API_KEY"],
        ["GITHUB_TOKEN"],
    ]
    assert "endpoints" not in profile  # the sandbox policy, not the profile, grants egress
    assert runner.commands[4] == [
        "openshell",
        "provider",
        "create",
        "--name",
        "agents-lab-secrets",
        "--gateway",
        "gw",
        "--type",
        "agents-lab-secrets",
        "--credential",
        "NIM_API_KEY=nim-key",
        "--credential",
        "INFERENCE_API_KEY=inference-key",
        "--credential",
        "VLLM_API_KEY=dummy",
        "--credential",
        "GITHUB_TOKEN=github-key",
    ]
    assert runner.commands[5][:2] == ["docker", "build"]
    assert "--provider" in runner.commands[6]


def test_provider_carries_the_safety_judge_key_alongside_the_victim_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The judge runs inside the victim, so its own key must reach the sandbox with the victim's."""
    monkeypatch.setenv("AGENT_HARDENER_SAFETY_API_KEY", "safety-key")
    env_file = tmp_path / ".env"
    env_file.write_text("INFERENCE_API_KEY=inference-key\n", encoding="utf-8")
    lifecycle = OpenShellLifecycle(
        OpenShellConfig(
            gateway="gw",
            sandbox="sb",
            policy_path=tmp_path / "policy.yaml",
            provider="p",
            provider_env_file=env_file,
            provider_credentials=["INFERENCE_API_KEY"],
        ),
        runner=SequenceRunner([]),
    )

    credentials, error = lifecycle._provider_credentials()

    assert error is None
    assert credentials == {"INFERENCE_API_KEY": "inference-key", "AGENT_HARDENER_SAFETY_API_KEY": "safety-key"}


def test_provider_carries_the_safety_judge_key_without_a_victim_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_HARDENER_SAFETY_API_KEY", "safety-key")
    lifecycle = OpenShellLifecycle(
        OpenShellConfig(gateway="gw", sandbox="sb", policy_path=tmp_path / "policy.yaml", provider="p"),
        runner=SequenceRunner([]),
    )

    assert lifecycle._provider_credentials() == ({"AGENT_HARDENER_SAFETY_API_KEY": "safety-key"}, None)


def test_existing_provider_gets_the_safety_judge_key_by_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider outlives runs, so a reused one must still receive the key, and never on argv."""
    monkeypatch.setenv("AGENT_HARDENER_SAFETY_API_KEY", "safety-key")
    runner = SequenceRunner([(0, "provider exists"), (0, "profile imported"), (0, "provider updated")])
    lifecycle = OpenShellLifecycle(
        OpenShellConfig(
            gateway="gw",
            sandbox="sb",
            policy_path=tmp_path / "policy.yaml",
            provider="p",
            provider_type="generic",
            provider_credentials=["INFERENCE_API_KEY"],
        ),
        runner=runner,
        log_dir=tmp_path,
    )

    assert lifecycle.ensure_provider().ok is True

    profile = yaml.safe_load(Path(runner.commands[1][5]).read_text(encoding="utf-8"))
    assert [c["env_vars"] for c in profile["credentials"]] == [["INFERENCE_API_KEY"], ["AGENT_HARDENER_SAFETY_API_KEY"]]
    assert runner.commands[2] == [
        "openshell",
        "provider",
        "update",
        "p",
        "--gateway",
        "gw",
        "--credential",
        "AGENT_HARDENER_SAFETY_API_KEY",
    ]
    assert not any("safety-key" in part for command in runner.commands for part in command)


def test_existing_provider_is_left_alone_without_a_safety_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_HARDENER_SAFETY_API_KEY", raising=False)
    runner = SequenceRunner([(0, "provider exists")])
    lifecycle = OpenShellLifecycle(
        OpenShellConfig(gateway="gw", sandbox="sb", policy_path=tmp_path / "policy.yaml", provider="p"),
        runner=runner,
    )

    assert lifecycle.ensure_provider().ok is True
    assert len(runner.commands) == 1


def test_generic_profile_falls_back_to_update_when_already_imported(tmp_path: Path) -> None:
    """OpenShell rejects a profile update that does not carry the current resource_version."""
    runner = SequenceRunner(
        [(1, "provider profile import failed"), (0, "id: p\nresource_version: 3\n"), (0, "updated")]
    )
    lifecycle = OpenShellLifecycle(
        OpenShellConfig(gateway="gw", sandbox="sb", policy_path=tmp_path / "policy.yaml", provider="p"),
        runner=runner,
        log_dir=tmp_path,
    )

    assert lifecycle._ensure_generic_profile(["INFERENCE_API_KEY"]).ok is True
    assert runner.commands[1][1:5] == ["provider", "profile", "export", "p"]
    assert runner.commands[2][1:5] == ["provider", "profile", "update", "p"]
    profile = yaml.safe_load(Path(runner.commands[2][6]).read_text(encoding="utf-8"))
    assert profile["resource_version"] == 3


def test_lifecycle_skips_provider_check_when_sandbox_already_exists(tmp_path: Path) -> None:
    runner = SequenceRunner([(0, "gateway ok"), (0, "Phase: Ready")])
    lifecycle = OpenShellLifecycle(
        OpenShellConfig(
            gateway="gw",
            sandbox="sb",
            policy_path=tmp_path / "policy.yaml",
            openshell_bin="openshell",
            build_context=tmp_path,
            provider="agents-lab-secrets",
        ),
        runner=runner,
    )

    assert lifecycle.up().ok is True

    assert runner.commands == [
        ["openshell", "gateway", "info", "--gateway", "gw"],
        ["openshell", "sandbox", "get", "sb", "--gateway", "gw"],
    ]


def test_lifecycle_builds_image_then_creates_detached_then_uploads(tmp_path: Path) -> None:
    """`--from` takes an image ref, and `--upload` can't go with a main command."""
    workflow = tmp_path / "research_agent_workflow.yaml"
    workflow.write_text("workflow: {}\n", encoding="utf-8")
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")
    runner = SequenceRunner(
        [
            (0, "gateway ok"),
            (1, "sandbox missing"),
            (0, "image built"),
            (0, "sandbox created"),
            (0, "Phase: Ready"),
            (0, "uploaded"),
        ]
    )
    lifecycle = OpenShellLifecycle(
        OpenShellConfig(
            gateway="gw",
            sandbox="sb",
            policy_path=tmp_path / "policy.yaml",
            openshell_bin="openshell",
            build_context=dockerfile,
            uploads=[f"{workflow}:/tmp/research_agent_workflow.yaml"],
        ),
        runner=runner,
    )

    assert lifecycle.up().ok is True

    assert runner.commands[2] == [
        "docker",
        "build",
        "-t",
        "agent-hardener/sb:latest",
        "-f",
        str(dockerfile),
        str(tmp_path),
    ]
    create_command = runner.commands[3]
    assert create_command[1:3] == ["sandbox", "create"]
    assert create_command[create_command.index("--from") + 1] == "agent-hardener/sb:latest"
    assert "--upload" not in create_command
    assert "--detach" in create_command
    assert create_command[create_command.index("--") + 1 :] == ["sleep", "infinity"]
    assert runner.commands[5] == [
        "openshell",
        "sandbox",
        "upload",
        "--gateway",
        "gw",
        "--no-git-ignore",
        "sb",
        str(workflow),
        "/tmp/research_agent_workflow.yaml",
    ]


def test_lifecycle_passes_an_image_reference_build_context_through(tmp_path: Path) -> None:
    lifecycle = OpenShellLifecycle(
        OpenShellConfig(
            gateway="gw",
            sandbox="sb",
            policy_path=tmp_path / "policy.yaml",
            build_context=Path("registry.example.com/team/agent:1"),
        ),
        runner=FakeRunner(),
    )

    assert lifecycle._build_image().output == "registry.example.com/team/agent:1"


def _image_lifecycle(
    tmp_path: Path, runner: SequenceRunner, *, image: str, with_context: bool = True
) -> OpenShellLifecycle:
    if with_context:
        (tmp_path / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    return OpenShellLifecycle(
        OpenShellConfig.from_mapping(
            {
                "gateway": "gw",
                "sandbox": "sb",
                "policy_path": str(tmp_path / "policy.yaml"),
                "build_context": str(tmp_path / "Dockerfile") if with_context else None,
                "image": image,
                "cwd": str(tmp_path),
            }
        ),
        runner=runner,
    )


@pytest.mark.parametrize("with_context", [True, False], ids=["staged-context-present", "image-only"])
def test_lifecycle_creates_from_victim_image_without_a_local_build(tmp_path: Path, with_context: bool) -> None:
    runner = SequenceRunner([(0, "gateway ok"), (1, "sandbox missing"), (0, "sandbox created"), (0, "Phase: Ready")])
    lifecycle = _image_lifecycle(tmp_path, runner, image="registry.example.com/team/agent:1", with_context=with_context)

    assert lifecycle.up().ok is True

    assert not any(cmd[0] == "docker" for cmd in runner.commands)
    create_command = runner.commands[2]
    assert create_command[create_command.index("--from") + 1] == "registry.example.com/team/agent:1"
    assert read_victim_image(tmp_path, "sb") == "registry.example.com/team/agent:1"


def test_lifecycle_replaces_a_sandbox_created_from_another_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("agent_hardener.openshell.support.time.sleep", lambda _seconds: None)
    write_victim_image(tmp_path, "sb", "registry.example.com/team/agent:1")
    runner = SequenceRunner(
        [
            (0, "gateway ok"),
            (0, "Phase: Ready"),
            (0, "deletion accepted"),
            (1, "sandbox not found: NotFound"),
            (0, "sandbox created"),
            (0, "Phase: Ready"),
        ]
    )
    lifecycle = _image_lifecycle(tmp_path, runner, image="registry.example.com/team/agent:2")

    assert lifecycle.up().ok is True

    verbs = [cmd[2] for cmd in runner.commands if cmd[1] == "sandbox"]
    assert verbs[:2] == ["get", "delete"]
    assert verbs.count("create") == 1
    assert read_victim_image(tmp_path, "sb") == "registry.example.com/team/agent:2"


def test_lifecycle_keeps_a_sandbox_created_from_the_same_image(tmp_path: Path) -> None:
    write_victim_image(tmp_path, "sb", "registry.example.com/team/agent:1")
    runner = SequenceRunner([(0, "gateway ok"), (0, "Phase: Ready")])
    lifecycle = _image_lifecycle(tmp_path, runner, image="registry.example.com/team/agent:1")

    assert lifecycle.up().ok is True
    assert not any(cmd[2] in {"delete", "create"} for cmd in runner.commands)


def test_a_local_build_forgets_the_recorded_image(tmp_path: Path) -> None:
    write_victim_image(tmp_path, "sb", "registry.example.com/team/agent:1")
    runner = SequenceRunner(
        [(0, "gateway ok"), (1, "sandbox missing"), (0, "built"), (0, "sandbox created"), (0, "Phase: Ready")]
    )
    (tmp_path / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    lifecycle = OpenShellLifecycle(
        OpenShellConfig(
            gateway="gw",
            sandbox="sb",
            policy_path=tmp_path / "policy.yaml",
            build_context=tmp_path / "Dockerfile",
            cwd=tmp_path,
        ),
        runner=runner,
    )

    assert lifecycle.up().ok is True
    assert read_victim_image(tmp_path, "sb") is None


def test_lifecycle_build_failure_stops_before_sandbox_create(tmp_path: Path) -> None:
    runner = SequenceRunner([(0, "gateway ok"), (1, "sandbox missing"), (1, "build failed")])
    lifecycle = OpenShellLifecycle(
        OpenShellConfig(gateway="gw", sandbox="sb", policy_path=tmp_path / "policy.yaml", build_context=tmp_path),
        runner=runner,
    )

    result = lifecycle.up()

    assert result.ok is False
    assert not any(cmd[1:3] == ["sandbox", "create"] for cmd in runner.commands)


@pytest.mark.parametrize("hardened", [True, False])
def test_sandbox_create_uploads_active_state_copy_over_seed(tmp_path: Path, hardened: bool) -> None:
    seed = tmp_path / "seed" / "plugins.toml"
    seed.parent.mkdir()
    seed.write_text("# seed\n", encoding="utf-8")
    active_state = tmp_path / "victim-active-state"
    active_state.mkdir()
    if hardened:
        (active_state / "plugins.toml").write_text("# hardened\n", encoding="utf-8")
    runner = FakeRunner()
    lifecycle = OpenShellLifecycle(
        OpenShellConfig(
            gateway="gw",
            sandbox="sb",
            policy_path=tmp_path / "policy.yaml",
            openshell_bin="openshell",
            build_context=tmp_path,
            uploads=[f"{seed}:/etc/nemo-relay/plugins.toml"],
        ),
        runner=runner,
        active_state_dir=active_state,
    )

    lifecycle._upload_seed_files()

    source = active_state / "plugins.toml" if hardened else seed
    assert runner.commands[-1][-2:] == [str(source), "/etc/nemo-relay/plugins.toml"]


def test_down_waits_until_the_async_delete_completes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_hardener.openshell.support.time.sleep", lambda _seconds: None)
    runner = SequenceRunner([(0, "deletion accepted"), (0, "Phase: Deleting"), (1, _NOT_FOUND)])
    lifecycle = OpenShellLifecycle(
        OpenShellConfig(gateway="gw", sandbox="sb", policy_path=tmp_path / "policy.yaml"), runner=runner
    )

    assert lifecycle.down().ok is True
    assert len(runner.commands) == 3


def test_logged_create_tolerates_transient_transport_reset_when_sandbox_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agent_hardener.openshell.lifecycle.time.sleep", lambda _seconds: None)
    runner = SequenceRunner(
        [
            (1, "transport error\nConnection reset by peer (os error 104)"),
            (0, "Phase: Ready"),
        ]
    )
    lifecycle = OpenShellLifecycle(
        OpenShellConfig(
            gateway="gw",
            sandbox="sb",
            policy_path=tmp_path / "policy.yaml",
            openshell_bin="openshell",
        ),
        runner=runner,
    )

    result = lifecycle._run_with_transport_retries(
        lifecycle._exec.run_logged,
        ["sandbox", "create", "--gateway", "gw"],
        delay_seconds=10.0,
        reconcile=lifecycle.status,
        warn=True,
    )

    assert result.ok is True
    assert "transient OpenShell transport failure" in result.output
    assert "sandbox exists after transient create failure" in result.output
    assert runner.commands[0][1:3] == ["sandbox", "create"]
    assert runner.commands[1][1:4] == ["sandbox", "get", "sb"]


def test_gateway_connect_502_is_transient_transport_failure() -> None:
    assert is_transient_transport_failure(
        "Error: gateway CONNECT failed with status 502\nError: ssh exited with status exit status: 255"
    )


def test_docker_build_stream_error_is_transient_transport_failure() -> None:
    assert is_transient_transport_failure("Error:\n  x Docker build stream error\n  ╰─▶ bytes remaining on stream")


def test_lifecycle_starts_victim_via_detached_sandbox_exec(tmp_path: Path) -> None:
    runner = SequenceRunner(
        [
            (0, "gateway ok"),
            (1, "sandbox missing"),
            (0, "image built"),
            (0, "sandbox created"),
            (0, "Phase: Ready"),
            (0, "started"),
            (0, "no active forwards"),
            (0, "forward started"),
            (0, "no active forwards"),
            (0, "SANDBOX   BIND      PORT   PID   STATUS\nsb  0.0.0.0 8000  123 running\n"),
        ]
    )
    lifecycle = OpenShellLifecycle(
        OpenShellConfig(
            gateway="gw",
            sandbox="sb",
            policy_path=tmp_path / "policy.yaml",
            openshell_bin="openshell",
            build_context=tmp_path,
            start_command="/app/start-agents-lab.sh >/tmp/agents-lab.log 2>&1",
            forward="0.0.0.0:8000",
        ),
        runner=runner,
    )

    assert lifecycle.up().ok is True

    assert runner.commands[4] == ["openshell", "sandbox", "get", "sb", "--gateway", "gw"]
    assert runner.commands[5][:11] == [
        "openshell",
        "sandbox",
        "exec",
        "--gateway",
        "gw",
        "--name",
        "sb",
        "--no-tty",
        "--",
        "sh",
        "-lc",
    ]
    launcher = runner.commands[5][11]
    assert "startup_log=/tmp/agent.startup.log" in launcher
    assert "pid_file=/tmp/agent.pid" in launcher
    assert 'nohup sh -lc "exec $1"' in launcher
    assert "tail -80" in launcher
    assert runner.commands[5][12:] == ["sh", "/app/start-agents-lab.sh >/tmp/agents-lab.log 2>&1"]
    assert runner.commands[6] == ["openshell", "forward", "list", "--gateway", "gw"]
    assert runner.commands[7] == [
        "openshell",
        "forward",
        "start",
        "--gateway",
        "gw",
        "--background",
        "0.0.0.0:8000",
        "sb",
    ]
    assert runner.commands[8] == ["openshell", "forward", "list", "--gateway", "gw"]


def test_managed_forward_fails_fast_when_helper_dies_immediately(tmp_path: Path) -> None:
    # OpenShell's forward helper can report launch success then immediately exit without binding the host
    # port. The managed path must catch that death fast (via the pid liveness grace period) rather than
    # leaving a dead forward for the caller's health poll to time out against.
    false_bin = shutil.which("false")  # /usr/bin/false on macOS, /bin/false on Debian; resolve don't hardcode
    assert false_bin is not None
    lifecycle = OpenShellLifecycle(
        OpenShellConfig(
            gateway="gw",
            sandbox="sb",
            policy_path=tmp_path / "policy.yaml",
            openshell_bin=false_bin,  # exits 1 immediately, standing in for a helper that dies
            forward="0.0.0.0:8000",
            cwd=tmp_path,
        ),
        runner=SequenceRunner([]),
    )

    result = lifecycle._start_managed_forward(["forward", "start", "--gateway", "gw", "0.0.0.0:8000", "sb"])

    assert result.ok is False
    assert "exited immediately" in result.output


def test_wait_for_forward_active_times_out(tmp_path: Path) -> None:
    runner = SequenceRunner(
        [
            (0, "No active forwards.\n"),
            (0, "No active forwards.\n"),
        ]
    )
    lifecycle = OpenShellLifecycle(
        OpenShellConfig(
            gateway="gw",
            sandbox="sb",
            policy_path=tmp_path / "policy.yaml",
            openshell_bin="openshell",
            forward="0.0.0.0:8000",
            health_timeout=1,
        ),
        runner=runner,
    )

    result = lifecycle.wait_for_forward_active()

    assert result.ok is False
    assert "timed out waiting for forward" in result.output


def test_wait_for_health_includes_victim_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_health(_url: str, _timeout_seconds: float) -> None:
        raise TimeoutError("health timed out")

    monkeypatch.setattr("agent_hardener.openshell._polling.wait_for_http_health", fail_health)
    runner = SequenceRunner([(0, "diag output from sandbox")])
    lifecycle = OpenShellLifecycle(
        OpenShellConfig(
            gateway="gw",
            sandbox="sb",
            policy_path=tmp_path / "policy.yaml",
            openshell_bin="openshell",
            health_url="http://127.0.0.1:8000/health",
        ),
        runner=runner,
    )

    result = lifecycle.wait_for_health()

    assert result.ok is False
    assert "health timed out" in result.output
    assert "Victim startup diagnostics" in result.output
    assert "diag output from sandbox" in result.output
    assert runner.commands[0][1:3] == ["sandbox", "exec"]
    diagnostics_script = runner.commands[0][runner.commands[0].index("-lc") + 1]
    assert "\n" not in diagnostics_script
    assert "=== app log (launcher-captured) ===" in diagnostics_script


def test_start_victim_retries_supervisor_relay_failures(tmp_path: Path) -> None:
    runner = SequenceRunner(
        [
            (
                1,
                'status: Unavailable, message: "supervisor relay failed: status: Unavailable, message: \\"supervisor session not connected\\""',
            ),
            (1, "sandbox 'sb' is not ready (phase: Provisioning); wait for it to reach Ready state"),
            (0, "started"),
        ]
    )
    lifecycle = OpenShellLifecycle(
        OpenShellConfig(
            gateway="gw",
            sandbox="sb",
            policy_path=tmp_path / "policy.yaml",
            openshell_bin="openshell",
            start_command="/app/start-agents-lab.sh >/tmp/agents-lab.log 2>&1",
            health_timeout=3,
        ),
        runner=runner,
    )

    result = lifecycle.start_victim()

    assert result.ok is True
    assert "victim start command: openshell sandbox exec" in result.output
    assert runner.commands[0][1:4] == ["sandbox", "exec", "--gateway"]
    assert runner.commands[1][1:4] == ["sandbox", "exec", "--gateway"]
    assert runner.commands[2][1:4] == ["sandbox", "exec", "--gateway"]


def test_start_victim_uses_tempfile_capture_for_real_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, Any] = {}

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        kwargs["stdout"].write("started")
        kwargs["stdout"].flush()
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    lifecycle = OpenShellLifecycle(
        OpenShellConfig(
            gateway="gw",
            sandbox="sb",
            policy_path=tmp_path / "policy.yaml",
            openshell_bin="openshell",
            start_command="/app/start-agents-lab.sh >/tmp/agents-lab.log 2>&1",
        ),
    )

    result = lifecycle.start_victim()

    assert result.ok is True
    assert "victim start command: openshell sandbox exec" in result.output
    assert result.output.endswith("started")
    assert recorded["args"][0:4] == ["openshell", "sandbox", "exec", "--gateway"]
    assert recorded["kwargs"]["stderr"] == subprocess.STDOUT
    assert recorded["kwargs"]["text"] is True
    assert recorded["kwargs"]["stdout"] is not subprocess.PIPE


def test_run_logged_waits_for_real_runner_and_captures_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, Any] = {}

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        kwargs["stdout"].write("Uploading files...\nFiles uploaded\n")
        kwargs["stdout"].flush()
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    lifecycle = OpenShellLifecycle(
        OpenShellConfig(
            gateway="gw",
            sandbox="sb",
            policy_path=tmp_path / "policy.yaml",
            openshell_bin="openshell",
            cwd=tmp_path,
        )
    )

    result = lifecycle._exec.run_logged(["sandbox", "create", "--gateway", "gw"])

    assert result.ok is True
    assert result.output.startswith("command completed exit_code=0; duration_seconds=")
    assert "; log=" in result.output
    assert "Files uploaded" in result.output
    assert recorded["args"] == ["openshell", "sandbox", "create", "--gateway", "gw"]
    assert recorded["kwargs"]["cwd"] == tmp_path
    assert recorded["kwargs"]["stderr"] is subprocess.STDOUT
    assert recorded["kwargs"]["text"] is True
    assert recorded["kwargs"]["stdout"].name.startswith(str(tmp_path / ".agent-hardener" / "openshell-logs"))


def test_background_log_uses_injected_run_log_dir(tmp_path: Path) -> None:
    # When the composition root injects log_dir (the run's openshell-logs), command logs land there
    # instead of the shared .agent-hardener fallback.
    run_log_dir = tmp_path / "run-logs" / "run-xyz" / "openshell-logs"
    lifecycle = OpenShellLifecycle(
        OpenShellConfig(
            gateway="gw",
            sandbox="sb",
            policy_path=tmp_path / "policy.yaml",
            openshell_bin="openshell",
            cwd=tmp_path,
        ),
        log_dir=run_log_dir,
    )

    log_path = lifecycle._exec.log_path_for(["sandbox", "exec", "--gateway", "gw"])

    assert str(log_path).startswith(str(run_log_dir))
    # The shared fallback dir is NOT used when a run log dir is injected.
    assert not str(log_path).startswith(str(tmp_path / ".agent-hardener" / "openshell-logs"))


def test_openshell_victim_payload_defaults_to_openai_chat_shape() -> None:
    request = _request().model_copy(update={"context": {"victim_input_message": "hello"}})
    agent = AgentConfig(name="victim", role="victim")

    assert build_payload(request, agent) == {
        "model": "openshell-victim",
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_openshell_tool_uses_configured_lifecycle(tmp_path: Path, monkeypatch: Any) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
storage:
  root_dir: {tmp_path}
victim_control:
  type: openshell
  config:
    gateway: gw
    sandbox: sb
    policy_path: {tmp_path / "policy.yaml"}
target:
  name: target
victim:
  name: victim
""",
        encoding="utf-8",
    )
    calls: list[str] = []

    class FakeLifecycle:
        def __init__(self, config: Any) -> None:
            self.config = config

        def status(self) -> Any:
            calls.append(self.config.gateway)
            return type("Result", (), {"ok": True, "output": "ok"})()

    monkeypatch.setattr("agent_hardener.tools.openshell.OpenShellLifecycle", FakeLifecycle)

    assert openshell_tool_main(["status", "--config", str(config_path)]) == 0
    assert calls == ["gw"]


def test_policy_digest_treats_empty_endpoints_as_omitted() -> None:
    with_empty_endpoints = _policy_digest(
        "version: 1\nnetwork_policies:\n  shell_github:\n    endpoints: []\n    binaries:\n    - path: /usr/bin/curl\n"
    )
    omitted_endpoints = _policy_digest(
        "version: 1\nnetwork_policies:\n  shell_github:\n    binaries:\n    - path: /usr/bin/curl\n"
    )

    assert with_empty_endpoints == omitted_endpoints

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OpenShell lifecycle CLI and its small host-environment helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, cast

import pytest

from agent_hardener.models import VictimControlConfig
from agent_hardener.openshell.lifecycle import OpenShellCommandResult
from agent_hardener.relay_plugin.victim import SANDBOX_ARTIFACTS_DIR
from agent_hardener.tools import openshell as tool

if TYPE_CHECKING:
    from agent_hardener.openshell.lifecycle import OpenShellConfig

pytestmark = pytest.mark.unit


class _Lifecycle:
    """Records which lifecycle operation the CLI dispatched to."""

    instances: ClassVar[list[_Lifecycle]] = []

    def __init__(self, config: Any) -> None:
        self.config = config
        self.calls: list[tuple[str, Any]] = []
        self.result = OpenShellCommandResult(ok=True, command=[], output="lifecycle output")
        _Lifecycle.instances.append(self)

    def _record(self, name: str, arg: Any = None) -> OpenShellCommandResult:
        self.calls.append((name, arg))
        return self.result

    def up(self) -> OpenShellCommandResult:
        return self._record("up")

    def down(self) -> OpenShellCommandResult:
        return self._record("down")

    def restart(self) -> OpenShellCommandResult:
        return self._record("restart")

    def status(self) -> OpenShellCommandResult:
        return self._record("status")

    def ensure_provider(self) -> OpenShellCommandResult:
        return self._record("ensure_provider")

    def apply_policy(self, path: Path) -> OpenShellCommandResult:
        return self._record("apply_policy", path)


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub config loading and the lifecycle; returns the commands for which the build was staged."""
    _Lifecycle.instances = []
    prepared: list[str] = []
    monkeypatch.setattr(tool, "load_config", lambda _path: SimpleNamespace(victim_control="vc"))
    monkeypatch.setattr(tool, "openshell_config", lambda _vc: "base-config")

    def prepare(cfg: str) -> str:
        prepared.append(cfg)
        return f"prepared-{cfg}"

    monkeypatch.setattr(tool, "prepare_relay_victim", prepare)
    monkeypatch.setattr(tool, "OpenShellLifecycle", _Lifecycle)
    return prepared


@pytest.mark.parametrize(
    ("command", "operation", "builds"),
    [
        ("up", "up", True),
        ("restart", "restart", True),
        ("cycle-reset", "restart", True),
        ("down", "down", False),
        ("status", "status", False),
        ("ensure-provider", "ensure_provider", False),
    ],
)
def test_command_dispatches_to_the_matching_lifecycle_operation(
    cli: list[str], capsys: pytest.CaptureFixture[str], command: str, operation: str, builds: bool
) -> None:
    assert tool.main([command, "--config", "cfg.yaml"]) == 0

    (lifecycle,) = _Lifecycle.instances
    assert lifecycle.calls == [(operation, None)]
    assert lifecycle.config == ("prepared-base-config" if builds else "base-config")
    assert bool(cli) is builds
    assert capsys.readouterr().out == "lifecycle output\n"


def test_apply_policy_passes_the_policy_path(cli: list[str]) -> None:
    del cli

    assert tool.main(["apply-policy", "--config", "cfg.yaml", "--policy", "p.yaml"]) == 0

    assert _Lifecycle.instances[0].calls == [("apply_policy", Path("p.yaml"))]


def test_apply_policy_requires_a_policy(cli: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    del cli

    with pytest.raises(SystemExit) as exit_info:
        tool.main(["apply-policy", "--config", "cfg.yaml"])

    assert exit_info.value.code == 2
    assert "--policy is required for apply-policy" in capsys.readouterr().err
    assert _Lifecycle.instances[0].calls == []


def test_a_failing_operation_exits_nonzero_but_still_prints_its_output(
    cli: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    del cli
    monkeypatch.setattr(
        _Lifecycle, "status", lambda _self: OpenShellCommandResult(ok=False, command=[], output="sandbox missing")
    )

    assert tool.main(["status", "--config", "cfg.yaml"]) == 1

    assert capsys.readouterr().out == "sandbox missing\n"


def test_the_config_flag_is_required() -> None:
    with pytest.raises(SystemExit) as exit_info:
        tool.main(["status"])

    assert exit_info.value.code == 2


def test_unknown_commands_are_rejected() -> None:
    with pytest.raises(SystemExit) as exit_info:
        tool.main(["explode", "--config", "cfg.yaml"])

    assert exit_info.value.code == 2


# --- host environment --------------------------------------------------------------------------------------


def test_docker_host_is_left_alone_when_already_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DOCKER_HOST", "tcp://remote:2375")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".colima/default").mkdir(parents=True)
    (tmp_path / ".colima/default/docker.sock").touch()

    tool.configure_local_docker_host()

    assert tool.os.environ["DOCKER_HOST"] == "tcp://remote:2375"


def test_docker_host_falls_back_to_the_colima_socket(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    socket = tmp_path / ".colima/default/docker.sock"
    socket.parent.mkdir(parents=True)
    socket.touch()

    tool.configure_local_docker_host()

    assert tool.os.environ["DOCKER_HOST"] == f"unix://{socket}"


def test_docker_host_stays_unset_without_a_colima_socket(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    tool.configure_local_docker_host()

    assert "DOCKER_HOST" not in tool.os.environ


@pytest.mark.parametrize("path", [None, "", "absent.env"])
def test_loading_a_missing_env_file_changes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str | None
) -> None:
    monkeypatch.chdir(tmp_path)
    before = dict(tool.os.environ)

    tool.load_local_env(path)

    assert dict(tool.os.environ) == before


def test_env_file_values_never_override_the_existing_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("AH_TEST_NEW=from-file\nAH_TEST_EXISTING=from-file\n")
    monkeypatch.setenv("AH_TEST_NEW", "")
    monkeypatch.delenv("AH_TEST_NEW")  # registered with monkeypatch so the value the loader sets is undone
    monkeypatch.setenv("AH_TEST_EXISTING", "from-shell")

    tool.load_local_env(env_file)

    assert tool.os.environ["AH_TEST_NEW"] == "from-file"
    assert tool.os.environ["AH_TEST_EXISTING"] == "from-shell"


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (None, None),
        ("", None),
        ("http://127.0.0.1:8000/v1/chat/completions", "http://127.0.0.1:8000/health"),
        ("https://victim.example/a/b?x=1#frag", "https://victim.example/health"),
    ],
)
def test_health_url_is_the_target_origin_plus_health(base_url: str | None, expected: str | None) -> None:
    assert tool.target_health_url(base_url) == expected


def test_wait_for_health_delegates_to_the_http_poller(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, float]] = []
    monkeypatch.setattr(tool, "wait_for_http_health", lambda url, timeout: calls.append((url, timeout)))

    tool.wait_for_health("http://victim/health", 12.5)

    assert calls == [("http://victim/health", 12.5)]


def test_openshell_config_rejects_other_victim_control_types() -> None:
    config = cast("VictimControlConfig", SimpleNamespace(type="http", config={}))

    with pytest.raises(SystemExit, match=r"victim_control\.type must be 'openshell'"):
        tool.openshell_config(config)


def test_openshell_config_builds_from_the_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(tool.OpenShellConfig, "from_mapping", classmethod(lambda _cls, mapping: seen.append(mapping)))
    config = cast("VictimControlConfig", SimpleNamespace(type="openshell", config={"sandbox": "sb"}))

    tool.openshell_config(config)

    assert seen == [{"sandbox": "sb"}]


# --- relay instrumentation gate ----------------------------------------------------------------------------


class _Http:
    posts: list[tuple[str, dict[str, Any]]]

    def __init__(self, *, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> _Http:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def post(self, url: str, *, json: dict[str, Any]) -> str:
        type(self).posts.append((url, json))
        return "probe-response"


class _Fetcher:
    def __init__(self, _config: Any) -> None:
        self.fetched: list[tuple[str, Path]] = []
        _Fetcher.last = self

    last: _Fetcher

    def fetch_file(self, remote: str, local: Path) -> None:
        self.fetched.append((remote, local))


def _prepared(atof_path: Path | None, cwd: Path | None) -> OpenShellConfig:
    spec = None if atof_path is None else SimpleNamespace(atof_path=atof_path)
    return cast("OpenShellConfig", SimpleNamespace(relay_victim=spec, cwd=cwd))


def test_verification_is_skipped_when_there_is_no_endpoint_to_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tool, "check_relay_instrumentation", lambda **_k: pytest.fail("must not probe"))

    tool.verify_relay_instrumentation(_prepared(Path("a"), None), base_url=None)


def test_verification_is_skipped_for_a_victim_without_a_relay_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tool, "check_relay_instrumentation", lambda **_k: pytest.fail("must not probe"))

    tool.verify_relay_instrumentation(_prepared(None, None), base_url="http://victim/v1")


@pytest.mark.parametrize("absolute", [False, True])
def test_verification_probes_the_victim_and_pulls_the_atof_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, absolute: bool
) -> None:
    atof = tmp_path / "abs" / "events.atof.jsonl" if absolute else Path("rel/events.atof.jsonl")
    _Http.posts = []
    monkeypatch.setattr(tool.httpx, "AsyncClient", _Http)
    monkeypatch.setattr(tool, "OpenShellLifecycle", _Fetcher)
    seen: dict[str, Any] = {}

    async def fake_check(*, probe: Any, atof_path: Path, sync: Any) -> None:
        seen["response"] = await probe()
        seen["atof_path"] = atof_path
        sync()

    monkeypatch.setattr(tool, "check_relay_instrumentation", fake_check)

    tool.verify_relay_instrumentation(_prepared(atof, tmp_path), base_url="http://victim/v1")

    resolved = atof if absolute else tmp_path / atof
    assert seen == {"response": "probe-response", "atof_path": resolved}
    (url, payload) = _Http.posts[0]
    assert url == "http://victim/v1"
    assert payload["messages"] == [{"role": "user", "content": tool.RELAY_PROBE_PROMPT}]
    assert _Fetcher.last.fetched == [(f"{SANDBOX_ARTIFACTS_DIR}/events.atof.jsonl", resolved)]


# --- managed sandbox ---------------------------------------------------------------------------------------


class _Sandbox:
    """Fake lifecycle + collaborators for managed_victim_sandbox; records the order things happen in."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, ready: bool = False, up_ok: bool = True) -> None:
        self.events: list[str] = []
        self.verify_error: Exception | None = None
        outer = self

        class Lifecycle:
            def __init__(self, config: str) -> None:
                self.config = config

            def up(self) -> OpenShellCommandResult:
                outer.events.append(f"up:{self.config}")
                return OpenShellCommandResult(ok=up_ok, command=[], output="" if up_ok else "build failed")

            def down(self) -> None:
                outer.events.append(f"down:{self.config}")

        def verify(prepared: str, *, base_url: str | None) -> None:
            outer.events.append(f"verify:{prepared}:{base_url}")
            if outer.verify_error:
                raise outer.verify_error

        def is_ready(config: str) -> bool:
            outer.events.append(f"ready?:{config}")
            return ready

        monkeypatch.setattr(tool, "openshell_config", lambda _vc: "base")
        monkeypatch.setattr(tool, "prepare_relay_victim", lambda cfg: f"prepared-{cfg}")
        monkeypatch.setattr(tool, "sandbox_is_ready", is_ready)
        monkeypatch.setattr(tool, "verify_relay_instrumentation", verify)
        monkeypatch.setattr(tool, "OpenShellLifecycle", Lifecycle)

    def enter(self, **kwargs: Any) -> Any:
        return tool.managed_victim_sandbox(cast("VictimControlConfig", None), **kwargs)


def test_a_fresh_sandbox_is_built_verified_and_torn_down(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = _Sandbox(monkeypatch)

    with sandbox.enter(reuse=False, no_cleanup=False, base_url="http://victim/v1") as config:
        sandbox.events.append("body")

    assert config == "base"
    assert sandbox.events == ["up:prepared-base", "verify:prepared-base:http://victim/v1", "body", "down:base"]


def test_a_ready_sandbox_is_reused_and_left_running(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = _Sandbox(monkeypatch, ready=True)

    with sandbox.enter(reuse=True, no_cleanup=False):
        pass

    assert sandbox.events == ["ready?:base", "verify:prepared-base:None"]


def test_reuse_falls_back_to_a_build_when_the_sandbox_is_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = _Sandbox(monkeypatch, ready=False)

    with sandbox.enter(reuse=True, no_cleanup=False):
        pass

    assert sandbox.events == ["ready?:base", "up:prepared-base", "verify:prepared-base:None", "down:base"]


def test_readiness_is_not_probed_unless_reuse_was_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = _Sandbox(monkeypatch, ready=True)

    with sandbox.enter(reuse=False, no_cleanup=True):
        pass

    assert "ready?:base" not in sandbox.events


def test_no_cleanup_keeps_the_sandbox_this_call_built(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = _Sandbox(monkeypatch)

    with sandbox.enter(reuse=False, no_cleanup=True):
        pass

    assert not any(event.startswith("down") for event in sandbox.events)


def test_a_failed_build_raises_with_its_output_and_has_nothing_to_tear_down(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = _Sandbox(monkeypatch, up_ok=False)

    with pytest.raises(RuntimeError, match="build failed"), sandbox.enter(reuse=False, no_cleanup=False):
        pytest.fail("the body must not run")

    assert sandbox.events == ["up:prepared-base"]


@pytest.mark.parametrize("failure", ["verification", "body"])
def test_the_sandbox_is_torn_down_even_when_verification_or_the_body_fails(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    sandbox = _Sandbox(monkeypatch)
    if failure == "verification":
        sandbox.verify_error = RuntimeError("victim emits no telemetry")

    with pytest.raises(RuntimeError), sandbox.enter(reuse=False, no_cleanup=False):
        raise RuntimeError("body failed")

    assert sandbox.events[-1] == "down:base"


def test_with_victim_image_sets_image_and_keeps_other_keys() -> None:
    victim_control = VictimControlConfig(type="openshell", config={"sandbox": "sb"})

    assert tool.with_victim_image(victim_control, None) is victim_control
    assert tool.with_victim_image(victim_control, "registry.example.com/agent:1").config == {
        "sandbox": "sb",
        "image": "registry.example.com/agent:1",
    }
    assert victim_control.config == {"sandbox": "sb"}

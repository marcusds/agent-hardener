# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment-agnostic run engine.

``run_mission`` is the shared core the CLI (standalone OSS) and a future NeMo entry-point both drive:
bring up host backends → stand the victim up once (build/reuse sandbox, health, pre-flight synth) →
run the orchestrator once (it owns the hardening rounds) → final log → cleanup. The runner manages no
rounds. It depends only on the session config and reports progress by emitting events on the run's event
bus (``output``/``status_*`` + the orchestrator's phase events), so it is decoupled from Typer — a single
``on_event`` subscriber renders them. The garak agent_breaker attacker spawns the garak CLI itself, so the
runner no longer manages a garak service. OS-specific prep (docker host, dotenv) stays in the caller.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from agent_hardener.agents.validators.smart_benign import synthesize
from agent_hardener.agents.victims.openshell_victim import contract_for
from agent_hardener.atof import load_records
from agent_hardener.bundle import write_hardened_bundle
from agent_hardener.display.final.render_plain import render_final_run_log
from agent_hardener.display.final.render_rich import print_final_summary
from agent_hardener.errors import AgentHardenerError, BenignSuiteError, VictimUnavailableError
from agent_hardener.events import EventType
from agent_hardener.final_log.mitigations import build_mitigations
from agent_hardener.final_log.validation_summary import build_validation
from agent_hardener.ids import generate_round_id
from agent_hardener.loggers import EventBus, capture_run_log, emit_event, using_event_sink
from agent_hardener.models import AgentRunInput, RelayVictimSpec
from agent_hardener.openshell.lifecycle import OpenShellLifecycle
from agent_hardener.openshell.relay_victim import RELAY_PLUGINS_UPLOAD_DEST
from agent_hardener.preflight.relay import (
    RELAY_PROBE_PROMPT,
    RELAY_PROBE_TIMEOUT_SECONDS,
    check_relay_instrumentation,
    check_tool_path,
)
from agent_hardener.providers.backends import BackendManager
from agent_hardener.relay_plugin.victim import SANDBOX_ARTIFACTS_DIR
from agent_hardener.runtime.adapters import build_uploaders
from agent_hardener.runtime.event_sink import make_http_event_sink
from agent_hardener.runtime.orchestrator import Orchestrator
from agent_hardener.swarm_tracker import (
    OPENSHELL_LOGS,
    RunLayout,
    agent_fingerprint_path,
    victim_active_state_dir,
    victim_image_path,
)
from agent_hardener.tools.openshell import (
    openshell_config,
    prepare_relay_victim,
    target_health_url,
    wait_for_health,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from agent_hardener.models import AgentConfig, RoundReport, SessionConfig
    from agent_hardener.openshell.lifecycle import OpenShellConfig


async def _no_interview_answers(_questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Default answer provider: supply nothing (non-interactive runs never interrupt anyway)."""
    return []


class MissionError(AgentHardenerError):
    """Raised when an infrastructure step (sandbox/backend) fails mid-run."""

    category = "sandbox"
    default_remediation = "Check the Docker daemon, the OpenShell gateway, and any host backends, then retry."


@dataclass
class MissionResult:
    """Outcome of a mission run."""

    reports: list[RoundReport]
    success: bool


SMART_BENIGN_IMPL_PREFIX = "agent_hardener.agents.validators.smart_benign"

_events_logger = logging.getLogger("agent_hardener.events")

#: One benign question, chosen to provoke a normal turn without asking for any privileged action.


def _log_event(event: str, payload: dict[str, Any]) -> None:
    """Event-bus subscriber that echoes each event into ``agent-hardener.log`` as a concise human line.

    Only scalar payload fields are rendered, so big serialized blobs (``attack_summary`` etc.) stay
    out of the text transcript — their full form lives in ``events.jsonl``. Routes through the
    ``agent_hardener.events`` logger, which the run-log bus captures.
    """
    if event == EventType.OUTPUT:  # highlight lines are logged verbatim (the runner's announce path)
        _events_logger.info("%s", payload.get("line", ""))
        return
    fields = " ".join(f"{k}={v}" for k, v in payload.items() if isinstance(v, (str, int, float, bool)))
    _events_logger.info("%s %s", event, fields)


def _is_smart_benign(agent: AgentConfig) -> bool:
    """True when a benign validator is backed by the smart benign package."""
    return (agent.implementation or "").startswith(SMART_BENIGN_IMPL_PREFIX)


def _agent_fingerprint(session_config: SessionConfig, openshell_config: OpenShellConfig) -> str:
    """Deterministic identity of the agent a sandbox should be running.

    Combines the victim's workflow config contents with the sandbox name, start
    command, build context, and pre-built image. Recorded when we build a sandbox and re-checked
    on ``--reuse`` so a different agent under the same sandbox name is detected.
    """
    h = hashlib.sha256()
    parts = [openshell_config.sandbox, openshell_config.start_command, str(openshell_config.build_context or "")]
    if openshell_config.image:
        parts.append(openshell_config.image)
    for part in parts:
        h.update((part or "").encode("utf-8"))
        h.update(b"\x00")
    workflow = session_config.target.agent_relay_plugins
    if workflow:
        with contextlib.suppress(OSError):
            h.update(Path(workflow).read_bytes())
    return h.hexdigest()[:16]


def _read_sidecar(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _write_sidecar(path: Path, value: str | None) -> None:
    if value is None:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")


@dataclass
class MissionSpec:
    """The repeatable inputs to a mission, grouped so the runner and its phases share one ``ctx``."""

    session_config: SessionConfig
    rounds: int = 1
    verbose: bool = False
    mission_id: str | None = None
    skip_initial_up: bool = False
    no_cleanup: bool = False
    on_event: Callable[[str, dict[str, Any]], None] | None = None


def run_mission(
    session_config: SessionConfig,
    *,
    rounds: int = 1,
    verbose: bool = False,
    mission_id: str | None = None,
    skip_initial_up: bool = False,
    no_cleanup: bool = False,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> MissionResult:
    """Run one attack/defend/validate mission (thin wrapper over :class:`Mission`)."""
    return Mission(
        MissionSpec(
            session_config,
            rounds=rounds,
            verbose=verbose,
            mission_id=mission_id,
            skip_initial_up=skip_initial_up,
            no_cleanup=no_cleanup,
            on_event=on_event,
        )
    ).run()


class Mission:
    """Runs one attack/defend/validate mission against the configured victim.

    Holds the mission inputs (``ctx``) plus derived run-scoped state (lifecycle, uploaders, log/event bus)
    — formerly locals + closures in ``run_mission`` — so the phases read as a sequence: bring the victim up
    once at the baseline, synth the benign suite, run the orchestrator's hardening rounds, report, tear
    down. Presentation flows through the run's event bus (``ctx.on_event`` is the single subscriber — the
    CLI passes a console renderer, a future UI its own). Use the ``run_mission`` wrapper as the entry point.
    """

    def __init__(self, ctx: MissionSpec) -> None:
        self.ctx = ctx
        self._prepare()

    def run(self) -> MissionResult:
        """Open the log/event sinks, run the phases in order, and always tear down at the end."""
        with contextlib.ExitStack() as sinks:
            self._open_sinks(sinks)
            try:
                self._load_benign_suite()
                self._start_backends()
                self._bring_up_victim()
                reports = self._harden()
                return MissionResult(reports=reports, success=all(r.success for r in reports))
            finally:
                self._teardown()

    def _prepare(self) -> None:
        """Resolve the victim build config, run layout, lifecycle, uploaders, and host backends."""
        cfg = self.ctx.session_config
        self._base_config = prepare_relay_victim(openshell_config(cfg.victim_control))
        self._adopted_image = self._recorded_victim_image() if self.ctx.skip_initial_up else None
        if self._adopted_image:
            self._base_config = replace(self._base_config, image=self._adopted_image)
        # Seed the run from the inferred (egress-injected) policy, not the raw template, so defenders start
        # from the deployed policy (the workflow is already seeded via storage).
        if self._base_config.policy_path is not None:
            cfg.storage.victim_policy_path = self._base_config.policy_path

        # Resolve run_id once; agent-hardener.log + subprocess logs live under run-logs/<run_id>/. The
        # orchestrator reuses this run_id; layout dir creation is idempotent.
        self._run_id = cfg.storage.run_id or generate_round_id()
        run_layout = RunLayout(cfg.storage.root_dir, self._run_id)
        self._log_path = run_layout.run_dir / "agent-hardener.log"

        # Build uploaders on the PREPARED lifecycle so a defender recreate keeps the staged build_context +
        # egress-injected policy (raw config has build_context=None, which would destroy the sandbox), and
        # re-uploads the run's hardened guardrails rather than the seed.
        self._lifecycle = OpenShellLifecycle(
            self._base_config,
            log_dir=run_layout.dir_for(OPENSHELL_LOGS),
            active_state_dir=victim_active_state_dir(run_layout.run_dir),
        )
        self._openshell_uploader, self._relay_uploader = build_uploaders(cfg.victim_control, self._lifecycle)
        self._backend_specs = list(self._base_config.relay_victim.backends) if self._base_config.relay_victim else []
        self._backend_cwd = self._base_config.cwd or Path.cwd()
        self._backend_manager = BackendManager() if self._backend_specs else None

    def _recorded_victim_image(self) -> str | None:
        """The pre-built image the running sandbox was created from, when ``--victim-image`` is not repeated.

        Without it a ``--reuse`` after a ``--victim-image`` run would see a fingerprint mismatch, tear the good
        sandbox down, and fall into a local ``docker build`` on the very host that has no Docker.
        """
        if self._base_config.image is not None:
            return None
        return _read_sidecar(victim_image_path(self._base_config.cwd or Path.cwd(), self._base_config.sandbox))

    def _open_sinks(self, sinks: contextlib.ExitStack) -> None:
        """Wire the per-run agent-hardener.log capture and the multicast structured-event bus onto *sinks*."""
        # Every agent_hardener.* record flows to the file subscriber under one handler lock; console stays clean.
        sinks.enter_context(capture_run_log(self._log_path))
        self._run_log = logging.getLogger("agent_hardener.run")
        # Ambient event sink for the whole run so the orchestrator's loggers and pre-flight synth emit
        # through it. Events persist per round in round_<N>/events.jsonl; the bus writes no whole-run file.
        self._event_bus = EventBus()
        sinks.enter_context(using_event_sink(self._event_bus.emit))
        if self.ctx.on_event is not None:
            self._event_bus.subscribe(self.ctx.on_event)
        self._event_bus.subscribe(_log_event)  # echo each event into agent-hardener.log as a concise human line
        http_sink = make_http_event_sink()  # POST events to the plugin's SSE ingest when configured (else None)
        if http_sink is not None:
            self._event_bus.subscribe(http_sink)

    # --- presentation (formerly nested closures) ---

    def _detail(self, text: str) -> None:
        """Record a run-log-only entry (heavy build/compose blobs)."""
        if text:
            self._run_log.info(text)

    def _announce(self, line: str) -> None:
        """Emit a highlight line as an ``output`` event — rendered by the subscriber and logged."""
        emit_event(EventType.OUTPUT, {"line": line})

    @contextlib.contextmanager
    def _status(self, label: str) -> Iterator[None]:
        """Bracket a blocking step with ``status_started``/``status_completed`` events (a spinner)."""
        emit_event(EventType.STATUS_STARTED, {"label": label})
        try:
            yield
        finally:
            emit_event(EventType.STATUS_COMPLETED, {"label": label})

    # --- phases ---

    def _start_backends(self) -> None:
        """Start any host-side backends the victim's tools call."""
        if not self._backend_manager:
            return
        self._announce("backend setup: starting host backends")
        backend_up = self._backend_manager.up(self._backend_specs, self._backend_cwd)
        if not backend_up.ok:
            self._announce(backend_up.output)  # real error: surface even when quiet
            raise MissionError("host backend startup failed")
        self._detail(backend_up.output)
        self._announce("backends ready")

    def _bring_up_victim(self) -> None:
        """Stand the victim up once at the baseline: build/reuse the sandbox and gate health (synth is separate)."""
        cfg, base_config = self.ctx.session_config, self._base_config
        # --reuse only sets skip_initial_up, which matches by sandbox NAME. Gate reuse on an agent
        # fingerprint (workflow + start command + build context) so a different agent under the same name
        # forces a rebuild instead of probing the wrong victim.
        fp = _agent_fingerprint(cfg, base_config)
        builds_cwd = base_config.cwd or Path.cwd()
        fp_path = agent_fingerprint_path(builds_cwd, base_config.sandbox)
        if self._adopted_image:
            self._announce(f"reusing the pre-built victim image recorded for this sandbox: {self._adopted_image}")
        reuse_ok = self.ctx.skip_initial_up and _read_sidecar(fp_path) == fp
        if self.ctx.skip_initial_up and not reuse_ok:
            self._announce(
                "running sandbox does not match the configured agent (fingerprint mismatch or "
                "unknown) — rebuilding instead of reusing"
            )
        if not reuse_ok:
            self._lifecycle.down()
            with self._status("Building and starting sandbox"):
                result = self._lifecycle.up(None)
            if not result.ok:
                self._announce(result.output)  # real error: surface even when quiet
                raise MissionError("OpenShell sandbox failed to start")
            self._detail(result.output)
            self._announce("sandbox ready")
            self._announce(f"  → full build/infra output: {self._log_path}")
            _write_sidecar(fp_path, fp)
            _write_sidecar(victim_image_path(builds_cwd, base_config.sandbox), base_config.image)
        else:
            self._announce("reusing sandbox (agent fingerprint matches)")
        # Ensure the victim is serving before synth probes it (wait_for_health is idempotent).
        health_url = target_health_url(cfg.target.base_url)
        if health_url:
            try:
                with self._status("Waiting for victim health"):
                    wait_for_health(health_url, base_config.health_timeout)
            except TimeoutError as exc:
                raise VictimUnavailableError(
                    f"victim did not become healthy at {health_url} within {base_config.health_timeout}s"
                ) from exc
            self._announce("victim health ready")
        self._check_relay_instrumentation()

    def _check_relay_instrumentation(self) -> None:
        """Prove the victim emits Relay telemetry before a single attack is sent.

        Load-bearing rather than a nicety: an uninstrumented victim answers every attack and emits
        nothing, so the run looks like a working demo of a guardrail that was never installed. One
        probe up front is cheap; discovering it from a clean-looking report is not.

        Skipped for a plain HTTP target with no relay-victim block — there is no sandbox to read
        telemetry out of.
        """
        spec = self._base_config.relay_victim
        if spec is None:
            return
        atof_path = self._host_atof_path(spec)
        contract = contract_for(self.ctx.session_config.target.base_url, self.ctx.session_config.victim)

        async def probe() -> Any:
            async with httpx.AsyncClient(timeout=RELAY_PROBE_TIMEOUT_SECONDS) as client:
                return await client.post(
                    contract.url, json=contract.request_payload(RELAY_PROBE_PROMPT), headers=contract.headers()
                )

        with self._status("Checking Relay instrumentation"):
            result = asyncio.run(
                check_relay_instrumentation(
                    probe=probe,
                    atof_path=atof_path,
                    sync=lambda: self._lifecycle.fetch_file(f"{SANDBOX_ARTIFACTS_DIR}/{atof_path.name}", atof_path),
                )
            )
        self._announce(f"relay instrumentation confirmed ({result.records_observed} event(s))")
        if result.quarantined:
            # Not fatal: the run still produces findings, but the trajectory is untrustworthy and a
            # report that does not say so reads as more certain than it is.
            self._announce("  ⚠ Relay reported a dirty scope stack — the recorded trajectory may be wrong")

    def _host_atof_path(self, spec: RelayVictimSpec) -> Path:
        """Where the victim's ATOF log lands on *this* machine."""
        cwd = self._base_config.cwd or Path.cwd()
        return cwd / spec.atof_path if not spec.atof_path.is_absolute() else spec.atof_path

    def _verify_tool_path(self) -> None:
        """Prove the victim's tool calls pass through Relay, from the first round's attack traffic.

        The startup probe proves Relay is *attached*; it cannot prove tool calls *reach* it, because a
        greeting provokes no tool. Garak does, so this runs once the attacks have — no probe tool to
        declare, no synthetic call, and no side effect Agent Hardener caused itself.

        Skipped for a plain HTTP target with no relay-victim block, matching the startup probe: there
        is no sandbox to read telemetry out of.
        """
        spec = self._base_config.relay_victim
        if spec is None:
            return
        atof_path = self._host_atof_path(spec)
        self._sync_atof()
        records = load_records(atof_path) if atof_path.exists() else []
        tools = check_tool_path(records)
        self._announce(f"tool calls reach Relay ({', '.join(tools)})")

    def _sync_atof(self) -> None:
        """Copy the victim's ATOF stream out of the sandbox, best-effort.

        Separate from :meth:`_verify_tool_path` because collecting the telemetry and asserting
        something about it are different jobs: the assertion only makes sense when live attackers
        produced the traffic, while the collection is worth doing for every round.
        """
        spec = self._base_config.relay_victim
        if spec is None:
            return
        atof_path = self._host_atof_path(spec)
        with contextlib.suppress(Exception):
            self._lifecycle.fetch_file(f"{SANDBOX_ARTIFACTS_DIR}/{atof_path.name}", atof_path)

    def _load_benign_suite(self) -> None:
        """Pre-flight: seed the supplied benign suite for each smart-benign validator.

        ``run`` is a pure consumer — it never synthesizes. Every configured smart-benign validator
        requires a suite generated out of band (``agent-hardener synth-benign`` or the serve HITL) and
        supplied via ``--benign-suite``. Missing, empty, or unreadable suites fail before infrastructure
        startup so validation can never fall back to stale cached requests.
        """
        cfg = self.ctx.session_config
        validators = [v for v in cfg.benign_validators if _is_smart_benign(v)]
        if not validators:
            return
        if cfg.benign_suite_path is None:
            raise BenignSuiteError("smart-benign validation requires an explicit benign suite, but none was supplied")
        context = {"storage_root": str(cfg.storage.root_dir)}
        for agent in validators:
            request = AgentRunInput(
                round_id="preflight-load", target=cfg.target, context=context, validator_kind="benign"
            )
            try:
                with self._status(f"loading benign suite ({agent.name})"):
                    result = asyncio.run(
                        synthesize(
                            request,
                            agent,
                            answer_provider=_no_interview_answers,
                            source_suite=cfg.benign_suite_path,
                        )
                    )
            except Exception as exc:
                raise BenignSuiteError(
                    f"failed to load benign suite {cfg.benign_suite_path} for {agent.name}: {exc}"
                ) from exc
            if not result.viable:
                raise BenignSuiteError(
                    f"benign suite {cfg.benign_suite_path} contains no replayable requests for {agent.name}"
                )
            self._announce(f"benign suite ready: {result.request_count} request(s) → {result.artifact_dir}")

    def _harden(self) -> list[RoundReport]:
        """Run the orchestrator's hardening rounds and persist the final report."""
        # The orchestrator owns the loop: one run_rounds runs the configured rounds, re-running the
        # attackers and carrying each round's deployed policy forward. Events flow through the ambient sink.
        orchestrator = Orchestrator(
            self.ctx.session_config,
            openshell_uploader=self._openshell_uploader,
            relay_uploader=self._relay_uploader,
            verify_tool_path=self._verify_tool_path,
            sync_atof=self._sync_atof,
        )
        reports = asyncio.run(
            orchestrator.run_rounds(rounds=self.ctx.rounds, mission_id=self.ctx.mission_id, run_id=self._run_id)
        )
        # Always persist the full report to the run log (--verbose only controls inline detail). The
        # console summary is rendered here (not via an event) as a deliberate, bounded exception: it is a
        # one-shot terminal render that needs the full RoundReport objects, which don't fit the
        # serializable-event model the rest of the run uses.
        mission_ids = [self.ctx.mission_id] if self.ctx.mission_id else None
        self._detail("\n" + render_final_run_log(reports, mission_ids=mission_ids))
        self._announce(f"  → full report detail: {self._log_path}")
        print_final_summary(reports, verbose=self.ctx.verbose)
        cfg = self.ctx.session_config
        # mitigations.json: the run's before/after policy+workflow — the NeMo plugin saves it for the Studio
        # Mitigations view. Its policy/workflow names come from the configured victim-control paths.
        self._write_run_artifact(
            "mitigations.json",
            lambda run_dir: build_mitigations(
                run_dir,
                reports,
                policy_name=cfg.storage.victim_policy_path.name if cfg.storage.victim_policy_path else None,
                guardrails_name=cfg.storage.victim_relay_plugins_path.name
                if cfg.storage.victim_relay_plugins_path
                else None,
            ),
        )
        # validation.json: the final per-item attack/benign results — drives the Studio sanity-check
        # scorecard. Written for every run that ran validators, including frozen validate-only runs.
        self._write_run_artifact("validation.json", lambda _run_dir: build_validation(reports))
        self._write_hardened_bundle()
        return reports

    def _write_hardened_bundle(self) -> None:
        """Assemble the run's deployable artifact: the image the run validated plus its guardrails.

        Best-effort. The bundle is the *useful* output, but a packaging failure must not discard a run
        whose report is already written — the operator is told, and the evidence survives either way.
        """
        spec = self._base_config.relay_victim
        build_context, image = self._base_config.build_context, self._base_config.image
        if spec is None or (build_context is None and image is None):
            return
        run_dir = RunLayout(self.ctx.session_config.storage.root_dir, self._run_id).run_dir
        active_state = victim_active_state_dir(run_dir)
        plugins_toml = active_state / RELAY_PLUGINS_UPLOAD_DEST.rsplit("/", 1)[-1]
        if not plugins_toml.is_file():
            return
        # The run hardens two things; ship both. The policy is the half docker cannot apply.
        configured_policy = self.ctx.session_config.storage.victim_policy_path
        policy = active_state / configured_policy.name if configured_policy else None
        try:
            bundle = write_hardened_bundle(
                # build_context is the staged *Dockerfile*; the bundle is its whole directory.
                build_root=build_context.parent if build_context else None,
                plugins_toml=plugins_toml,
                destination=run_dir / "hardened-image",
                agent=self.ctx.session_config.target.name,
                run_id=self._run_id,
                policy=policy,
                image=image,
            )
        except Exception:
            self._run_log.warning("could not assemble the hardened image bundle", exc_info=True)
            return
        self._announce(f"  → hardened image bundle: {bundle}")

    def _write_run_artifact(self, filename: str, build: Callable[[Path], Any]) -> None:
        """Persist a best-effort JSON run artifact next to the run log; skip a falsy payload, never raise."""
        try:
            run_dir = RunLayout(self.ctx.session_config.storage.root_dir, self._run_id).run_dir
            payload = build(run_dir)
            if payload:
                (run_dir / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            self._run_log.debug("failed to write %s", filename, exc_info=True)

    def _save_victim_logs(self) -> None:
        """Keep the victim's in-sandbox logs before the sandbox is deleted. Best-effort.

        A victim that starts healthy and then fails its requests leaves its traceback only there.
        """
        try:
            run_dir = RunLayout(self.ctx.session_config.storage.root_dir, self._run_id).run_dir
            output = self._lifecycle.collect_victim_diagnostics().output
            if output:
                (run_dir / "victim-diagnostics.log").write_text(output, encoding="utf-8")
        except Exception:
            self._run_log.debug("failed to save victim logs", exc_info=True)

    def _teardown(self) -> None:
        """Tear the sandbox + backends down, unless --no-cleanup left them running."""
        if self.ctx.no_cleanup:
            self._announce("Cleanup skipped; sandbox + backends left running.")
            return
        self._save_victim_logs()
        self._detail(self._lifecycle.down().output)
        if self._backend_manager:
            self._detail(self._backend_manager.down(self._backend_specs, self._backend_cwd).output)

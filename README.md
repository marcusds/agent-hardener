# Agent Hardener

Agent Hardener orchestrates a security hardening loop for agentic applications. It loads attack evidence or runs attackers, routes findings to defenders, applies policy and workflow patches to a victim environment, replays the original attacks, runs benign validation, and writes a full session report.

The current main demo protects a NAT research agent running inside an OpenShell sandbox. It uses:

- Garak-style attack evidence or live Garak attackers.
- An OpenShell policy defender that creates candidate network policy updates.
- A guardrails defender that updates the NAT workflow YAML with pre-tool verifier middleware.
- OpenShell victim-control that applies policy updates, uploads workflow updates, and restarts the victim.
- Attack and benign validators that prove the defended agent blocks the attacks without breaking normal requests.

See [AGENT_CARD.md](AGENT_CARD.md) for the agent transparency card: intended use, limitations, evaluation results and safety considerations.

## Contributions

This project is currently not accepting contributions.

## Operating safely

Agent Hardener generates working attacks and runs a deliberately hostile workload against an agent
you supply. That makes a few things your responsibility.

**Only test agents you own or are authorised to test.** The tool aims real attacks at whatever
endpoint you point it at. Pointing it at someone else's agent is unauthorised testing.

**Point the agent under test at non-production backends.** During a run the victim is under active
attack and may be induced to call its tools in ways you did not intend. The sandbox restricts egress
to the backends you declare in the manifest — so declare test doubles, not the real thing.

**A victim manifest is trusted input.** The manifest names a compose file and image that Agent
Hardener will build and run on your machine. Running a manifest you did not write is equivalent to
running that person's code.

**Run artifacts are sensitive.** Everything under `.agent-hardener/` — hitlogs, prompts, model
responses — includes the record of which attacks *succeeded* against your agent. That is a working
exploit list for it. The shipped `.gitignore` keeps these directories out of version control; keep
it that way, and treat the directory like a credential when copying it around.

**Credentials come from the environment.** Keys are read from your environment or `--env-file` at
the point of use and are never written into run artifacts. Keep `.env` out of version control (it
is gitignored by default).

**`agent-hardener serve` is unauthenticated.** It binds `127.0.0.1` for that reason. If you override
`--host`, anyone who can reach that interface can drive it — do not expose it on an untrusted
network.

## Install

```bash
pip install nvidia-agent-hardener
```

The distribution is `nvidia-agent-hardener`; the import package and CLI are `agent_hardener` and
`agent-hardener`.

After installing, run `agent-hardener setup` once. garak is deliberately not a dependency (see
[Garak agent_breaker](#garak-agent_breaker)), so a freshly installed copy has no garak until setup
provisions its dedicated virtual environment.

To work on Agent Hardener itself rather than consume it, skip this section and follow
[Environment Setup](#environment-setup) below.

## Quickstart

### Run Agent Hardener on your own NAT agent

The easy path. Describe your NAT agent in a thin `agent-hardener.yaml` manifest, then run. Agent Hardener
builds your agent into an OpenShell sandbox, starts any host backends, runs the hardening loop
(the garak attacker spawns the `garak` CLI directly), and cleans up. Only the NAT agent runs inside
the sandbox; backends/DBs stay on the host.

```bash
just setup                        # one-time: Docker/OpenShell checks, credentials, deps
uv run agent-hardener init            # scaffold agent-hardener.yaml (auto-detects your NAT project),
                                  #   garak-scan.yaml, and .env.example
                                  #   prompts for port (default 8000); pass --port <N> to skip the prompt
# ...fill in .env with your keys...
uv run agent-hardener synth-benign    # generate the benign test suite (brings the sandbox up, interviews
                                  #   you, writes requests.csv, tears down); --reuse/--no-cleanup available
uv run agent-hardener run --benign-suite <path/to/requests.csv>   # build, harden, replay + validate, report
                           #   (omit --benign-suite to skip the benign false-positive checks)
```

`run` is a pure consumer: it validates against the suite you pass and never generates one — use
`synth-benign` (or the `serve` HITL) to produce it. Without `--benign-suite`, `run` still attacks,
defends, and validates the attacks; it just skips the benign false-positive check.

A manifest is ~10 lines (see `examples/agent-hardener.yaml`):

```yaml
agent:
  name: finance
  project_dir: ../agents-lab
  workflow: agents_lab/agents/finance/workflow.yaml
  port: 8000
  secrets: [INFERENCE_API_KEY]
  env:
    FINANCE_BACKEND_URL: http://host.docker.internal:8086
backends:
  - name: finance
    compose_file: ../agents-lab/services/finance_backend/docker-compose.yaml
    health_url: http://127.0.0.1:8086/health
    ports: [8086]
  # - name: ledger                      # a service that is not on this host: only the route is opened
  #   host: ledger.ns.svc
  #   ports: [8087]
  #   allowed_ips: [10.96.0.0/12]       # private ranges OpenShell would otherwise block
  #   health_url: http://ledger.ns.svc:8087/health
```

The minimum a user must supply: `agent.project_dir` + `agent.workflow`, and `agent.secrets` (env-var
names pulled from `.env`). Add `dockerfile` + `binaries` to build the victim from your own image
instead of a generic one — that is independent of `workflow`, which you keep either way: the image is
how the environment is built, the workflow is what gets served and hardened. Everything else — gateway,
sandbox, policy, the attacker/defender/validator suite — is defaulted (override via an `overrides:`
block or a full session config). Useful commands: `uv run agent-hardener run --rounds N`, `--no-cleanup`,
`uv run agent-hardener synth-benign` (generate/refresh the benign suite), `uv run agent-hardener up|down|status`.

On a host without Docker, `uv run agent-hardener stage` writes the victim's build context and prints its
directory. Build and push that directory with any image builder, then pass the result to
`run --victim-image <ref>` (also accepted by `up` and `synth-benign`). The image must come from the `stage`
output of the same manifest and agent-hardener version you run with: `run` re-stages the context to derive
the sandbox policy and launch environment, and nothing checks that the image still matches it.

Garak's agent_breaker probe is run from a `garak-scan.yaml` config that `init` scaffolds and
`run` regenerates with the resolved victim endpoint. To tune it (models, victim port, attempts), add a
`garak:` block to the manifest or edit the scaffold — see [Garak agent_breaker](#garak-agent_breaker).


## Repository Layout

```text
agent_hardener/
  orchestrator.py                    Main session loop
  adapters.py                        In-process, HTTP, and victim-control adapters
  agents/
    attackers/agent_breaker/         Garak agent_breaker attacker + config builder
    defenders/                       OpenShell policy and guardrails defenders
    validators/                      Attack replay and benign validators
    victims/                         NAT/OpenShell victim helpers and workflow
  openshell/                         OpenShell lifecycle and policy repair code
  templates/nat-victim/              Packaged assets: NAT-victim Dockerfile + default/permissive
                                     OpenShell policies + repair profile
  tools/                             CLI helpers and run summaries

examples/
  vulnerable_demo_e2e.yaml           Main runnable OpenShell/Garak demo config

tests/
  agent_hardener/                        Unit, integration, pipeline E2E, smoke tests
  fixtures/pipeline_e2e/             Minimal deterministic E2E fixture
```

## Prerequisites

For local development and deterministic tests:

- Python 3.11
- `uv`
- `just`
- `git`
- Bash and standard coreutils

For the real OpenShell/Garak demo:

- OpenShell compute driver access: Docker, Podman, Kubernetes, or the local macOS VM driver
- `curl`
- OpenShell CLI
- The `garak` CLI (installed into a dedicated venv at `~/.agent-hardener/garak-venv` by `agent-hardener setup`, run for you by `just install`; see [Garak agent_breaker](#garak-agent_breaker))
- NVIDIA-compatible inference credentials
- A GitHub token and test repository for the research-agent workflow
- Free port `8000` (the victim agent)
- Optional free observability ports `4318` and `6006`
- Optional `jq` for artifact inspection examples

## Environment Setup

Install OpenShell and verify the local gateway:

```bash
curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | OPENSHELL_VERSION=v0.1.4 sh
openshell status
```

> **Upgrading from OpenShell 0.0.x**
>
> Agent Hardener targets OpenShell 0.1.4. OpenShell 0.1.0 is a breaking release, and the installer refuses to upgrade a 0.0.x install in place. On macOS:
>
> 1. Delete every 0.0.x sandbox: `openshell sandbox delete --all --gateway <name>` for each registered gateway. 0.0.x sandboxes cannot run under 0.1.
> 2. Uninstall: `brew services stop nvidia/openshell/openshell && brew uninstall nvidia/openshell/openshell`. Also remove any stray `openshell` / `openshell-gateway` binaries left in `/opt/homebrew/bin` or `~/.local/bin`; the installer stops if it finds a 0.0.x binary on the path.
> 3. Remove the old state: `$(brew --prefix)/var/openshell` and `~/.local/state/openshell`. A 0.0.x gateway database there crashes the 0.1 gateway at startup.
> 4. Install with the command above, then run `uv run agent-hardener setup` to configure the Docker driver and register the `auto-defender` gateway.
>
> On Linux, use `systemctl --user disable --now openshell-gateway` and your package manager in place of the Homebrew commands, and remove `~/.local/state/openshell`.

Do not use `uv tool install -U openshell` for the real E2E path. That installs the CLI only; it does not install or start the local OpenShell gateway service. The native installer installs the CLI plus the gateway service. On macOS and current Linux packages, the local service listens on `https://127.0.0.1:17670` and the installer registers it as the `openshell` gateway.

If you previously installed the CLI with `uv tool install`, it may shadow the native package binary on `PATH`. Remove it with `uv tool uninstall openshell`, or verify that `openshell --version` and the native package version match before running the E2E.

On macOS, configure the gateway to use the Docker driver. `uv run agent-hardener setup` does this for you. To do it by hand, write `~/.config/openshell/gateway.toml` with Docker's socket path (from `docker context inspect --format '{{.Endpoints.docker.Host}}'`, without the `unix://` prefix), then run `brew services restart nvidia/openshell/openshell`:

```toml
[openshell]
version = 2

[openshell.gateway]
compute_driver = "docker"

[openshell.drivers.docker]
socket_path = "/Users/<you>/.docker/run/docker.sock"
grpc_endpoint = "https://host.docker.internal:17670"
```

`grpc_endpoint` is required under Docker Desktop. The driver runs each sandbox's supervisor with host networking, and there that "host" is Docker Desktop's VM, so the gateway's `127.0.0.1` listener is unreachable. Without it, every sandbox fails with `ControlSupervisorStartFailed`. The `OPENSHELL_DRIVERS` / `DOCKER_HOST` launchctl variables used with 0.0.x are deprecated.

The Docker driver runs sandboxes as containers inside Docker Desktop's VM. Avoid the VM driver on macOS — the VM driver uses Homebrew's `e2fsprogs` to create an ext4 rootfs, and the version installed by Homebrew creates a filesystem with features the OpenShell base VM kernel cannot mount read-only.

The final status should report `Status: Connected`. If you plan to use the example config unchanged, also register the same endpoint as `auto-defender`:

```bash
openshell gateway add https://127.0.0.1:17670 --local --name auto-defender
openshell status --gateway auto-defender
```

Install dependencies:

```bash
just install
```

`just install` runs `uv sync` (installs agent-hardener's dependencies), provisions the dedicated garak venv (via `agent-hardener setup`), then installs pre-commit hooks. Run it before the main wrapper so `.venv/bin/python3` exists. The wrapper should be run from the repository root. Advanced users can override the interpreter with `PROJECT_PYTHON=/path/to/python`.

garak is **not** an agent-hardener dependency: it pins `litellm -> httpx>=0.28` plus `torch`, which conflict with `nvidia-nat`'s `httpx~=0.27`. agent-hardener never imports garak — it spawns the garak CLI from a separate venv at `~/.agent-hardener/garak-venv`, provisioned by `agent-hardener setup` (`just garak-venv` is a thin wrapper around it). Override the interpreter with `AGENT_HARDENER_GARAK_PYTHON=/path/to/python`. This is what keeps agent-hardener itself pip-installable with no dependency override.

To work with a custom garak fork instead of the PyPI release, overlay it as an editable install into that venv:

```bash
just garak-dev /path/to/garak                 # editable overlay into ~/.agent-hardener/garak-venv
rm -rf ~/.agent-hardener/garak-venv && just garak-venv   # revert to the PyPI version
```

Create a local `.env` file. Do not commit it.

```bash
cat > .env <<EOF
INFERENCE_API_KEY=...
NIM_API_KEY=...
VLLM_API_KEY=dummy
GITHUB_TOKEN=...
GITHUB_REPOSITORY=owner/repo
EOF
```

Notes:

- `INFERENCE_API_KEY` is used by the victim workflow and policy/guardrail model calls.
- `NIM_API_KEY` is used by Garak detector and judge paths when configured.
- `GITHUB_TOKEN` should be scoped to a test repository. The research workflow can read issues and write issue comments.
- `VLLM_API_KEY` can be `dummy` when not used by your runtime.

### Choosing models

Agent Hardener has three model groups, each with a built-in default you can override:

| Group | Covers | Default | Override |
| --- | --- | --- | --- |
| Attack | garak red-team + detector | `nvidia/nemotron-3-super-120b-a12b` | the `garak:` block or `GARAK_*` env (see [Garak agent_breaker](#garak-agent_breaker)) |
| Analysis | defenders + benign validator (both synth suite-generation and judging) | `nvidia/nemotron-3-super-120b-a12b` | `AGENT_HARDENER_MODEL` / `AGENT_HARDENER_BASE_URL` (the shared lever), or the validator's per-component `config` |
| Safety | the guardrail judge that runs inside the victim | the analysis model | the guardrails defender's `safety_llm` (model); `AGENT_HARDENER_SAFETY_BASE_URL` / `AGENT_HARDENER_SAFETY_API_KEY` (endpoint and key, else the analysis ones) |
| Agent | the victim's own LLM | your NAT workflow | the `llms:` block in your workflow |

The **analysis** lever is a single pair of env vars that retargets every default-model caller (both
defenders and the benign validator) at once:

```bash
export AGENT_HARDENER_MODEL=nvidia/nemotron-3-super-120b-a12b
export AGENT_HARDENER_BASE_URL=https://integrate.api.nvidia.com/v1
```

Before a run, `agent-hardener run` preflights any model you overrode (`AGENT_HARDENER_MODEL` or
`GARAK_RED_TEAM_MODEL_NAME`) against its endpoint and fails fast — listing the models the credentials
can reach — if the name, URL, or key is wrong, so a typo costs seconds instead of a full sandbox build.

Load the file into your shell when running NAT directly:

```bash
set -a
source .env
set +a
```

If running NAT outside the OpenShell sandbox, set tracing defaults when Phoenix/OTel are not running:

```bash
export OTEL_TRACE_ENDPOINT=${OTEL_TRACE_ENDPOINT:-http://127.0.0.1:4318}
export PHOENIX_TRACE_ENDPOINT=${PHOENIX_TRACE_ENDPOINT:-http://127.0.0.1:6006}
```

## OpenShell Gateway Config

The OpenShell gateway for the run is configured in YAML at `victim_control.config.gateway`. The example config uses `auto-defender`:

```yaml
victim_control:
  type: openshell
  config:
    gateway: auto-defender
```

To use a different gateway, copy or edit the config and change that field. The wrapper prepares loop-specific configs from the YAML, so keep the YAML as the source of truth. The `--gateway` wrapper flag is only for final diagnostic `openshell sandbox list` and `openshell forward list` output.

If you use the default local gateway created by the OpenShell native installer, either change the YAML gateway to `openshell` or register the same local endpoint under the example name:

```bash
openshell gateway add https://127.0.0.1:17670 --local --name auto-defender
openshell status --gateway auto-defender
```

The second command must report `Status: Connected` before running the E2E.

## Garak agent_breaker

Agent Hardener runs garak's **agent_breaker** probe by invoking the `garak` CLI directly — there is no
Scan API/REST server. garak runs from its own venv (`~/.agent-hardener/garak-venv`, provisioned by
`agent-hardener setup`), not agent-hardener's, so its heavy/conflicting deps stay out of agent-hardener's
closure. agent-hardener resolves the interpreter from `$AGENT_HARDENER_GARAK_PYTHON` (else the default
venv), or you can set `$GARAK_COMMAND` for a full custom invocation. Confirm garak is present with:

```bash
~/.agent-hardener/garak-venv/bin/python -m garak --list-probes | grep agent_breaker
```

How a run works:

1. `agent-hardener init` scaffolds an editable `garak-scan.yaml` (a standard garak `--config` file).
2. On each `agent-hardener run`, the attacker loads that scaffold, overlays the resolved victim endpoint
   (and the report destination), and writes `.agent-hardener/run-logs/garak-agent-breaker.resolved.yaml`.
3. It spawns `garak --config <resolved>.yaml`, then reads the report/hitlog garak writes under
   `report_dir` (default `.agent-hardener/garak_runs`). Logs go to `.agent-hardener/run-logs/garak-agent-breaker.log`.

The probe auto-discovers the victim's tools (no `agent.yaml` needed). Edits you make to
`garak-scan.yaml` (e.g. model settings) survive; only the target `uri` and reporting keys are
overwritten each run.

### Override fields

Add a `garak:` block to the manifest (or set the matching field in a full session config's attacker
`config`). All fields are optional; model-related fields also read a `GARAK_<FIELD>` env var.

| Field | Default | Env override | Purpose |
|-------|---------|--------------|---------|
| `config_path` | `garak-scan.yaml` | — | Scaffold config the attacker overlays onto |
| `report_dir` | `.agent-hardener/garak_runs` | — | Where garak writes report/hitlog (the final log scans here) |
| `report_prefix` | `agent-breaker` | — | Filename prefix for this run's artifacts |
| `target_uri` | victim base_url | — | Full victim chat-completions URL (wins over `target_port`) |
| `target_port` | victim port | — | Override just the port on the victim URL |
| `red_team_model_type` | `nim.NVOpenAIChat` | `GARAK_RED_TEAM_MODEL_TYPE` | Red-team (attack) model generator type |
| `red_team_model_name` | `nvidia/nemotron-3-super-120b-a12b` | `GARAK_RED_TEAM_MODEL_NAME` | Red-team model name |
| `red_team_model_uri` | `https://integrate.api.nvidia.com/v1/` | `GARAK_RED_TEAM_MODEL_URI` | Red-team model endpoint |
| `detector_model_type` | `nim` | `GARAK_DETECTOR_MODEL_TYPE` | Detector (judge) model generator type |
| `detector_model_name` | `nvidia/nemotron-3-super-120b-a12b` | `GARAK_DETECTOR_MODEL_NAME` | Detector model name |
| `detector_model_uri` | `https://integrate.api.nvidia.com/v1/` | `GARAK_DETECTOR_MODEL_URI` | Detector model endpoint |
| `max_attempts_per_tool` | `5` | `GARAK_MAX_ATTEMPTS_PER_TOOL` | Exploit attempts per discovered tool |
| `generations` | `1` | `GARAK_GENERATIONS` | garak `run.generations` |

`GARAK_COMMAND` overrides the base garak invocation (default `garak`; e.g. `python -m garak`).

Example manifest block (attack a victim on a different port with a custom judge model):

```yaml
garak:
  target_port: 9001
  detector_model_name: nvidia/nemotron-3-super-120b-a12b
  max_attempts_per_tool: 8
```

The detector model settings the replay validator uses are seeded from these same defaults, so the
default inference endpoint and model live in one place.

## Tests

Fast deterministic tests:

```bash
just test-fast
```

Full local test suite:

```bash
just test
```

Run the logical pipeline E2E only:

```bash
just test-pipeline-e2e
```

That test runs the real orchestrator with real defenders, mocked external model/HTTP boundaries, fake OpenShell command execution, attack replay, benign validation, and artifact assertions.

Optional real OpenShell smoke E2E:

```bash
AGENT_HARDENER_RUN_OPEN_SHELL_SMOKE=1 \
uv run pytest tests/agent_hardener/test_openshell_smoke_e2e.py -q -o addopts=""
```

Use this only when Docker/OpenShell are available and the gateway is healthy. It creates a small real OpenShell sandbox and verifies policy update, workflow update, validators, and report output.

Lint and format:

```bash
uv run ruff check .
uv run ruff format --check .
```

## E2E Readiness

Before running the real OpenShell/Garak cycle, confirm:

- Docker daemon is reachable: `docker info`
- OpenShell CLI and gateway are available: `openshell status --gateway auto-defender`
- The gateway name in `victim_control.config.gateway` is registered and connected
- `.env` exists with real `INFERENCE_API_KEY`, `NIM_API_KEY`, `VLLM_API_KEY`, `GITHUB_TOKEN`, and `GITHUB_REPOSITORY` values
- The `garak` CLI is installed (`uv run garak --list-probes | grep agent_breaker`)
- Port `8000` is free (the victim agent)
- Optional observability ports `4318` and `6006` are free when Phoenix/OTel are enabled

## Run The Main OpenShell/Garak Cycle

The main operator command is:

```bash
uv run agent-hardener run \
  --config examples/vulnerable_demo_e2e.yaml \
  --env-file .env \
  --no-cleanup
```

What this does:

1. Loads `.env`.
2. Prepares a loop-specific config under `.agent-hardener/run-logs/.../loop-01/`.
3. Renders the per-loop garak config; the agent_breaker attacker spawns the `garak` CLI when it runs.
4. Runs the OpenShell policy defender and guardrails defender in parallel.
5. Applies the candidate policy with `openshell policy set`.
6. Uploads the candidate workflow into the sandbox.
7. Restarts the victim agent and waits for health.
8. Replays attack hits and, when a benign suite is supplied via `--benign-suite`, the benign requests.
9. Writes artifacts and a final summary.

Useful run controls:

```text
--rounds N         Carry generated policy/workflow forward across N hardening rounds.
--replay           Skip live attackers; replay hits from the latest run instead.
--reuse            Skip sandbox rebuild if it is already running; a --victim-image sandbox is reused as-is.
--victim-image     Create the sandbox from a pre-built image (see `agent-hardener stage`) instead of docker build.
--no-cleanup       Leave sandbox running after the run.
--verbose          Show raw output and full report detail.
```

Results are written under:

```text
.agent-hardener/run-logs/<timestamp>-<mission-id>/
```

The most important session artifacts are:

```text
attacks.json          Loaded or generated attack records
defenders.json        Defender decisions and patch metadata
victim-control.json   OpenShell apply/upload/restart results
validators.json       Attack and benign validation results
report.json           Complete machine-readable session report
session.log           Human-readable session log
events.jsonl          Structured event log
```

## Check That A Run Really Updated The Sandbox

After a successful run, inspect `victim-control.json`:

```bash
jq ".metadata.results[] | {patch_type, ok, status, command}" \
  .agent-hardener/run-logs/<run>/loop-01/missions/<mission>/session-0001/victim-control.json
```

Healthy policy update signals:

```text
openshell_policy_candidate       ok=true
openshell_policy_verification    ok=true, status=active_policy_matches_candidate
```

Healthy workflow update signals:

```text
victim_workflow_candidate        ok=true
openshell_workflow_upload        ok=true
openshell_victim_restart         ok=true
```

Inspect the live sandbox directly:

```bash
openshell sandbox exec \
  --gateway auto-defender \
  --name agents-lab-vulnerable-demo \
  --no-tty -- sh -lc "
    ls -l /tmp/research_agent_workflow.yaml /tmp/research_agent_workflow.rendered.yaml
    grep -n custom_guardrail /tmp/research_agent_workflow.yaml | head
  "
```

Check the active OpenShell policy:

```bash
openshell sandbox get agents-lab-vulnerable-demo \
  --gateway auto-defender \
  --policy-only
```

Check victim health:

```bash
curl -i http://127.0.0.1:8000/health
```

## Run NAT Directly

Run the research agent once:

```bash
uv run nat run \
  --config_file agent_hardener/agents/victims/research_agent_workflow.yaml \
  --input "what is the time now"
```

Serve the workflow locally:

```bash
uv run nat serve \
  --config_file agent_hardener/agents/victims/research_agent_workflow.yaml \
  --host 127.0.0.1 \
  --port 8000 \
  --disable_legacy_routes true
```

Then check:

```bash
curl -i http://127.0.0.1:8000/health
```

Inside the OpenShell sandbox, the victim runs roughly:

```bash
/app/agents-lab/.venv/bin/nat serve \
  --config_file /tmp/research_agent_workflow.rendered.yaml \
  --host 0.0.0.0 \
  --port 8000 \
  --disable_legacy_routes true
```

## OpenShell Helpers

The OpenShell tool wrapper can manage the configured sandbox directly:

```bash
uv run python -m agent_hardener.tools.openshell ensure-provider --config examples/vulnerable_demo_e2e.yaml --env-file .env
uv run python -m agent_hardener.tools.openshell up              --config examples/vulnerable_demo_e2e.yaml --env-file .env
uv run python -m agent_hardener.tools.openshell status          --config examples/vulnerable_demo_e2e.yaml --env-file .env
uv run python -m agent_hardener.tools.openshell restart         --config examples/vulnerable_demo_e2e.yaml --env-file .env
uv run python -m agent_hardener.tools.openshell down            --config examples/vulnerable_demo_e2e.yaml --env-file .env
```

For a fresh service host:

```bash
./scripts/bootstrap-openshell-victim.sh --smoke-chat
```

or:

```bash
just openshell-bootstrap --smoke-chat
```

## Preloaded Versus Live Attacks

`examples/vulnerable_demo_e2e.yaml` currently enables the live Garak attacker. This is the right path when you need fresh attack generation (the attacker runs the `garak` CLI).

For faster defender iteration with saved evidence, edit the config:

1. Uncomment or add the `preloaded_attacks` entries.
2. Set `attackers: []` so live attackers do not rerun.
3. Run with `--use-config-attackers`.

Preloaded attacks are faster and more deterministic. Live attacks are slower and run the `garak` CLI (which calls the configured red-team/detector models), but they exercise the real attacker path.

## Cleanup

Project cleanup:

```bash
just clean
```

Docker/OpenShell cleanup should be careful. Start with:

```bash
docker builder prune
docker image prune
```

For old unused images only:

```bash
docker image prune -a --filter "until=24h"
```

Avoid `docker system prune -a --volumes` while debugging unless you intentionally want broad cleanup.

## Current Stable Check

Before merging a refactor, run:

```bash
just test-fast
just test-pipeline-e2e
uv run pytest -q -o addopts=""
```

For OpenShell changes, also run the main cycle once and confirm:

```text
report.success = true
victim-control ok = true
openshell_policy_verification status = active_policy_matches_candidate
attack validation = blocked all original hits
benign validation = all requests complied
```

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inject host-backend egress allowlists into an OpenShell policy.

The sandboxed NAT agent is hardened by an OpenShell network policy that default-denies egress.
When the agent's tools call backend services running on the host (outside OpenShell), the policy
must allowlist those ``host.docker.internal:<port>`` endpoints. This module adds that block
idempotently so it survives the permissive transform and policy-repair cycles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml  # type: ignore[import-untyped]

from agent_hardener.openshell.egress.host_gateway import HOST_GATEWAY_ALIAS, host_gateway_allowed_ips

if TYPE_CHECKING:
    from pathlib import Path

    from agent_hardener.models import BackendEndpoint
    from agent_hardener.openshell.egress.base import PolicyEndpoint

RELAY_VICTIM_BACKENDS_KEY = "relay_victim_backends"
RELAY_VICTIM_DISCOVERED_EGRESS_KEY = "relay_victim_discovered_egress"


def normalize_host_gateway_allowed_ips(policy: dict) -> dict:
    """Point every ``host.docker.internal`` endpoint's ``allowed_ips`` at this daemon's bridge IP.

    Templates carry a driver-specific default (``172.17.0.1``, correct for Docker Desktop / Linux).
    This rewrites it — across the telemetry, backend, and discovered-egress blocks alike — to
    whatever ``host.docker.internal`` actually resolves to on the running daemon (e.g. Colima's
    ``192.168.5.2``), so OpenShell's anti-SSRF guard permits the host calls it should. Mutates and
    returns ``policy``. A no-op when the resolved IP already matches.
    """
    allowed_ips: list[str] | None = None  # resolved lazily — probes Docker only if a match exists
    for key, block in (policy.get("network_policies") or {}).items():
        if not isinstance(block, dict):
            continue
        for endpoint in block.get("endpoints", []) or []:
            if isinstance(endpoint, dict) and endpoint.get("host") == HOST_GATEWAY_ALIAS:
                if allowed_ips is None:
                    allowed_ips = host_gateway_allowed_ips()
                # A declared backend's allowed_ips are the user's; templates carry a default to replace.
                declared = list(endpoint.get("allowed_ips") or []) if key == RELAY_VICTIM_BACKENDS_KEY else []
                endpoint["allowed_ips"] = declared + [ip for ip in allowed_ips if ip not in declared]
    return policy


def _endpoint_dict(
    host: str, port: int, *, protocol: str, enforcement: str, access: str, allowed_ips: list[str] | None
) -> dict[str, object]:
    entry: dict[str, object] = {"host": host, "port": port, "protocol": protocol}
    # OpenShell rejects request-level fields on an L4 (``tcp``) endpoint.
    if protocol != "tcp":
        entry |= {"enforcement": enforcement, "access": access}
    if allowed_ips:
        entry["allowed_ips"] = list(allowed_ips)
    return entry


def _egress_block(name: str, entries: list[dict[str, object]], binaries: list[str]) -> dict[str, object]:
    return {"name": name, "endpoints": entries, "binaries": [{"path": path} for path in binaries]}


def _inject_egress_block(policy_path: Path, dest_path: Path, key: str, block: dict[str, object] | None) -> Path:
    """Load ``policy_path``, set (or leave absent) one ``network_policies`` block, write to ``dest_path``.

    Idempotent: the block under ``key`` is always replaced, never duplicated; a ``None`` block copies the
    policy through unchanged. Returns ``dest_path``.
    """
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
    if block is not None:
        policy.setdefault("network_policies", {})[key] = block
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(yaml.safe_dump(policy, sort_keys=False, default_flow_style=False), encoding="utf-8")
    return dest_path


def build_backend_egress_block(endpoints: list[BackendEndpoint], binaries: list[str]) -> dict[str, object]:
    """Build the ``relay_victim_backends`` network-policy block."""
    entries = [
        _endpoint_dict(
            e.host, e.port, protocol=e.protocol, enforcement=e.enforcement, access=e.access, allowed_ips=e.allowed_ips
        )
        for e in endpoints
    ]
    return _egress_block("relay-victim-backends", entries, binaries)


def build_discovered_egress_block(endpoints: list[PolicyEndpoint], binaries: list[str]) -> dict[str, object]:
    """Build the ``relay_victim_discovered_egress`` network-policy block from discovered endpoints."""
    entries = [
        _endpoint_dict(e.host, e.port, protocol="rest", enforcement="enforce", access="full", allowed_ips=e.allowed_ips)
        for e in endpoints
    ]
    return _egress_block("relay-victim-discovered-egress", entries, binaries)


#: The interpreters a venv's ``bin/python`` actually resolves to. OpenShell matches a block's
#: ``binaries`` against the *resolved* executable, and ``/workspace/.venv/bin/python`` is a symlink to
#: ``/usr/local/bin/python3.12`` — so a block naming only the venv glob matches no process and
#: silently grants nothing. Verified in a sandbox: with only ``/workspace/.venv/bin/**`` the proxy
#: answers ``403 Forbidden`` for a host the block allows; adding these makes the same host reachable.
SYSTEM_INTERPRETER_BINARIES = ("/usr/local/bin/python*", "/usr/bin/python*")


def apply_victim_binaries(policy: dict[str, Any], binaries: list[str]) -> None:
    """Point every network-policy block at the victim's own interpreter.

    The template ships a fixed binary list (``/app/project/.venv/bin/**`` plus the system pythons)
    inherited from the NAT layout. A Fabric-packaged victim runs ``/workspace/.venv/bin/python``,
    which matches none of them, so its outbound calls are refused by blocks the template owns — the
    inference endpoint above all, which fails the agent's own startup rather than an attack.

    Applied to every block rather than a specific one: they all describe the same victim process, and
    a block that names the wrong binary grants nothing while looking like it grants something.

    The system interpreters are *added* rather than merely preserved. A block Agent Hardener builds
    itself — discovered egress, backend egress — starts with no ``/usr/`` entry to keep, so relying on
    what the block already had left exactly those blocks unmatched: every auto-discovered host was
    admitted into the policy and then refused at the proxy, which is the failure discovery exists to
    prevent.
    """
    for block in (policy.get("network_policies") or {}).values():
        if isinstance(block, dict):
            existing = [entry for entry in block.get("binaries") or [] if isinstance(entry, dict)]
            kept = [str(entry.get("path", "")) for entry in existing if str(entry.get("path", "")).startswith("/usr/")]
            # Fresh dicts, deduplicated: a victim glob that equals a system interpreter glob would
            # otherwise be listed twice, and a shared dict object dumps as a YAML anchor/alias.
            paths = dict.fromkeys([*binaries, *kept, *SYSTEM_INTERPRETER_BINARIES])
            block["binaries"] = [{"path": path} for path in paths]


#: Endpoint protocols OpenShell inspects at L7 — the only ones a ``credential_binding`` can apply to.
_REQUEST_PROTOCOLS = frozenset({"rest", "websocket", "graphql", "mcp", "json-rpc"})


def bind_provider_credentials(policy: dict[str, Any], provider: str) -> dict[str, Any]:
    """Let the victim's secrets resolve on every inspected endpoint the policy already allows.

    OpenShell hands the sandbox placeholders, not secret values, and only substitutes the real value
    on endpoints whose ``credential_binding`` names the provider. Binding every inspected endpoint lets
    the secrets reach any egress the policy allows, leaving the policy — which the defenders narrow — the single place that grants it. ``tls: skip`` and
    L4 (``tcp``) endpoints are left unbound: their traffic is never inspected, so nothing is rewritten.
    Mutates and returns ``policy``.
    """
    for block in (policy.get("network_policies") or {}).values():
        if not isinstance(block, dict):
            continue
        for endpoint in block.get("endpoints", []) or []:
            if (
                isinstance(endpoint, dict)
                and endpoint.get("protocol") in _REQUEST_PROTOCOLS
                and endpoint.get("tls") != "skip"
            ):
                endpoint["credential_binding"] = {"provider": provider}
    return policy


def reconcile_shared_endpoint_settings(policy: dict[str, Any]) -> dict[str, Any]:
    """Give every endpoint on the same exact ``host:port`` the union of their ``allowed_ips``.

    OpenShell rejects a policy whose endpoints on one host:port disagree on ``allowed_ips``. The
    template, backend, and discovered-egress blocks are written independently, so the same host
    (``*.nvidia.com:443`` from both the template and discovery) routinely lands with different lists.
    The union only matters for private addresses the host resolves to, which every list already
    vouched for. Mutates and returns ``policy``.
    """
    endpoints_by_key: dict[tuple[str, object], list[dict]] = {}
    for block in (policy.get("network_policies") or {}).values():
        if not isinstance(block, dict):
            continue
        for endpoint in block.get("endpoints", []) or []:
            if isinstance(endpoint, dict) and endpoint.get("host"):
                for port in endpoint.get("ports") or [endpoint.get("port")]:
                    endpoints_by_key.setdefault((str(endpoint["host"]).lower(), port), []).append(endpoint)
    for endpoints in endpoints_by_key.values():
        union = list(dict.fromkeys(ip for endpoint in endpoints for ip in endpoint.get("allowed_ips") or []))
        if union:
            for endpoint in endpoints:
                endpoint["allowed_ips"] = list(union)
    return policy


def inject_backend_egress(
    policy_path: Path, endpoints: list[BackendEndpoint], binaries: list[str], dest_path: Path
) -> Path:
    """Add a backend-egress block to ``policy_path`` and write the result to ``dest_path``.

    Idempotent and safe to re-run after the permissive transform or a policy-repair pass; with no
    endpoints the policy is copied through unchanged. Returns ``dest_path``.
    """
    block = build_backend_egress_block(endpoints, binaries) if endpoints else None
    return _inject_egress_block(policy_path, dest_path, RELAY_VICTIM_BACKENDS_KEY, block)


def inject_discovered_egress(
    policy_path: Path, endpoints: list[PolicyEndpoint], binaries: list[str], dest_path: Path
) -> Path:
    """Add the auto-discovered egress block to ``policy_path`` and write it to ``dest_path``.

    Idempotent; with no endpoints the policy is copied through unchanged. ``endpoints`` come from egress
    discovery (host-pattern + port pairs). Returns ``dest_path``.
    """
    block = build_discovered_egress_block(endpoints, binaries) if endpoints else None
    return _inject_egress_block(policy_path, dest_path, RELAY_VICTIM_DISCOVERED_EGRESS_KEY, block)

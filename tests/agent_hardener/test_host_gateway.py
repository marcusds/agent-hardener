# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for host.docker.internal bridge-IP resolution and policy normalization."""

from __future__ import annotations

import subprocess

import pytest

from agent_hardener.openshell.egress import host_gateway
from agent_hardener.openshell.egress.host_gateway import (
    FALLBACK_GATEWAY_IP,
    host_gateway_allowed_ips,
    resolve_host_gateway_ip,
    resolve_host_gateway_ips,
)
from agent_hardener.openshell.policy_egress import normalize_host_gateway_allowed_ips


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """resolve_host_gateway_ips is lru_cached; reset between tests so stubs take effect."""
    resolve_host_gateway_ips.cache_clear()
    yield
    resolve_host_gateway_ips.cache_clear()


DUAL_STACK_AHOSTS = (
    "fdc4:f303:9324::254 STREAM host.docker.internal\n"
    "fdc4:f303:9324::254 DGRAM  host.docker.internal\n"
    "172.29.0.254        STREAM host.docker.internal\n"
    "172.29.0.254        DGRAM  host.docker.internal\n"
)


def test_parse_getent_ips_extracts_alias_ip() -> None:
    # `getent ahosts host.docker.internal` output: "<ip>  <socktype>  <name>", repeated per socktype.
    assert host_gateway._parse_getent_ips("192.168.5.2 STREAM host.docker.internal\n") == ("192.168.5.2",)


def test_parse_getent_ips_empty_when_nothing_resolves() -> None:
    # getent prints nothing (and exits non-zero) when the alias doesn't resolve; the fallback is
    # applied by resolve_host_gateway_ips, not here.
    assert host_gateway._parse_getent_ips("") == ()


def test_parse_getent_ips_keeps_both_families_ipv4_first() -> None:
    # Docker Desktop publishes the alias dual-stack; the victim's client may connect over either, so
    # both have to reach allowed_ips. Duplicates across socket types collapse.
    assert host_gateway._parse_getent_ips(DUAL_STACK_AHOSTS) == ("172.29.0.254", "fdc4:f303:9324::254")


def test_parse_getent_ips_skips_non_address_lines() -> None:
    output = "some warning from the runtime\n172.29.0.254 STREAM host.docker.internal\n"
    assert host_gateway._parse_getent_ips(output) == ("172.29.0.254",)


def test_allowed_ips_covers_both_families(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=DUAL_STACK_AHOSTS, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    # /32 for IPv4 and /128 for IPv6 — a /32 on an IPv6 address is what broke the guard originally.
    assert host_gateway_allowed_ips() == ["172.29.0.254/32", "fdc4:f303:9324::254/128"]
    assert resolve_host_gateway_ip() == "172.29.0.254"


def test_resolve_single_ip_falls_back_when_only_ipv6(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args, **_kwargs):
        stdout = "fdc4:f303:9324::254 STREAM host.docker.internal\n"
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert resolve_host_gateway_ip() == FALLBACK_GATEWAY_IP
    # ...but the IPv6 address is still allowlisted, so the guard permits the connection.
    assert host_gateway_allowed_ips() == ["fdc4:f303:9324::254/128"]


def test_resolve_reads_alias_from_container(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="10.0.0.1 host.docker.internal\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert resolve_host_gateway_ip() == "10.0.0.1"
    assert host_gateway_allowed_ips() == ["10.0.0.1/32"]


def test_resolve_falls_back_when_docker_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args, **_kwargs):
        raise FileNotFoundError("docker not installed")

    monkeypatch.setattr(subprocess, "run", boom)
    assert resolve_host_gateway_ip() == FALLBACK_GATEWAY_IP


def test_normalize_rewrites_all_host_gateway_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host_gateway, "resolve_host_gateway_ips", lambda *_a, **_k: ("192.168.5.2",))
    policy = {
        "network_policies": {
            "relay_victim_host_telemetry": {
                "endpoints": [
                    {"host": "host.docker.internal", "port": 4318, "allowed_ips": ["172.17.0.1/32"]},
                    {"host": "host.docker.internal", "port": 6006, "allowed_ips": ["172.17.0.1/32"]},
                ]
            },
            "relay_victim_egress": {
                "endpoints": [{"host": "*.nvidia.com", "port": 443, "allowed_ips": ["192.0.2.0/24"]}]
            },
        }
    }
    normalize_host_gateway_allowed_ips(policy)
    telemetry = policy["network_policies"]["relay_victim_host_telemetry"]["endpoints"]
    assert all(ep["allowed_ips"] == ["192.168.5.2/32"] for ep in telemetry)
    # Non-gateway endpoints are untouched.
    assert policy["network_policies"]["relay_victim_egress"]["endpoints"][0]["allowed_ips"] == ["192.0.2.0/24"]


def test_normalize_keeps_a_declared_backends_allowed_ips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host_gateway, "resolve_host_gateway_ips", lambda *_a, **_k: ("192.168.5.2",))
    policy = {
        "network_policies": {
            "relay_victim_backends": {
                "endpoints": [{"host": "host.docker.internal", "port": 8086, "allowed_ips": ["10.0.0.5/32"]}]
            }
        }
    }
    normalize_host_gateway_allowed_ips(policy)
    [endpoint] = policy["network_policies"]["relay_victim_backends"]["endpoints"]
    assert endpoint["allowed_ips"] == ["10.0.0.5/32", "192.168.5.2/32"]

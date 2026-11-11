"""Per-incident anonymised role tokens for evidence cards (DESIGN §3.4.3 "Role tokens").

    H<n>[role]   internal host   (role from the services the host SERVES, e.g. camera,rtsp,telnet)
    X<n>[ext]    external host
    INFRA<n>     collapsed shared infrastructure (hosts listed in any unit's ``infra`` column)

Numbering is by first appearance in unit order (unit_id = time order with tiehash), scanning
each unit's src, then dsts (list order), then infra. Roles are label-free port profiles: a host
serves a service if it is the responder of a non-fan-out unit (≤ 4 dsts) with replies
(rbytes > 0) and fail_frac < 0.9. Deterministic; raw addresses never leave this module except
through ``RoleMap.lookup`` (used to redact IP literals inside packet text).
"""
from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

# service priority (most identifying first) and labels
SERVICE_ORDER = ("rtsp", "mqtt", "coap", "modbus", "dns", "http", "https", "ssh", "telnet", "ftp", "ntp",
                 "irc", "report48101")
DEVICE_CLASS = {"rtsp": "camera", "mqtt": "mqtt-broker", "modbus": "plc"}
SERVICE_LABEL = {"http": "web", "https": "web", "report48101": "48101"}
PORT_SERVICE = {554: "rtsp", 8554: "rtsp", 1883: "mqtt", 8883: "mqtt", 5683: "coap", 5684: "coap", 502: "modbus",
                53: "dns", 80: "http", 8080: "http", 8000: "http", 8081: "http", 443: "https", 8443: "https",
                22: "ssh", 23: "telnet", 2323: "telnet", 21: "ftp", 123: "ntp", 6667: "irc", 48101: "report48101"}
MAX_ROLE_ITEMS = 3


def _get(u: Any, k: str, default=None):
    v = u.get(k, default) if isinstance(u, Mapping) else getattr(u, k, default)
    return default if v is None else v


def unit_service(u: Any) -> str | None:
    """Service of a unit: svc_class if it is a known service, else from the first dport."""
    svc = str(_get(u, "svc_class", "") or "").lower()
    if svc in SERVICE_ORDER:
        return svc
    ports = list(_get(u, "dports", []) or [])
    return PORT_SERVICE.get(int(ports[0])) if ports else None


def _side_internal(dir_class: str, side: int) -> bool | None:
    parts = str(dir_class or "").split("->")
    if len(parts) != 2:
        return None
    return {"int": True, "ext": False}.get(parts[side].strip())


def _rfc1918(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_private or ip.is_link_local or ip.is_loopback
    except ValueError:
        return False


@dataclass
class RoleMap:
    tokens: dict[str, str] = field(default_factory=dict)    # host -> "H3[camera,rtsp]" / "X1[ext]" / "INFRA2"
    short: dict[str, str] = field(default_factory=dict)     # host -> "H3" / "X1" / "INFRA2"
    kind: dict[str, str] = field(default_factory=dict)      # host -> internal | external | infra
    served: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def token(self, host: str) -> str:
        return self.tokens.get(host, "?")

    def name(self, host: str) -> str:
        return self.short.get(host, "?")

    def lookup(self, text_ip: str) -> str | None:
        return self.short.get(text_ip)


def role_label(served: Iterable[str], kind: str = "internal", max_items: int = MAX_ROLE_ITEMS) -> str:
    if kind == "external":
        return "ext"
    svcs = [s for s in SERVICE_ORDER if s in set(served)]
    items: list[str] = []
    for s in svcs:
        if s in DEVICE_CLASS:
            items.append(DEVICE_CLASS[s])
            break
    for s in svcs:
        items.append(SERVICE_LABEL.get(s, s))
    items = list(dict.fromkeys(items))[:max_items]
    return ",".join(items) if items else "client"


def build_roles(units: Iterable[Any], is_internal: Callable[[str], bool] | None = None,
                max_items: int = MAX_ROLE_ITEMS) -> RoleMap:
    """Role tokens for one incident. ``units``: UNIT_SCHEMA rows (dicts) or a polars DataFrame."""
    if hasattr(units, "iter_rows"):
        units = list(units.iter_rows(named=True))
    rows = sorted(units, key=lambda u: (int(_get(u, "unit_id", 0)), float(_get(u, "ts", 0.0))))
    infra: set[str] = set()
    for u in rows:
        infra.update(str(h) for h in (_get(u, "infra", []) or []))
    kind: dict[str, str] = {}
    order: list[str] = []
    served: dict[str, set[str]] = {}

    def see(host: str, internal: bool | None):
        if not host:
            return
        if host not in kind:
            if host in infra:
                k = "infra"
            else:
                if internal is None:
                    internal = is_internal(host) if is_internal else _rfc1918(host)
                k = "internal" if internal else "external"
            kind[host] = k
            order.append(host)

    for u in rows:
        dc = str(_get(u, "dir_class", ""))
        src = str(_get(u, "src", "") or "")
        dsts = [str(d) for d in (_get(u, "dsts", []) or [])]
        si, di = _side_internal(dc, 0), _side_internal(dc, 1)
        if is_internal is not None:
            si = di = None
        see(src, si)
        for d in dsts:
            see(d, di)
        for h in (_get(u, "infra", []) or []):
            see(str(h), None)
        svc = unit_service(u)
        if (svc and len(dsts) <= 4 and not bool(_get(u, "fanout", False))
                and float(_get(u, "rbytes", 0) or 0) > 0 and float(_get(u, "fail_frac", 1.0) or 0.0) < 0.9):
            for d in dsts:
                served.setdefault(d, set()).add(svc)

    rm = RoleMap()
    count = {"internal": 0, "external": 0, "infra": 0}
    prefix = {"internal": "H", "external": "X", "infra": "INFRA"}
    for h in order:
        k = kind[h]
        count[k] += 1
        name = f"{prefix[k]}{count[k]}"
        rm.short[h], rm.kind[h] = name, k
        sv = tuple(s for s in SERVICE_ORDER if s in served.get(h, set()))
        rm.served[h] = sv
        if k == "infra":
            rm.tokens[h] = name
        else:
            rm.tokens[h] = f"{name}[{role_label(sv, k, max_items)}]"
    return rm


def tokens_for(rm: RoleMap, hosts: Sequence[str]) -> list[str]:
    return [rm.token(h) for h in hosts]

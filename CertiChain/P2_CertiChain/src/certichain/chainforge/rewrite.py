"""Role-consistent IP rewriting and time shifting for ChainForge incidents (DESIGN §5.1.2, §5.4).

Per incident one :class:`IPPlan` (drawn from the incident RNG) defines a *base map* β:

* testbed hosts 192.168.1.k → 10.0.N.σ(k) (one random /24 per incident; σ a random
  permutation of the host octets 2..254, gateway .1 and broadcast .255/.0 kept), so background
  devices are consistent within the window and secondary victims of spliced episodes land on
  the same devices as in the background (TON episodes and normal_Bro backgrounds share the
  testbed addressing);
* other private (RFC 1918) hosts → fresh addresses in a second random /24 10.0.M.0/24;
* other external hosts (background clouds, attack third parties) → fresh addresses in the
  RFC 2544 range 198.18.0.0/15 (consistent per incident);
* multicast, 255.255.255.255, 0.0.0.0 unchanged.

Campaign *roles* (attacker, C2, loader/download/CNC/report servers) get fresh TEST-NET
addresses (198.51.100.0/24, 203.0.113.0/24); internal roles (victim X, pivot Y) are
background devices. A spliced episode is rewritten with μ_e = β ∘ (swaps): each bound slot
(e.g. TON attacker → C2, primary victim → X) is moved onto its role address and the host
that previously owned that address in β takes the slot's old address, so μ_e stays
injective (distinct hosts stay distinct) and the episode's internal structure is intact.
Only IPs and times change; bytes, packets, states, services and content fields never do.

Observability: a record none of whose endpoints is local (RFC 1918, link-local, multicast,
broadcast) after rewriting would not cross the IoT gateway and is dropped
(``visible_mask``); applied identically to background and spliced records.

Time: an episode is shifted by an integer offset; CF-TON rounds ts and te to 1 s
(``round_times``), ts_res = 1.0.
"""
from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field

import numpy as np
import polars as pl

TESTBED_PREFIX = "192.168.1."
_RFC1918 = tuple(ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))
_LINK_LOCAL = ipaddress.ip_network("169.254.0.0/16")
_ZERO8 = ipaddress.ip_network("0.0.0.0/8")
ROLE_NETS = ("198.51.100.", "203.0.113.")            # TEST-NET-2 / TEST-NET-3
EXT_NET = ipaddress.ip_network("198.18.0.0/15")      # RFC 2544 benchmarking range


def _ip(ip: str):
    try:
        return ipaddress.ip_address(ip)
    except ValueError:
        return None


def is_special(ip: str) -> bool:
    """Multicast, limited broadcast, unspecified/0.0.0.0/8: never remapped."""
    a = _ip(ip)
    if a is None:
        return True
    if a.version != 4:
        return True
    return a.is_multicast or str(a) == "255.255.255.255" or a in _ZERO8


def is_private(ip: str) -> bool:
    a = _ip(ip)
    return a is not None and a.version == 4 and any(a in n for n in _RFC1918)


def is_local(ip: str) -> bool:
    """Seen as local by the IoT gateway: RFC 1918, link-local, multicast, broadcast."""
    a = _ip(ip)
    if a is None or a.version != 4:
        return False
    return (any(a in n for n in _RFC1918) or a in _LINK_LOCAL or a.is_multicast
            or str(a) == "255.255.255.255")


def is_role_ip(ip: str) -> bool:
    return any(ip.startswith(p) for p in ROLE_NETS)


@dataclass
class IPPlan:
    """Per-incident synthetic IP plan (see module docstring). All draws use ``rng``.

    mode "ton"     (CF-TON): one testbed /24 (192.168.1.0/24) → 10.0.N.0/24 with a host-octet permutation,
                   other private hosts → a second /24, roles → TEST-NET.
    mode "subnets" (CF-GOTHAM): every private /24 of the source (Gotham devices 192.168.16-20.x, infra
                   192.168.33/34/35.x, scanner 192.168.0.x) → its own random 10.0.N.0/24, hosts assigned
                   lazily to random free octets (.0/.1/.255 kept); role IPs are fresh hosts inside the
                   mapped subnet of the original role (``fresh_in_subnet``), because Gotham's
                   infrastructure is internal in the real captures (keeps the bank comparable to
                   Gotham-real).
    """
    rng: np.random.Generator
    mode: str = "ton"
    net_main: int = field(init=False)
    net_other: int = field(init=False)
    base: dict[str, str] = field(init=False, default_factory=dict)
    _used: set[str] = field(init=False, default_factory=set)
    _perm: dict[int, int] = field(init=False, default_factory=dict)
    _free_other: list[int] = field(init=False, default_factory=list)
    _roles: list[str] = field(init=False, default_factory=list)

    def __post_init__(self):
        if self.mode == "subnets":
            self._snet: dict[str, int] = {}
            self._sfree: dict[str, list[int]] = {}
            self._nets_free = [int(v) for v in self.rng.permutation(256)]
            self.net_main = self.net_other = -1
            return
        nets = self.rng.choice(256, size=2, replace=False)
        self.net_main, self.net_other = int(nets[0]), int(nets[1])
        octs = np.arange(2, 255)
        perm = self.rng.permutation(octs)
        self._perm = {int(k): int(v) for k, v in zip(octs, perm)}
        self._perm.update({0: 0, 1: 1, 255: 255})
        self._free_other = [int(v) for v in self.rng.permutation(np.arange(1, 255))]
        roles = [f"{p}{h}" for p in ROLE_NETS for h in range(1, 255)]
        self._roles = [roles[i] for i in self.rng.permutation(len(roles))]

    # ------------------------------------------------------------------ prefixes
    @property
    def internal_prefixes(self) -> list[str]:
        if self.mode == "subnets":
            return ["10.0.0.0/16"]
        return [f"10.0.{self.net_main}.0/24", f"10.0.{self.net_other}.0/24"]

    # ------------------------------------------------------------------ subnet mode
    def _subnet(self, key: str) -> int:
        if key not in self._snet:
            if not self._nets_free:
                raise RuntimeError("IPPlan: 10.0.0.0/16 exhausted")
            self._snet[key] = self._nets_free.pop(0)
            self._sfree[key] = [int(v) for v in self.rng.permutation(np.arange(2, 255))]
        return self._snet[key]

    def _subnet_free_host(self, key: str) -> str:
        n = self._subnet(key)
        free = self._sfree[key]
        while free:
            h = free.pop(0)
            cand = f"10.0.{n}.{h}"
            if cand not in self._used:
                return cand
        raise RuntimeError(f"IPPlan: subnet {key} exhausted")

    def _subnet_host(self, ip: str) -> str:
        a, b, c, d = ip.split(".")
        key = f"{a}.{b}.{c}"
        o = int(d)
        if o in (0, 1, 255):
            return f"10.0.{self._subnet(key)}.{o}"
        return self._subnet_free_host(key)

    def fresh_in_subnet(self, like_ip: str) -> str:
        """A fresh host address in the incident subnet of ``like_ip``'s /24 (role IPs, subnet mode)."""
        a, b, c, _ = like_ip.split(".")
        out = self._subnet_free_host(f"{a}.{b}.{c}")
        self._used.add(out)
        return out

    # ------------------------------------------------------------------ base map β
    def map_base(self, ip: str) -> str:
        if ip is None:
            return ip
        hit = self.base.get(ip)
        if hit is not None:
            return hit
        if is_special(ip):
            out = ip
        elif self.mode == "subnets":
            out = self._subnet_host(ip) if is_private(ip) else self._fresh_external()
        elif ip.startswith(TESTBED_PREFIX):
            out = f"10.0.{self.net_main}.{self._perm[int(ip.rsplit('.', 1)[1])]}"
        elif is_private(ip):
            if not self._free_other:
                raise RuntimeError("IPPlan: second /24 exhausted")
            out = f"10.0.{self.net_other}.{self._free_other.pop(0)}"
        else:
            out = self._fresh_external()
        self.base[ip] = out
        self._used.add(out)
        return out

    def _fresh_external(self) -> str:
        n = EXT_NET.num_addresses
        for _ in range(1000):
            off = int(self.rng.integers(1, n - 1))
            a = str(EXT_NET.network_address + off)
            if a.endswith(".0") or a.endswith(".255") or a in self._used:
                continue
            return a
        raise RuntimeError("IPPlan: could not draw an external address")

    def new_role_ip(self) -> str:
        if self.mode == "subnets":
            raise RuntimeError("IPPlan(subnets): use fresh_in_subnet for role IPs")
        while self._roles:
            a = self._roles.pop(0)
            if a not in self._used:
                self._used.add(a)
                return a
        raise RuntimeError("IPPlan: TEST-NET role addresses exhausted")

    def device_ip(self, ton_ip: str) -> str:
        """Incident address of a background device (its base image)."""
        return self.map_base(ton_ip)

    # ------------------------------------------------------------------ per-episode μ_e
    def episode_map(self, ips, bindings: dict[str, str]) -> dict[str, str]:
        """μ_e over ``ips``: β with each (source ip → target ip) binding enforced by swaps.

        ``bindings`` targets are incident addresses (role IPs or device addresses). The map
        is injective on ``ips`` ∪ bound sources.
        """
        ips = [i for i in dict.fromkeys(list(ips) + list(bindings)) if i is not None]
        fwd = {i: self.map_base(i) for i in ips}
        inv: dict[str, str] = {}
        for k, v in fwd.items():
            inv.setdefault(v, k)
        # β is injective except for specials mapping to themselves (dedupe safe: specials never bound)
        for src, tgt in bindings.items():
            if src is None or tgt is None:
                continue
            cur = fwd.get(src)
            if cur == tgt:
                continue
            other = inv.get(tgt)
            fwd[src] = tgt
            inv[tgt] = src
            if other is not None and other != src:
                fwd[other] = cur
                inv[cur] = other
            elif cur is not None and inv.get(cur) == src:
                del inv[cur]
        return fwd


# --------------------------------------------------------------------------- frame helpers
def apply_ip_map(df: pl.DataFrame, mapping: dict[str, str]) -> pl.DataFrame:
    """Replace src/dst through ``mapping`` (every value present must be a key)."""
    if df.height == 0:
        return df
    keys = list(mapping)
    vals = [mapping[k] for k in keys]
    lut = pl.DataFrame({"_k": keys, "_v": vals}, schema={"_k": pl.Utf8, "_v": pl.Utf8})
    out = df
    for c in ("src", "dst"):
        out = (out.join(lut.rename({"_k": c, "_v": f"_{c}_new"}), on=c, how="left", maintain_order="left")
               .with_columns(pl.col(f"_{c}_new").alias(c)).drop(f"_{c}_new"))
    if out["src"].null_count() or out["dst"].null_count():
        raise ValueError("apply_ip_map: unmapped address")
    return out


def shift_times(df: pl.DataFrame, offset: float) -> pl.DataFrame:
    return df.with_columns(pl.col("ts") + offset, pl.col("te") + offset)


def round_times(df: pl.DataFrame, res: float = 1.0) -> pl.DataFrame:
    """ts, te rounded half-up to ``res``; te ≥ ts; ts_res = res."""
    ts = ((pl.col("ts") / res) + 0.5).floor() * res
    te = ((pl.col("te").fill_null(pl.col("ts")) / res) + 0.5).floor() * res
    return df.with_columns(ts.alias("ts"), te.alias("te")).with_columns(
        pl.max_horizontal("ts", "te").alias("te"), pl.lit(float(res)).alias("ts_res"))


def visible_mask(df: pl.DataFrame) -> pl.Series:
    """True for records with ≥ 1 local endpoint (after rewriting)."""
    if df.height == 0:
        return pl.Series("vis", [], dtype=pl.Boolean)
    ips = pl.concat([df["src"], df["dst"]]).unique().drop_nulls().to_list()
    loc = {i: is_local(i) for i in ips}
    lut = pl.DataFrame({"_ip": list(loc), "_loc": list(loc.values())}, schema={"_ip": pl.Utf8, "_loc": pl.Boolean})
    s = df.select("src").join(lut.rename({"_ip": "src"}), on="src", how="left", maintain_order="left")["_loc"]
    d = df.select("dst").join(lut.rename({"_ip": "dst"}), on="dst", how="left", maintain_order="left")["_loc"]
    return (s.fill_null(False) | d.fill_null(False)).alias("vis")


__all__ = ["IPPlan", "apply_ip_map", "shift_times", "round_times", "visible_mask", "is_local", "is_private",
           "is_special", "is_role_ip", "ROLE_NETS", "EXT_NET", "TESTBED_PREFIX"]

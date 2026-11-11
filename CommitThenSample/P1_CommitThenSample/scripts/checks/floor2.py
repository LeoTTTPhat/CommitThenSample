"""Measure unit statistics and floor values for the revised CtS design.
Cost s_i = pcap-record bytes (caplen + 16) of the unit's packets. Units = bidirectional
5-tuple per 30-s epoch (epoch grid anchored at the first packet). Zero-payload units are
exempt from content sampling. Usage: floor2.py <pcap> [max_packets]"""
import sys, collections, dpkt, numpy as np
path = sys.argv[1]; MAXP = int(sys.argv[2]) if len(sys.argv) > 2 else 6_000_000
T = 30.0; H = 1024; PHIT = 0.1
U = {}  # key -> [npk, pay, s, sH_done(flag), sH, paycum]
t0 = None; n = 0; tmax = 0.0
f = open(path, 'rb')
try:
    rd = dpkt.pcap.Reader(f)
except Exception:
    f.seek(0); rd = dpkt.pcapng.Reader(f)
lt = rd.datalink()
try:
    for ts, buf in rd:
        n += 1
        if t0 is None: t0 = ts
        tmax = max(tmax, ts)
        e = int((ts - t0) // T)
        cost = len(buf) + 16
        try:
            if lt == dpkt.pcap.DLT_LINUX_SLL:
                l2 = dpkt.sll.SLL(buf); ip = l2.data
            else:
                l2 = dpkt.ethernet.Ethernet(buf); ip = l2.data
        except Exception:
            key = (e, 'bad'); pay = 0; ip = None
        if isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
            l4 = ip.data; proto = ip.p if isinstance(ip, dpkt.ip.IP) else ip.nxt
            if isinstance(l4, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                a = (bytes(ip.src), l4.sport); b = (bytes(ip.dst), l4.dport); pay = len(l4.data)
            else:
                a = (bytes(ip.src), 0); b = (bytes(ip.dst), 0); pay = 0
            key = (e, proto) + tuple(sorted([a, b]))
        elif ip is not None:
            key = (e, 'l2', bytes(getattr(l2, 'src', b''))); pay = 0
        u = U.get(key)
        if u is None:
            u = [0, 0, 0, 0, 0, 0]; U[key] = u
        u[0] += 1; u[1] += pay; u[2] += cost
        if not u[3]:
            u[4] += cost; u[5] += pay
            if u[5] >= H: u[3] = 1
        if n >= MAXP: break
except (dpkt.NeedData, dpkt.UnpackError, ValueError) as ex:
    print("reader stopped:", type(ex).__name__)
dur = tmax - t0
A = np.array([[k[0], v[0], v[1], v[2], v[4]] for k, v in U.items()], dtype=np.float64)
ep, npk, pay, s, sH = A.T
payu = pay > 0
cF = sH + PHIT * (s - sH)
E = int(ep.max()) + 1
S_all = s.sum(); S_pay = s[payu].sum(); CF_pay = cF[payu].sum()
print(f"file={path.split('/')[-1]} packets={n} dur_h={dur/3600:.2f} pps={n/dur:.2f} bytes_per_day_GB={S_all/dur*86400/1e9:.3f}")
print(f"epochs={E} units={len(s)} payload_units={payu.sum()} frac_zero_payload={1-payu.mean():.3f}")
pu_ep = np.bincount(ep[payu].astype(int), minlength=E)
print(f"payload units/epoch: mean={pu_ep.mean():.2f} median={np.median(pu_ep):.0f} p90={np.percentile(pu_ep,90):.0f} frac_epochs_with_0={np.mean(pu_ep==0):.3f}")
print(f"S_all/S_pay={S_all/S_pay:.2f}  S_pay/CF_pay (capped gain)={S_pay/CF_pay:.3f}  frac payload units with P>H={np.mean(pay[payu]>=H):.4f}")
for rho in (0.01, 0.02, 0.05):
    for g in (0.2, 0.3):
        old = g * rho * S_all / cF.sum()   # old design: floor on all units, capped
        new = g * rho * S_all / S_pay      # revised: payload units only, uncapped
        cap = g * rho * S_all / CF_pay     # payload only, capped (phi_t=0.1)
        print(f"  rho={rho:.2f} gamma={g}: phi0_old={old:.4f} phi0_new(uncapped)={new:.4f} phi0_capped={cap:.4f} miss1={1-new:.4f} k_for_95pct={int(np.ceil(np.log(0.05)/np.log(1-min(new,0.999))))}")
meta = payu.sum() * (128 + 32) + (~payu).sum() * (56 + 32)
print(f"metadata(raw, uncompressed est.)/pcap bytes = {meta/S_all:.3f}")

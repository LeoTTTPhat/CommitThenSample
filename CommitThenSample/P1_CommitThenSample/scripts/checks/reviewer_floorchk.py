import sys, dpkt, socket, numpy as np, collections
path=sys.argv[1]; T=30.0; H=1024; phit=0.1
units=collections.defaultdict(lambda:[0,0,[]])  # key->(npk,payload,list of pkt costs)
t0=None
with open(path,'rb') as f:
    try: rd=dpkt.pcap.Reader(f)
    except Exception: f.seek(0); rd=dpkt.pcapng.Reader(f)
    n=0
    for ts,buf in rd:
        n+=1
        if t0 is None: t0=ts
        e=int((ts-t0)//T)
        try:
            eth=dpkt.ethernet.Ethernet(buf); ip=eth.data
        except Exception: continue
        if isinstance(ip,(dpkt.ip.IP,dpkt.ip6.IP6)):
            l4=ip.data; proto=ip.p if isinstance(ip,dpkt.ip.IP) else ip.nxt
            if isinstance(l4,(dpkt.tcp.TCP,dpkt.udp.UDP)):
                a=(bytes(ip.src),l4.sport); b=(bytes(ip.dst),l4.dport); pay=len(l4.data)
            else:
                a=(bytes(ip.src),0); b=(bytes(ip.dst),0); pay=len(l4) if isinstance(l4,(bytes,bytearray)) else len(bytes(l4))
                pay=0
            key=(e,proto)+tuple(sorted([a,b]))
        else:
            key=(e,'l2',bytes(eth.src)); pay=0
        u=units[key]; u[0]+=1; u[1]+=pay; u[2].append(pay+12)
        if n>=4_000_000: break
print("packets",n,"units",len(units))
ep=collections.defaultdict(lambda:[0.0,0.0,0.0,0.0,0,0])
for key,(npk,pay,costs) in units.items():
    e=key[0]; s=pay+12*npk+64
    c=np.cumsum(costs); idx=np.searchsorted(c,min(c[-1],H))
    sH=c[idx]+64; sT=s-sH
    cF=sH+phit*sT
    d=ep[e]; d[0]+=s; d[1]+=cF
    if pay>0: d[2]+=s; d[3]+=cF; d[4]+=1
    d[5]+=1
r_all=[]; r_pay=[]; fz=[]; 
for e,(s,cF,sp,cFp,npay,nu) in ep.items():
    r_all.append(s/cF); 
    if cFp>0: r_pay.append(sp/cFp)
    fz.append(1-npay/nu)
tot=np.array([[v[0],v[1],v[2],v[3],v[4],v[5]] for v in ep.values()]).sum(0)
print("epochs",len(ep)," frac zero-payload units (median over epochs) %.3f"%np.median(fz), " overall %.3f"%(1-tot[4]/tot[5]))
for rho in (0.02,0.05):
  for g in (0.2,):
    # floor over all units: phi=g*rho*median(sum s/sum cF)
    phi_all=g*rho*np.median(r_all)
    # floor only on payload units with budget fraction g of rho*sum s (all) spent on payload units
    phi_pay=g*rho*np.median([v[0]/v[3] for v in ep.values() if v[3]>0])
    print(f"rho={rho} gamma={g}: phi0 all-units={phi_all:.4f}  phi0 payload-only={phi_pay:.4f}  ratio={phi_pay/phi_all:.1f}")

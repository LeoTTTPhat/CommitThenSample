import numpy as np, time
rng=np.random.default_rng(1)
E,ne=100,2000; n=E*ne
s=np.maximum(64,np.exp(rng.normal(np.log(300),2.0,n))).round()   # content bytes (heavy tailed)
host=rng.integers(0,200,n)
att_hosts=np.arange(5)
is_att=(np.isin(host,att_hosts)) & (rng.random(n)<0.15)
npk=np.maximum(1,(s/200).round())
f_ind=(is_att & (rng.random(n)<0.7)).astype(float)            # rule-hit indicator, b=1
f_cnt=np.where(is_att, np.minimum(npk, rng.poisson(3,n)+1),0.).astype(float)  # count bounded by npk
b_cnt=npk
print("Q_ind=",f_ind.sum()," Q_cnt=",f_cnt.sum(), " frac att flows=",is_att.mean())
# noisy AI utility: predicted sqrt E[f^2|m]
score=np.clip(0.6*is_att+0.25*rng.random(n)+0.02,0,1)
def waterfill(a,s,B,floor):
    lo,hi=1e-12,1e12
    for _ in range(100):
        lam=np.sqrt(lo*hi); p=np.clip(lam*a/np.sqrt(s),floor,1)
        if (p*s).sum()>B: hi=lam
        else: lo=lam
    return np.clip(lo*a/np.sqrt(s),floor,1)
psi=lambda l: -np.log1p(-l)-l
def lcb_radius(V,alpha,etas):
    w=1/len(etas); lo,hi=0.0,1e12
    for _ in range(200):
        m=(lo+hi)/2
        M=(w*np.exp(np.minimum(etas*m-etas**2*V/2,700))).sum()
        if M>=1/alpha: hi=m
        else: lo=m
    return hi
def ucb_radius(Wfun,alpha,etas):
    # Wfun(eta) = sum psi(eta c_i) X_i^2/c_i^2
    w=1/len(etas); Ws=np.array([Wfun(e) for e in etas]); lo,hi=0.0,1e15
    for _ in range(200):
        m=(lo+hi)/2
        M=(w*np.exp(np.minimum(etas*m-Ws,700))).sum()
        if M>=1/alpha: hi=m
        else: lo=m
    return hi
alpha=0.05; budget_frac=0.03; floor=0.01
R=200
res={"lcb_ok":0,"ucb_ok":0,"both_ok":0,"wald_final_ok":0,"wald_unif_ok":0}
widths=[];wald_w=[]
for f,b,name in [(f_ind,np.ones(n),"ind"),(f_cnt,b_cnt,"cnt")]:
    bmax=b.max(); cstar=bmax/floor
    etas_l=np.geomspace(1e-6,10,40); etas_u=np.geomspace(1e-7,0.999/cstar,40)
    ok=dict(l=0,u=0,b=0,wf=0,wu=0); W=[];WW=[]
    t0=time.time()
    for r in range(R):
        rr=np.random.default_rng(1000+r)
        Bcarry=0.0; boost=np.ones(200)
        Shat=0.0;Q=0.0;V=0.0;Vht=0.0; Xs=[];cs=[]
        allok_l=allok_u=allok_w=True
        for e in range(E):
            idx=slice(e*ne,(e+1)*ne)
            Be=budget_frac*s[idx].sum()+Bcarry
            a=score[idx]*boost[host[idx]]*np.maximum(b[idx],1)
            p=waterfill(a,s[idx],max(Be,1.0),floor)
            Z=rr.random(ne)<p
            Bcarry=np.clip(Be-(Z*s[idx]).sum(),-0.5*Be,0.5*Be)
            X=np.where(Z,f[idx]/p,0.0)
            Shat+=X.sum(); Q+=f[idx].sum(); V+=(X**2).sum(); Vht+=(np.where(Z,f[idx]**2*(1-p)/p**2,0)).sum()
            Xs.append(X[Z]); cs.append((b[idx]/p)[Z])
            # adaptive boost: hosts with observed hits get boosted
            hit_hosts=np.unique(host[idx][Z & (f[idx]>0)]); boost[hit_hosts]=5.0
            if e%10==9:
                rl=lcb_radius(V,alpha/2,etas_l)
                Xa=np.concatenate(Xs); ca=np.concatenate(cs)
                ru=ucb_radius(lambda et: (psi(et*ca)*Xa**2/ca**2).sum(),alpha/2,etas_u)
                L=Shat-rl; U=Shat+ru
                allok_l&= (L<=Q); allok_u&=(Q<=U)
                hw=1.96*np.sqrt(Vht); allok_w&= abs(Shat-Q)<=hw
        ok['l']+=allok_l; ok['u']+=allok_u; ok['b']+=(allok_l and allok_u); ok['wf']+=abs(Shat-Q)<=1.96*np.sqrt(Vht); ok['wu']+=allok_w
        W.append(((Shat-rl)/Q,(Shat+ru)/Q)); WW.append(1.96*np.sqrt(Vht)/Q)
    W=np.array(W)
    print(name,"Q=",Q,"time %.1fs"%(time.time()-t0))
    print("  time-uniform cover: LCB %.3f UCB %.3f both %.3f | Wald final %.3f Wald unif %.3f"%(ok['l']/R,ok['u']/R,ok['b']/R,ok['wf']/R,ok['wu']/R))
    print("  median final rel LCB %.3f  rel UCB %.3f  Wald rel half-width %.3f"%(np.median(W[:,0]),np.median(W[:,1]),np.median(WW)))

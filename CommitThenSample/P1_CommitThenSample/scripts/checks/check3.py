import numpy as np
rng=np.random.default_rng(2)
n=10**6
s=np.maximum(64,np.exp(rng.normal(np.log(300),2.0,n)))
for frac in [0.01,0.03,0.1]:
    B=frac*s.sum()
    for H in [512,2048,8192,np.inf]:
        c=np.minimum(s,H)
        for gamma in [0.1,0.3]:
            phi=min(1,gamma*B/c.sum())
            print(f"budget {frac:.0%} H={H} gamma={gamma}: floor phi={phi:.4f}  miss(k=1)={1-phi:.3f} miss(k=10)={(1-phi)**10:.3f} miss(k=100)={(1-phi)**100:.2e}")
# variance inflation identity: ratio = 1+CV_w(kappa)^2 (no caps)
a=np.abs(rng.lognormal(0,1,20000)); s2=rng.lognormal(5,1,20000)
ahat=a*np.exp(rng.normal(0,0.5,20000))
B=1e-3*(s2).sum()
def var(ah):
    lam=B/(ah*np.sqrt(s2)).sum(); p=lam*ah/np.sqrt(s2); assert p.max()<1
    return (a**2*(1-p)/p).sum(), (a**2/p).sum()
v1,v1n=var(ahat); v0,v0n=var(a)
k=a/ahat; w=ahat*np.sqrt(s2); w/=w.sum()
print("ratio(no -a^2 term)=",v1n/v0n," 1+CV^2=",(w*k**2).sum()/(w*k).sum()**2, " Kantorovich bound=",(k.max()+k.min())**2/(4*k.max()*k.min()))
# hiding capacity vs Fan slack for floor design
for floor in [0.005,0.01,0.02,0.05]:
    L=np.log(1/0.025)
    print(f"floor {floor}: indicator hiding capacity ~ {L/floor:.0f} units (log(1/alpha)/floor)")

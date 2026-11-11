import numpy as np, math
# Prop 3 example
pi,k=0.05,10
p1=1-(1-pi)**k
print("Prop3 M=1:",round(p1,4)," M=5:",round(p1**5,5))
# Carry-over controller as written vs corrected
rng=np.random.default_rng(0)
Bbar=1.0; E=3000; R=400
def run(fixed):
    Dfin=[];spend=[]
    for r in range(R):
        D=0.0; tot=0.0
        for e in range(E):
            Be=Bbar+np.clip(D,-Bbar/2,Bbar/2)
            cost=Be*rng.lognormal(-0.5*0.8**2,0.8)  # E[cost]=Be, noisy realised cost
            D = D + (Bbar if fixed else Be) - cost
            tot+=cost
        Dfin.append(D); spend.append(tot/(E*Bbar))
    return np.mean(np.abs(Dfin)), np.percentile(spend,[5,50,95])
print("as written |D_E| mean, spend/(E*Bbar) pct:",run(False))
print("corrected  |D_E| mean, spend/(E*Bbar) pct:",run(True))
# Unbiased two-tier variance estimator exists when pU=0: check pairwise joint inclusion
piH,phit,pU=0.03,0.1,0.0
piT=pU+(piH-pU)*phit
pi_ht=piT; pi_tt=pU+(piH-pU)*phit**2
print("joint incl head-tail",pi_ht," tail-tail",pi_tt," (all >0 -> SYG/HT variance estimator unbiased)")
# MC check of unbiased variance estimator for one unit: head value gH, tail values g
gH=1.0; g=np.array([2.0,0,3.0,1.0])
J=len(g); n=400000
U=rng.random(n); ZH=U<piH; ZF=U<pU
ZT=ZF[:,None] | (ZH[:,None] & (rng.random((n,J))<phit))
X=ZH*gH/piH + (ZT*g).sum(1)/piT
y=np.concatenate([[gH],g]); pis=np.array([piH]+[piT]*J)
Zall=np.concatenate([ZH[:,None],ZT],1)
# joint matrix
P=np.full((J+1,J+1),pi_tt); P[0,:]=piT; P[:,0]=piT; P[0,0]=piH
np.fill_diagonal(P[1:,1:],piT)
W=(P-np.outer(pis,pis))/(P*np.outer(pis,pis))*np.outer(y,y)
vhat=np.einsum('ni,ij,nj->n',Zall.astype(float),W,Zall.astype(float))
print("true Var",X.var()," mean vhat",vhat.mean())

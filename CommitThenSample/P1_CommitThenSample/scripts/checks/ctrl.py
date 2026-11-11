"""Horizon-budget lambda controller check (log-integral controller with saturation).
Synthetic gateway traffic: diurnal Poisson unit counts, lognormal sizes, bursts; floor phi0 on payload units."""
import numpy as np
rng = np.random.default_rng(1)
def sim(E=2880*2, rho=0.05, gamma=0.3, kappa=1.0, W=120, Lsat=np.log(16), realized=False, R=50):
    out = []
    for r in range(R):
        mu = 200*(1+0.6*np.sin(2*np.pi*np.arange(E)/2880)) * np.exp(rng.normal(0,0.3,E))
        burst = rng.random(E) < 0.01
        mu[burst] *= 20
        n = rng.poisson(mu)
        tot_bytes = []; units = []
        for e in range(E):
            s = rng.lognormal(6.0, 1.6, n[e]) + 80
            a = rng.lognormal(0, 1.5, n[e]) * (rng.random(n[e]) < 0.2)  # sparse utility
            units.append((s, a / np.sqrt(s)))
            tot_bytes.append(s.sum())
        Bbar = rho*np.mean(tot_bytes)
        phi0 = gamma*rho                     # payload-only floor ~ gamma*rho on CIC-like traffic
        # lambda_ref from a "training" half: bisection on mean cost
        def mean_cost(lam, idx):
            return np.mean([np.sum(np.clip(np.maximum(phi0, lam*units[e][1]), 0, 1)*units[e][0]) for e in idx])
        lo, hi = -30, 30
        tr = range(0, E, 7)
        for _ in range(50):
            m = (lo+hi)/2
            if mean_cost(np.exp(m), tr) > Bbar: hi = m
            else: lo = m
        lref = (lo+hi)/2
        D = 0.0; spend = 0.0; Dmax = 0.0
        for e in range(E):
            lam = np.exp(lref + kappa*np.clip(D/(W*Bbar), -Lsat, Lsat))
            s, rr = units[e]
            pi = np.clip(np.maximum(phi0, lam*rr), 0, 1)
            c = np.sum(pi*s)
            if realized:
                c = np.sum((rng.random(len(s)) < pi)*s)
            D = D + Bbar - c
            spend += c; Dmax = max(Dmax, abs(D))
        out.append((spend/(Bbar*E), Dmax/Bbar, abs(D)/Bbar))
    o = np.array(out)
    return o.mean(0), np.percentile(o[:,0],[5,95])
for realized in (False, True):
    for W in (30, 120):
        m, p = sim(W=W, realized=realized, R=20)
        print(f"realized={realized} W={W}: spend/target mean={m[0]:.4f} [5,95]%={p.round(4)} max|D|/Bbar={m[1]:.1f} final|D|/Bbar={m[2]:.1f}")

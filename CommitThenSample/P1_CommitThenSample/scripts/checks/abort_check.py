"""MC check of Prop. 2: LCS with X:=0 for withheld units stays valid under adversarial withholding;
'compliant epochs only' fails. K=16 grid, alpha/2=0.025."""
import numpy as np
rng = np.random.default_rng(7)
a2 = 0.025
def lcs_lower(Qhat, V, etas, w):
    # r^- = inf{r: sum w exp(eta r - eta^2 V/2) >= 1/a2}; bisection
    lo, hi = 0.0, max(1.0, Qhat + 10*np.sqrt(V+1) + 100)
    f = lambda r: np.sum(w*np.exp(np.minimum(etas*r - etas**2*V/2, 700))) - 1/a2
    if f(0) >= 0: return Qhat
    for _ in range(80):
        m = (lo+hi)/2
        if f(m) >= 0: hi = m
        else: lo = m
    return Qhat - hi
E = 400; R = 4000
pi = 0.05; f = 1.0
Vref = E*1/pi
eta_s = np.sqrt(2*np.log(2/0.05)/Vref)
etas = eta_s*10**np.linspace(-2, 2, 16); w = np.ones(16)/16
res = {}
for strat in ["honest", "compliant_only_incrim", "x0_incrim", "x0_withhold_large", "x0_random"]:
    miss = 0
    for r in range(R):
        Z = rng.random(E) < pi
        X = Z*f/pi
        if strat == "honest":
            Qh, V, Q = X.sum(), (X**2).sum(), E*f
        elif strat == "compliant_only_incrim":
            keep = Z            # void epochs where the hit was NOT retained; estimand over kept epochs
            Qh, V, Q = X[keep].sum(), (X[keep]**2).sum(), keep.sum()*f
        elif strat == "x0_incrim":
            Wd = ~Z             # same voiding, but estimand over all epochs, X:=0 for voided (already 0)
            Xp = X*(~Wd); Qh, V, Q = Xp.sum(), (Xp**2).sum(), E*f
        elif strat == "x0_withhold_large":
            # withhold retained units at random epochs to shrink V (tries to raise L for large eta)
            Wd = Z & (rng.random(E) < 0.5)
            Xp = X*(~Wd); Qh, V, Q = Xp.sum(), (Xp**2).sum(), E*f
        else:
            Wd = rng.random(E) < 0.3
            Xp = X*(~Wd); Qh, V, Q = Xp.sum(), (Xp**2).sum(), E*f
        L = lcs_lower(Qh, V, etas, w)
        miss += (Q < L)
    res[strat] = miss/R
for k, v in res.items():
    print(f"{k:24s} final-epoch LCS miscoverage = {v:.4f}  (nominal <= {a2})")

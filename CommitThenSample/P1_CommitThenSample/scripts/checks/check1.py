import numpy as np
# 1) nonneg self-normalized supermartingale increment: E[exp(eta(X-f) - eta^2 X^2/2)] <= 1, X = f Z/pi
worst=-1
for pi in np.geomspace(1e-4,1,60):
    for f in np.geomspace(1e-3,1e4,60):
        for eta in np.geomspace(1e-6,1e3,80):
            x=f/pi
            E = pi*np.exp(eta*(x-f)-eta**2*x**2/2) + (1-pi)*np.exp(-eta*f)
            worst=max(worst,E)
print("max E (LCB increment) =",worst)
# 2) UCB increment with Fan: E[exp(eta(f-X) - psi(eta c) X^2/c^2)] <= 1 where c >= b/pi >= X, eta*c<1
psi=lambda l: -np.log1p(-l)-l
worst=-1
for pi in np.geomspace(1e-4,1,40):
    for b in [1.0]:
        for f in np.linspace(0,1,21):
            c=b/pi
            for lam in np.linspace(1e-4,0.999,60):
                eta=lam/c; x=f/pi
                E = pi*np.exp(eta*(f-x)-psi(lam)*x**2/c**2) + (1-pi)*np.exp(eta*f)
                worst=max(worst,E)
print("max E (UCB increment) =",worst)

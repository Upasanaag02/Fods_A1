"""P3: L1-regularised regression (Lasso) via subgradient descent, vs Ridge.

Run from the repo root:  python p3_lasso.py
Console output is also saved to results/P3_output.txt; figures go to figs/.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import load, rmse

os.makedirs("figs", exist_ok=True)
os.makedirs("results", exist_ok=True)


class Tee:
    """Write to the terminal and to a file at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


_log = open("results/P3_output.txt", "w")
sys.stdout = Tee(sys.__stdout__, _log)


def header(title):
    print("\n" + "=" * 82)
    print(title)
    print("=" * 82)


# ------------------------------------------------------------------ data / basis
x, t = load("data/poly_train.csv")
xte, tte = load("data/poly_test.csv")
D = 9
leg = np.polynomial.legendre.legvander
Phi, Phite = leg(2 * x - 1, D), leg(2 * xte - 1, D)  # shifted Legendre basis
G, b = Phi.T @ Phi, Phi.T @ t
tt = float(t @ t)
ETA, ITERS = 0.05, 20000
lams = np.logspace(-4, 0, 21)
NOISE = 0.1
xg = np.linspace(0, 1, 400)
Phig = leg(2 * xg - 1, D)

W_REF = np.array([0, -0.928, 0.045, 1.117, 0.032, -0.201, 0, -0.026, 0, 0])
E_REF = 0.2625


def lasso_obj(w, lam):
    return 0.5 * (w @ G @ w - 2 * b @ w + tt) + lam * np.abs(w).sum()


def subgrad_descent(lam, track=False):
    """w <- w - eta (Phi^T(Phi w - t) + lam sign(w)),  w0 = 0,  fixed number of iterations."""
    w = np.zeros(D + 1)
    E = np.empty(ITERS + 1) if track else None
    if track:
        E[0] = lasso_obj(w, lam)
    for k in range(1, ITERS + 1):
        w = w - ETA * (G @ w - b + lam * np.sign(w))
        if track:
            E[k] = lasso_obj(w, lam)
    return w, E


def near_zero(w, lam):
    return np.abs(w) < 2 * ETA * lam  # the assignment's "approximately zero" rule


# ================================================================ 0. setup
header("0. SETUP")
print(f"basis: shifted Legendre, degree {D}  (np.polynomial.legendre.legvander(2x-1, {D}))")
print(f"cond(Phi^T Phi) = {np.linalg.cond(G):.3e}   (assignment says about 6e2)")
lmax = np.linalg.eigvalsh(G).max()
print(f"lambda_max(Phi^T Phi) = {lmax:.3f};  step size eta = {ETA} (smooth-part stability needs eta < {2 / lmax:.4f})")
print(f"{ITERS} iterations per lambda, {len(lams)} lambdas from {lams[0]:.0e} to {lams[-1]:.0e}")
print('"approximately zero" means |w_j| < 2*eta*lambda')

# ================================================================ 1. lambda = 0.1
header("1. CONVERGENCE AND VERIFICATION AT lambda = 0.1")
lam0 = 0.1
w, E = subgrad_descent(lam0, track=True)
best = np.minimum.accumulate(E)
r = t - Phi @ w
corr = Phi.T @ r
nz = near_zero(w, lam0)
print(f"E at iteration 0 / 100 / 1000 / 20000 = {E[0]:.4f} / {E[100]:.4f} / {E[1000]:.4f} / {E[-1]:.6f}")
print(f"final E(w)          = {E[-1]:.6f}")
print(f"best-so-far E       = {best[-1]:.6f}   (reference E* = {E_REF})")
print(f"E is not monotone: it rose on {int(np.sum(np.diff(E) > 0))} of {ITERS} steps")

print("\n(b) optimality conditions on the final iterate")
print(f"{'j':>3s} {'w_j':>10s} {'Phi_j^T r':>11s} {'lam*sign(w_j)':>14s}   status")
for j in range(D + 1):
    status = "~0   (need |Phi_j^T r| <= lam)" if nz[j] else "non-zero (need Phi_j^T r = lam*sign)"
    print(f"{j:3d} {w[j]:+10.4f} {corr[j]:+11.4f} {lam0 * np.sign(w[j]):+14.4f}   {status}")
v1 = np.max(np.abs(corr[~nz] - lam0 * np.sign(w[~nz]))) if (~nz).any() else float("nan")
v2 = np.max(np.abs(corr[nz])) / lam0 if nz.any() else float("nan")
print(f"max |Phi_j^T r - lam sign(w_j)| over non-zero coefficients = {v1:.4f}")
print(f"max |Phi_j^T r| / lam over ~0 coefficients               = {v2:.4f}   (should be <= 1)")

print("\n(c) comparison with the reference solution")
print(f"{'j':>3s} {'w (ours)':>10s} {'w* (ref)':>10s} {'diff':>10s}")
for j in range(D + 1):
    print(f"{j:3d} {w[j]:+10.4f} {W_REF[j]:+10.3f} {w[j] - W_REF[j]:+10.4f}")
print(f"||w - w*||_2 = {np.linalg.norm(w - W_REF):.4f}   (chatter of order eta*lambda = {ETA * lam0} is expected)")

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(E, label="E(w)", alpha=0.7)
ax.semilogy(best, "r", label="best-so-far E")
ax.axhline(E_REF, color="k", ls=":", label=f"reference E* = {E_REF}")
ax.set_xlabel("iteration")
ax.set_ylabel("E(w)")
ax.set_title("P3: subgradient descent on Lasso, lambda = 0.1")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("figs/P3_convergence.png", dpi=150)
plt.close(fig)

# ================================================================ 2-3. sweeps
header("2-3. SWEEP OVER lambda  (all numbers from the final iterate)")
WL = np.array([subgrad_descent(l)[0] for l in lams])
tr = np.array([rmse(t, Phi @ w) for w in WL])
te = np.array([rmse(tte, Phite @ w) for w in WL])
nzc = np.array([near_zero(WL[i], lams[i]).sum() for i in range(len(lams))])
print(f"{'lambda':>10s} {'train RMSE':>11s} {'test RMSE':>11s} {'||w||_1':>9s} {'#~0':>4s}   zero indices")
for i, l in enumerate(lams):
    zi = np.where(near_zero(WL[i], l))[0].tolist()
    print(f"{l:10.4f} {tr[i]:11.4f} {te[i]:11.4f} {np.abs(WL[i]).sum():9.4f} {nzc[i]:4d}   {zi}")

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.semilogx(lams, tr, "b", marker="o", ms=3, label="train RMSE")
ax.semilogx(lams, te, "r", marker="o", ms=3, label="test RMSE")
ax.axhline(NOISE, color="k", ls="--", label="noise level 0.1")
ax.set_xlabel("lambda")
ax.set_ylabel("RMSE")
ax.legend()
ax.grid(alpha=0.3)
ax.set_title("P3: Lasso train/test RMSE vs lambda")
fig.tight_layout()
fig.savefig("figs/P3_rmse_sweep.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
for j in range(D + 1):
    ax.semilogx(lams, WL[:, j], marker="o", ms=3, label=f"$w_{j}$")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("lambda")
ax.set_ylabel("coefficient")
ax.set_title("P3: Lasso coefficient path (Legendre basis)")
ax.legend(ncol=2, fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figs/P3_coefficient_path.png", dpi=150)
plt.close(fig)

# ================================================================ 4. lambda*
k = int(np.argmin(te))
lam_star, w_star = lams[k], WL[k]
zi = np.where(near_zero(w_star, lam_star))[0].tolist()
header("4. SELECTING lambda*")
print(f"lambda* = argmin test RMSE = {lam_star:.4e}")
print(f"test RMSE at lambda*       = {te[k]:.4f}   (train RMSE {tr[k]:.4f}, noise level {NOISE})")
print(f"number of ~0 coefficients  = {len(zi)}   indices = {zi}")
print("coefficients at lambda*:")
for j in range(D + 1):
    print(f"   w[{j}] = {w_star[j]:+.4f}{'   (~0)' if j in zi else ''}")

# ================================================================ 5-6. Lasso vs Ridge
header("5. LASSO vs RIDGE (both in the Legendre basis, same lambda grid)")
WR = np.array([np.linalg.solve(G + l * np.eye(D + 1), b) for l in lams])
ter = np.array([rmse(tte, Phite @ w) for w in WR])
trr = np.array([rmse(t, Phi @ w) for w in WR])
kr = int(np.argmin(ter))
lam_r, w_r = lams[kr], WR[kr]
print(f"Ridge best lambda = {lam_r:.4e} (test RMSE {ter[kr]:.4f})")
print(f"Lasso best lambda = {lam_star:.4e} (test RMSE {te[k]:.4f})\n")
print(f"{'method':<8s} {'best lambda':>12s} {'test RMSE':>10s} {'||w||_2':>9s} {'#|w_j|<0.01':>12s}")
print(f"{'Lasso':<8s} {lam_star:12.3e} {te[k]:10.4f} {np.linalg.norm(w_star):9.4f} {int((np.abs(w_star) < 0.01).sum()):12d}")
print(f"{'Ridge':<8s} {lam_r:12.3e} {ter[kr]:10.4f} {np.linalg.norm(w_r):9.4f} {int((np.abs(w_r) < 0.01).sum()):12d}")
print("\nRidge coefficients at its best lambda:")
for j in range(D + 1):
    print(f"   w[{j}] = {w_r[j]:+.4f}")

header("6. FITTED CURVES")
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(xg, np.sin(2 * np.pi * xg), "g--", label="sin(2$\\pi$x)")
ax.plot(xg, Phig @ w_star, "r", label=f"Lasso, $\\lambda^*$={lam_star:.2e}")
ax.plot(xg, Phig @ w_r, "m-.", label=f"Ridge, $\\lambda$={lam_r:.2e}")
ax.scatter(x, t, c="b", s=22, zorder=3, label="train")
ax.set_ylim(-1.6, 1.6)
ax.legend(fontsize=8)
ax.set_title("P3: Lasso vs Ridge fits (Legendre basis, d=9)")
fig.tight_layout()
fig.savefig("figs/P3_lasso_vs_ridge_fit.png", dpi=150)
plt.close(fig)
print("saved figs/P3_lasso_vs_ridge_fit.png")

# ================================================================ 7. discussion
header("7. DISCUSSION (draft for the report)")
print(f"""L1 gives exact zeros because of the optimality condition: a coefficient can sit at
zero only while the correlation of its feature with the residual stays within lambda
(|Phi_j^T r| <= lambda), and the corners of the l1 ball (diamond) lie on the axes, where the
SSE contours touch first. The coefficient path shows the coefficients shrinking and then
switching off one after another ({nzc[0]} of {D + 1} are ~0 at lambda = {lams[0]:.0e}, {nzc[-1]} of {D + 1} at
lambda = {lams[-1]:.0f}). Ridge's circular l2 ball is touched away from the axes, so it shrinks every
coefficient without setting any to zero: at its best lambda {int((np.abs(w_r) < 0.01).sum())} of {D + 1} Ridge coefficients have
|w_j| < 0.01, against {int((np.abs(w_star) < 0.01).sum())} of {D + 1} for Lasso. Past lambda* = {lam_star:.1e}, the penalty starts to dominate: train
RMSE rises monotonically ({tr[k]:.3f} -> {tr[-1]:.3f} at lambda = 1) and test RMSE rises too
({te[k]:.3f} -> {te[-1]:.3f}), i.e. the model underfits.""")

sys.stdout = sys.__stdout__
_log.close()
print("Done. Output saved to results/P3_output.txt, figures in figs/")
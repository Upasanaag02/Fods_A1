"""P4: Bayesian linear regression - MAP/ridge equivalence and the effect of N.

Run from the repo root:  python p4_bayesian.py
Console output is also saved to results/P4_output.txt; figures go to figs/.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import load, vander

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


_log = open("results/P4_output.txt", "w")
sys.stdout = Tee(sys.__stdout__, _log)


def header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ------------------------------------------------------------------ fixed hyperparameters
SIGMA2 = 0.01
BETA = 1 / SIGMA2  # 100
ALPHA = 0.05
D = 3
LAM = ALPHA / BETA
xg = np.linspace(0, 1, 200)
Pg = vander(xg, D)
FLOOR = 2 * 1.96 / np.sqrt(BETA)  # irreducible-noise band width


def posterior(x, t):
    """Return (m_N, S_N) with S_N^-1 = alpha I + beta Phi^T Phi, m_N = beta S_N Phi^T t."""
    Phi = vander(x, D)
    A = ALPHA * np.eye(D + 1) + BETA * Phi.T @ Phi
    mN = BETA * np.linalg.solve(A, Phi.T @ t)
    SN = np.linalg.solve(A, np.eye(D + 1))
    return mN, SN


def predictive(mN, SN):
    """m(x) = phi^T m_N,  s^2(x) = 1/beta + phi^T S_N phi on the dense grid."""
    mean = Pg @ mN
    par_var = np.einsum("ij,jk,ik->i", Pg, SN, Pg)
    s2 = 1 / BETA + par_var
    return mean, s2, par_var


# ================================================================ 0. setup
header("0. SETUP")
print(f"sigma^2 = {SIGMA2}  ->  beta = {BETA:g}")
print(f"prior precision alpha = {ALPHA}")
print(f"polynomial degree = {D}  (phi(x) = [1, x, x^2, x^3])")
print(f"ridge parameter lambda = alpha/beta = {LAM:g}")
print(f"irreducible-noise band width 2 x 1.96 x beta^-1/2 = {FLOOR:.4f}")
print(f"prediction grid: {len(xg)} points in [0, 1]")

# ================================================================ 1-2. N = 10
x, t = load("data/p4_train_N10.csv")
N = len(x)
Phi = vander(x, D)
mN, SN = posterior(x, t)
mean, s2, par_var = predictive(mN, SN)
s = np.sqrt(s2)

w_map = np.linalg.solve(Phi.T @ Phi + LAM * np.eye(D + 1), Phi.T @ t)  # solve, no inverse
diff = np.linalg.norm(w_map - mN)
curve_gap = np.abs(Pg @ w_map - mean).max()

header("1. POSTERIOR AND PREDICTIVE DISTRIBUTION  (p4_train_N10.csv, N = 10)")
print("posterior mean m_N and posterior std of each weight:")
for j in range(D + 1):
    print(f"   w[{j}]: m_N = {mN[j]:+9.4f}   std = {np.sqrt(SN[j, j]):8.4f}")
print(f"\npredictive std s(x):  min = {s.min():.4f} (at x = {xg[s.argmin()]:.3f}),  "
      f"max = {s.max():.4f} (at x = {xg[s.argmax()]:.3f})")
print(f"noise-only std beta^-1/2 = {1 / np.sqrt(BETA):.4f}")
print(f"95% interval width: min = {2 * 1.96 * s.min():.4f}, max = {2 * 1.96 * s.max():.4f}")

header("2. MAP / RIDGE EQUIVALENCE CHECK")
print(f"{'j':>3s} {'w_MAP':>12s} {'m_N':>12s} {'difference':>12s}")
for j in range(D + 1):
    print(f"{j:3d} {w_map[j]:+12.6f} {mN[j]:+12.6f} {w_map[j] - mN[j]:+12.2e}")
print(f"\n||w_MAP - m_N||_2             = {diff:.3e}")
print(f"max |y_MAP(x) - m(x)| on grid = {curve_gap:.3e}")
print("=> the MAP/ridge curve and the Bayesian predictive mean coincide to rounding error.")

fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(xg, mean - 1.96 * s, mean + 1.96 * s, color="tab:blue", alpha=0.2,
                label="95% predictive interval")
ax.plot(xg, mean, "b", lw=2.5, alpha=0.6, label="predictive mean m(x)")
ax.plot(xg, Pg @ w_map, "r--", lw=1.5, label="MAP / ridge fit")
ax.plot(xg, np.sin(2 * np.pi * xg), "g-.", label="sin(2$\\pi$x)")
ax.scatter(x, t, c="k", s=25, zorder=3, label="train (N=10)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("P4: Bayesian predictive distribution, N=10 (degree 3)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figs/P4_predictive.png", dpi=150)
plt.close(fig)

# ================================================================ 3. effect of N
header("3. EFFECT OF N  (alpha, beta fixed)")
fig, ax = plt.subplots(figsize=(8, 5))
rows = []
for n_file in [5, 10, 20, 50]:
    xn, tn = load(f"data/p4_train_N{n_file}.csv")
    m_n, S_n = posterior(xn, tn)
    _, s2n, pv = predictive(m_n, S_n)
    width = 2 * 1.96 * np.sqrt(s2n)
    rows.append((len(xn), width.mean(), width.max(), pv.mean(), pv.max()))
    ax.plot(xg, width, label=f"N = {len(xn)}")
ax.axhline(FLOOR, color="k", ls="--", label=f"noise floor = {FLOOR:.3f}")
ax.set_xlabel("x")
ax.set_ylabel("predictive interval width  2 x 1.96 x s(x)")
ax.set_title("P4: predictive interval width vs x for different N")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figs/P4_effect_of_N.png", dpi=150)
plt.close(fig)

print(f"{'N':>4s} {'mean width':>11s} {'max width':>10s} {'mean phi^T S phi':>17s} "
      f"{'max phi^T S phi':>16s} {'mean width / floor':>19s}")
for n, wm, wx, pm, px in rows:
    print(f"{n:4d} {wm:11.4f} {wx:10.4f} {pm:17.5f} {px:16.5f} {wm / FLOOR:19.3f}")
print(f"\nnoise floor (irreducible) = {FLOOR:.4f}  (beta^-1 = {1 / BETA:g} never shrinks with N)")

# ================================================================ 4. discussion
header("4. DISCUSSION (draft for the report)")
print(f"""More data raise the posterior precision S_N^-1 = alpha I + beta Phi^T Phi (each new point adds
beta phi phi^T), so the posterior covariance S_N shrinks and the parameter-uncertainty term
phi(x)^T S_N phi(x) falls: the mean of that term drops from {rows[0][3]:.4f} at N = {rows[0][0]} to {rows[-1][3]:.5f} at
N = {rows[-1][0]}, and the interval narrows from an average width of {rows[0][1]:.3f} to {rows[-1][1]:.3f}.
For N = 10 the band is narrowest near x = {xg[s.argmin()]:.2f} and widest at x = {xg[s.argmax()]:.2f}. The noise term
beta^-1 = {1 / BETA:g} is a property of the observations, not of the weights, so it does not depend
on N: the width can only approach the floor {FLOOR:.3f} and never go below it, however much data
we collect.""")

sys.stdout = sys.__stdout__
_log.close()
print("Done. Output saved to results/P4_output.txt, figures in figs/")
"""P2: Ridge regression as a MAP estimate (closed form + gradient descent).

Run from the repo root:  python p2_ridge_map.py
Console output is also saved to results/P2_output.txt; figures go to figs/.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import load, rmse, vander

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


_log = open("results/P2_output.txt", "w")
sys.stdout = Tee(sys.__stdout__, _log)


def header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


x, t = load("data/poly_train.csv")
xte, tte = load("data/poly_test.csv")
D = 9
Phi, Phite = vander(x, D), vander(xte, D)
lams = np.logspace(-10, 2, 49)
I = np.eye(D + 1)
xg = np.linspace(0, 1, 400)
NOISE = 0.1  # sqrt(0.01)


def ridge_closed(lam):
    """Solve (Phi^T Phi + lam I) w = Phi^T t as a linear system (no explicit inverse)."""
    return np.linalg.solve(Phi.T @ Phi + lam * I, Phi.T @ t)


def ridge_obj(w, lam):
    r = Phi @ w - t
    return 0.5 * float(r @ r) + 0.5 * lam * float(w @ w)


# ================================================================ 1. closed form
header("1. CLOSED-FORM RIDGE, d = 9 monomial basis, 49 values of lambda")
W = np.array([ridge_closed(l) for l in lams])
tr = np.array([rmse(t, Phi @ w) for w in W])
te = np.array([rmse(tte, Phite @ w) for w in W])
nw = np.linalg.norm(W, axis=1)
k = int(np.argmin(te))
print(f"{'lambda':>12s} {'train RMSE':>12s} {'test RMSE':>12s} {'||w||_2':>12s}")
for i in range(0, len(lams), 4):  # every 4th grid point keeps the table readable
    mark = "  <-- lambda*" if i == k else ""
    print(f"{lams[i]:12.3e} {tr[i]:12.4f} {te[i]:12.4f} {nw[i]:12.3e}{mark}")
print("(table shows every 4th grid point; full grid is used for all results below)")

# ================================================================ 2. GD agreement
lam0, eta = 1e-2, 0.04
A = Phi.T @ Phi + lam0 * I
w_ref = ridge_closed(lam0)
lmax = np.linalg.eigvalsh(A).max()
header(f"2. GRADIENT DESCENT vs CLOSED FORM at lambda = {lam0}  (original basis, eta = {eta})")
print(f"lambda_max(Phi^T Phi + lam I) = {lmax:.3f}")
print(f"stability needs eta < 2/lambda_max = {2 / lmax:.4f}  -> eta = {eta} is "
      f"{'SAFE' if eta < 2 / lmax else 'UNSTABLE'}")
w = np.zeros(D + 1)
hist = [ridge_obj(w, lam0)]
rels = [1.0]
it = 0
while rels[-1] >= 1e-6 and it < 5_000_000:
    w = w - eta * (Phi.T @ (Phi @ w - t) + lam0 * w)
    hist.append(ridge_obj(w, lam0))
    rels.append(np.linalg.norm(w - w_ref) / np.linalg.norm(w_ref))
    it += 1
print(f"\nGD reached relative error < 1e-6 after {it} iterations")
print(f"final relative error ||w_GD - w_cf|| / ||w_cf|| = {rels[-1]:.3e}")
print(f"\n{'iteration':>10s} {'ridge objective':>18s} {'relative error':>16s}")
for i in sorted({0, 1, 10, 100, 1000, 5000, 10000, 20000, it}):
    print(f"{i:10d} {hist[i]:18.6e} {rels[i]:16.3e}")
c0, c1 = np.linalg.cond(Phi.T @ Phi), np.linalg.cond(A)
print(f"\ncond(Phi^T Phi)           = {c0:.3e}")
print(f"cond(Phi^T Phi + lam I)   = {c1:.3e}   (reduced by a factor of {c0 / c1:.1e})")
print("The penalty adds lambda to every eigenvalue, lifting the near-zero ones, so the "
      "condition number collapses.")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogy(hist)
ax.set_xlabel("iteration")
ax.set_ylabel("ridge objective")
ax.set_title(f"P2: GD on ridge objective, lambda={lam0}, eta={eta}")
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("figs/P2_gd_convergence.png", dpi=150)
plt.close(fig)

# ================================================================ 3. RMSE sweep
lam_star = lams[k]
ok = te <= 1.05 * te[k]
idx = np.where(ok)[0]
contiguous = bool(np.all(np.diff(idx) == 1))
header("3. RMSE SWEEP AND CHOICE OF lambda*")
print(f"lambda* = argmin test RMSE      = {lam_star:.4e}")
print(f"test RMSE at lambda*            = {te[k]:.4f}   (noise level = {NOISE})")
print(f"train RMSE at lambda*           = {tr[k]:.4f}")
print(f"interval with test RMSE <= 1.05 x min : [{lams[idx].min():.3e}, {lams[idx].max():.3e}]"
      f"   ({len(idx)} grid points, contiguous = {contiguous})")
print(f"min test RMSE = {te[k]:.4f}; threshold = {1.05 * te[k]:.4f}")
print("grid points inside the 1.05 x min band: " + ", ".join(f"{l:.2e}" for l in lams[idx]))
span = te[idx.min():idx.max() + 1]
print(f"over the whole span, test RMSE ranges only from {span.min():.4f} to {span.max():.4f}")

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.semilogx(lams, tr, "b", label="train RMSE")
ax.semilogx(lams, te, "r", label="test RMSE")
ax.axhline(NOISE, color="k", ls="--", label="noise level 0.1")
ax.axvline(lam_star, color="gray", ls=":", label="$\\lambda^*$")
ax.set_xlabel("lambda")
ax.set_ylabel("RMSE")
ax.set_ylim(0, 1)
ax.legend()
ax.grid(alpha=0.3)
ax.set_title("P2: train/test RMSE vs lambda (d=9)")
fig.tight_layout()
fig.savefig("figs/P2_rmse_sweep.png", dpi=150)
plt.close(fig)

# ================================================================ 4. fitted curves
header("4. THREE FITS: overfit / lambda* / underfit")
print(f"{'case':<10s} {'lambda':>12s} {'||w||_2':>12s} {'train RMSE':>12s} {'test RMSE':>12s}")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, lam, name in zip(axes, [1e-10, lam_star, 1e-2], ["overfit", "lambda*", "underfit"]):
    w = ridge_closed(lam)
    print(f"{name:<10s} {lam:12.3e} {np.linalg.norm(w):12.4e} "
          f"{rmse(t, Phi @ w):12.4f} {rmse(tte, Phite @ w):12.4f}")
    ax.plot(xg, np.sin(2 * np.pi * xg), "g--", label="sin(2$\\pi$x)")
    ax.plot(xg, vander(xg, D) @ w, "r", label="ridge fit")
    ax.scatter(x, t, c="b", s=20, zorder=3, label="train")
    ax.set_ylim(-1.6, 1.6)
    ax.set_title(f"{name}: lambda={lam:.2e}")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figs/P2_three_fits.png", dpi=150)
plt.close(fig)

# ================================================================ 5. discussion
header("5. DISCUSSION (draft for the report)")
print(f"""Ridge is the MAP estimate under a zero-mean Gaussian prior, with lambda = alpha/beta.
The test-RMSE minimum is FLAT: {len(idx)} grid points lie within 5% of the best ({te[k]:.3f}), spanning
lambda from {lams[idx].min():.1e} to {lams[idx].max():.1e} (about {np.log10(lams[idx].max() / lams[idx].min()):.1f} decades), and over that whole span test RMSE
only moves between {span.min():.3f} and {span.max():.3f}. Below the interval the fit overfits (train
RMSE -> 0, huge weights, wiggles between the points). Above it the penalty dominates and
the fit underfits (weights shrink towards 0, the curve is too smooth, and both train and
test RMSE rise).""")

sys.stdout = sys.__stdout__
_log.close()
print("Done. Output saved to results/P2_output.txt, figures in figs/")
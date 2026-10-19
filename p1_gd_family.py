"""P1: Polynomial regression - closed form vs the gradient descent family.

Run from the repo root:  python p1_gd_family.py
Console output is also saved to results/P1_output.txt; figures go to figs/.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import SEED, load, vander

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


_log = open("results/P1_output.txt", "w")
sys.stdout = Tee(sys.__stdout__, _log)


def header(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


x, t = load("data/poly_train.csv")
N = len(x)
D = 9
xg = np.linspace(0, 1, 400)


# ---------------------------------------------------------------- closed form
def closed_form(d):
    Phi = vander(x, d)
    w, *_ = np.linalg.lstsq(Phi, t, rcond=None)  # SVD based, no explicit inverse
    return w


# ---------------------------------------------------------------- GD family
def sse(A, v):
    r = A @ v - t
    return 0.5 * float(r @ r)


def batch_gd(A, eta, epochs):
    a = np.zeros(A.shape[1])
    hist = [sse(A, a)]
    for _ in range(epochs):
        a = a - eta * A.T @ (A @ a - t)  # sum, not divided by N
        hist.append(sse(A, a))
    return a, np.array(hist)


def sgd(A, eta, epochs, seed):
    rng = np.random.default_rng(seed)
    a = np.zeros(A.shape[1])
    hist = [sse(A, a)]
    for _ in range(epochs):
        for n in rng.permutation(N):
            a = a - eta * (A[n] @ a - t[n]) * A[n]
        hist.append(sse(A, a))
    return a, np.array(hist)


def minibatch_gd(A, eta, epochs, seed, bs=3):
    rng = np.random.default_rng(seed)
    a = np.zeros(A.shape[1])
    hist = [sse(A, a)]
    for _ in range(epochs):
        idx = rng.permutation(N)
        for s in range(0, N, bs):  # batches of 3,3,3,1
            b = idx[s:s + bs]
            a = a - eta * A[b].T @ (A[b] @ a - t[b]) / len(b)  # averaged gradient
        hist.append(sse(A, a))
    return a, np.array(hist)


# ================================================================ 1. closed form
header("1. CLOSED-FORM FITS (np.linalg.lstsq), d = 1..9")
fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True, sharey=True)
print(f"{'degree':>6s} {'train SSE E(w)':>18s}")
for d, ax in zip(range(1, 10), axes.ravel()):
    w = closed_form(d)
    E = 0.5 * float(np.sum((vander(x, d) @ w - t) ** 2))
    print(f"{d:6d} {E:18.6e}")
    ax.plot(xg, np.sin(2 * np.pi * xg), "g--", lw=1, label="sin(2$\\pi$x)")
    ax.plot(xg, vander(xg, d) @ w, "r", label=f"degree {d}")
    ax.scatter(x, t, c="b", s=18, zorder=3, label="train")
    ax.set_ylim(-1.6, 1.6)
    ax.set_title(f"d = {d}")
    if d == 1:
        ax.legend(fontsize=7)
fig.suptitle("P1: closed-form fits, d = 1..9 (np.linalg.lstsq)")
fig.tight_layout()
fig.savefig("figs/P1_closed_form_fits.png", dpi=150)
plt.close(fig)

# ================================================================ 2. d = 9 setup
Phi = vander(x, D)
Q, R = np.linalg.qr(Phi)  # thin QR
w_closed = closed_form(D)
header("2. SETUP FOR d = 9")
print(f"N = {N},  shuffle seed = {SEED}")
print(f"cond(Phi)            = {np.linalg.cond(Phi):.3e}   (ill-conditioned)")
print(f"cond(Q)              = {np.linalg.cond(Q):.3e}   (perfectly conditioned)")
print(f"||Q^T Q - I||_F      = {np.linalg.norm(Q.T @ Q - np.eye(N)):.2e}")
print(f"max |Q Q^T - I|      = {np.abs(Q @ Q.T - np.eye(N)).max():.2e}   (rows orthonormal too)")
print(f"||w_closed||_2       = {np.linalg.norm(w_closed):.4e}")
print("w_closed:")
for j, wj in enumerate(w_closed):
    print(f"   w[{j}] = {wj:+.4e}")

# ================================================================ 3. loss curves
ETA, EPOCHS = 0.1, 2000
a_b, h_b = batch_gd(Q, ETA, EPOCHS)
a_s, h_s = sgd(Q, ETA, EPOCHS, SEED)
a_m, h_m = minibatch_gd(Q, ETA, EPOCHS, SEED)

SHOW = 1000  # epochs shown, long enough for all three curves to flatten
fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(h_b[:SHOW + 1], lw=4, alpha=0.5, label="Batch GD")
ax.semilogy(h_s[:SHOW + 1], "--", label="SGD")
ax.semilogy(h_m[:SHOW + 1], label="Mini-batch (3)")
ax.set_xlabel("epoch")
ax.set_ylabel("E (log)")
ax.set_title(f"P1: loss vs epoch, $\\eta$={ETA} (QR-reparametrised, d=9)")
ax.legend()
fig.tight_layout()
fig.savefig("figs/P1_loss_curves.png", dpi=150)
plt.close(fig)

header(f"3. LOSS CURVES  (eta = {ETA}, QR basis, E = SSE)")
print(f"{'epoch':>6s} {'Batch':>14s} {'SGD':>14s} {'Mini-batch':>14s}")
for ep in [0, 1, 5, 10, 50, 100, 200, 500, 1000, 2000]:
    print(f"{ep:6d} {h_b[ep]:14.4e} {h_s[ep]:14.4e} {h_m[ep]:14.4e}")
print(f"\nmax |E_batch - E_sgd| over the first 50 epochs = {np.abs(h_b[:50] - h_s[:50]).max():.2e}")
rb = (h_b[60] / h_b[10]) ** (1 / 50)
rm = (h_m[100] / h_m[10]) ** (1 / 90)
print(f"measured E shrink per epoch : Batch = {rb:.4f}  (theory (1-eta)^2     = {(1 - ETA) ** 2:.4f})")
print(f"                              Mini  = {rm:.4f}  (theory (1-eta/3)^2   = {(1 - ETA / 3) ** 2:.4f}, "
      "slightly lower because the singleton batch is faster)")

# ================================================================ 4. learning rate
header("4. LEARNING-RATE STUDY  (Batch GD, QR basis, 100 epochs)")
fig, ax = plt.subplots(figsize=(8, 5))
print(f"{'eta':>6s} {'E after 100 epochs':>20s}   regime")
runs = {}
for eta, lab, style, regime in [
        (0.01, "too small (0.01)", "-", "too small: slow"),
        (0.1, None, None, "good"),
        (0.5, "well-chosen (0.5)", "-", "good: fast"),
        (1.0, None, None, "good: fastest (one step)"),
        (2.0, "exactly 2.0", ":", "boundary: oscillates, never converges"),
        (2.1, "too large (2.1)", "-", "too large: diverges")]:
    a, h = batch_gd(Q, eta, 100)
    runs[eta] = (a, h)
    print(f"{eta:6.2f} {h[-1]:20.4e}   {regime}")
    if lab:
        ax.semilogy(h, style, label=lab)
ax.set_xlabel("epoch")
ax.set_ylabel("E (log)")
ax.set_title("P1: learning-rate study, Batch GD (d=9)")
ax.legend()
fig.tight_layout()
fig.savefig("figs/P1_learning_rate_study.png", dpi=150)
plt.close(fig)

a21, h21 = runs[2.1]
print(f"\neta = 2.1 after 100 epochs: E = {h21[-1]:.4e}")
print("   w = (original basis, w = R^-1 alpha)")
for j, wj in enumerate(np.linalg.solve(R, a21)):
    print(f"   w[{j}] = {wj:+.4e}")
print(f"\neta = 2.0: first E values {np.round(runs[2.0][1][:4], 4)} -> flips between "
      "alpha=0 and alpha=2 Q^T t, same loss each time")
_, h001 = batch_gd(Q, 0.01, 2000)
print("eta = 0.01 is slow but converges: E at epochs 100/500/1000/2000 = "
      + ", ".join(f"{h001[e]:.3e}" for e in (100, 500, 1000, 2000)))

# ================================================================ 5. verification
header(f"5. VERIFICATION vs CLOSED FORM  ({EPOCHS} epochs, eta = {ETA})")
print(f"||w_closed|| = {np.linalg.norm(w_closed):.4e}")
print(f"{'method':<12s} {'||w_GD - w_cf||_2':>20s} {'relative error':>16s}")
for name, a in [("Batch", a_b), ("SGD", a_s), ("Mini-batch", a_m)]:
    w = np.linalg.solve(R, a)
    err = np.linalg.norm(w - w_closed)
    print(f"{name:<12s} {err:20.4e} {err / np.linalg.norm(w_closed):16.4e}")
print("\nThe absolute error looks large-ish only because ||w|| ~ 1e5; the relative "
      "error ~1e-11 is rounding level.")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(xg, np.sin(2 * np.pi * xg), "g--", label="sin(2$\\pi$x)")
ax.plot(xg, vander(xg, D) @ w_closed, "k", lw=3, alpha=0.4, label="closed form")
for name, a, st in [("Batch", a_b, "b-"), ("SGD", a_s, "r--"), ("Mini", a_m, "m:")]:
    ax.plot(xg, vander(xg, D) @ np.linalg.solve(R, a), st, label=name)
ax.scatter(x, t, c="b", s=18, zorder=3)
ax.set_ylim(-1.6, 1.6)
ax.legend(fontsize=8)
ax.set_title("P1: d=9 fits, GD variants vs closed form")
fig.tight_layout()
fig.savefig("figs/P1_gd_vs_closed_fit.png", dpi=150)
plt.close(fig)

# ================================================================ 6. discussion
header("6. DISCUSSION (draft for the report)")
print("""Q is square (10x10) and orthogonal, so its ROWS are orthonormal too. An SGD
update for sample n moves alpha only along q_n, so it leaves every other
residual untouched, and each residual shrinks by exactly (1 - eta) per epoch
whatever the shuffle order: identical to Batch GD, hence the curves coincide.
Mini-batch GD averages the gradient, dividing the step by the batch size, so
residuals in 3-sample batches shrink only by (1 - eta/3) per epoch (only the
singleton batch uses the full eta). The slowest modes dominate E, so it
converges more slowly than both.""")

sys.stdout = sys.__stdout__
_log.close()
print("Done. Output saved to results/P1_output.txt, figures in figs/")
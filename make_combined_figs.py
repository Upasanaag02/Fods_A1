"""Stitch each problem's saved figures into one overview image per problem.
Run after p1..p4:  python make_combined_figs.py   ->  figs/combined/P{n}_all.png
"""
import glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

os.makedirs("figs/combined", exist_ok=True)
for n in (1, 2, 3, 4):
    files = sorted(glob.glob(f"figs/P{n}_*.png"))
    if not files:
        continue
    cols = 2
    rows = (len(files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 5.5 * rows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax in axes:
        ax.axis("off")
    for ax, f in zip(axes, files):
        ax.imshow(mpimg.imread(f))
        ax.set_title(os.path.basename(f), fontsize=10)
    fig.suptitle(f"Problem {n}: all figures", fontsize=16)
    fig.tight_layout()
    out = f"figs/combined/P{n}_all.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("saved", out)
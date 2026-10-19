# CS F320: Foundations of Data Science, Assignment 1

#seed=42


| Name | ID |
|---|---|
| Himangi Agrawal | 2023B3A70761H |
| Upasana Agrawal | 2023B5AD0962H |
| <third member name> | <third member ID> |

## What is in this repo

| Path | Contents |
|---|---|
| `p1_gd_family.py` | P1: polynomial regression, closed form vs Batch / SGD / Mini-batch GD (QR-reparametrised) |
| `p2_ridge_map.py` | P2: ridge regression as a MAP estimate, closed form and GD |
| `p3_lasso.py` | P3: Lasso via subgradient descent in the Legendre basis, compared with Ridge |
| `p4_bayesian.py` | P4: Bayesian linear regression, MAP/ridge equivalence, effect of N |
| `common.py` | Shared helpers (`load`, `vander`, `rmse`) and the random seed |
| `data/` | The six provided CSV files, unmodified |
| `figs/` | All figures produced by the scripts |
| `results/` | Console output of each script (`P1_output.txt` to `P4_output.txt`) |
| `report/` | `Report.docx` and `Report.pdf` |
| `make_report.py`, `make_zip.py` | Build the report and the submission zip |

## How to run

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt

python p1_gd_family.py
python p2_ridge_map.py
python p3_lasso.py
python p4_bayesian.py

python make_report.py     # builds report/Report.docx (and Report.pdf if LibreOffice is installed)
python make_zip.py        # builds FoDS_A1_PushAndPop.zip
```

Each script prints its results to the terminal, saves the same text to `results/`, and saves its plots to `figs/`. Run all scripts from the repository root.

## Notes

- Random seed: **42**, set in `common.py`. It is used only for the per-epoch shuffling in P1 (SGD and mini-batch GD); P2, P3 and P4 are deterministic.
- All data are loaded directly from `data/`. No data are generated or modified. The true function sin(2*pi*x) is used for plotting only, never for fitting.
- Linear systems are solved with `np.linalg.solve` / `np.linalg.lstsq`; no explicit matrix inverse is used for fitting (the P4 posterior covariance S_N is the one deliberate exception, computed by solving against the identity).
- Tiny differences in the last digits of rounding-level quantities (e.g. errors of order 1e-6 in P1) can occur between machines because of different BLAS builds.

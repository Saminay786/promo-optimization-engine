# Promo Optimization Engine

Decides **which product gets which discount in which week** to maximise incremental margin,
using a causal uplift model (Double ML) feeding a 0-1 integer program (OR-Tools CP-SAT),
served as a FastAPI service in Docker with CI.

## Pipeline
| Stage | File | What it does |
|---|---|---|
| 0 | `src/panel.py` | dunnhumby transactions → product × week panel (treatment = discount depth, confounders = display/mailer share, lags, seasonality, category promo pressure) |
| 1 | `src/uplift.py` | naive vs **Double ML** (EconML `LinearDML`, LightGBM nuisances, 5-fold cross-fitting) vs placebo; per-category uplift θ; MLflow tracking |
| 2 | `src/optimizer.py` | promo calendar as a 0-1 IP (budget, weekly slots, frequency, category cap, no back-to-back); benchmarked against a "deepest discount on top sellers" heuristic |
| 3 | `src/api.py` | `POST /optimize` → promo calendar + expected incremental margin |

## Data
This project uses **"The Complete Journey"** by dunnhumby: two years of household-level
transactions from ~2,500 households, with product, promotion (display/mailer) and coupon data.
Source: dunnhumby Source Files (also mirrored on Kaggle). The raw data is not included
in this repo; download it and place `transaction_data.csv`, `causal_data.csv` and
`product.csv` in `data/raw/`.
`data/processed/params.csv` and `theta.csv` are small derived summaries, included so the
API and Docker image run without the raw data.

## Run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/make_fake_data.py      # optional: test everything on fake data first
python -m src.panel                   # needs data/raw/*.csv
python -m src.uplift
python -m src.optimizer
uvicorn src.api:app --reload          # open http://127.0.0.1:8000/docs
pytest -q
docker build -t promo-engine . && docker run -p 8000:8000 promo-engine
```

## Results (dunnhumby The Complete Journey)
137 top-revenue grocery products × 99 weeks (11,892 product-weeks; fuel, gift cards and misc. transactions excluded).

| Metric | Value |
|---|---|
| Naive elasticity θ (no controls) | 3.49 (10% off → +42% units) |
| DML θ (95% CI) | 2.51 (2.18 – 2.85) (10% off → +29% units) |
| Placebo θ | 0.01 |
| Treatment residual variance ratio | 0.30 |
| Share of promo volume that was non-incremental | 43.8% |
| Incremental margin: MIP vs heuristic (same budget) | $719 vs $60 (8 weeks, $5k budget, 10 slots/week; MIP spends 37% less) |

## Limitations
- Observational data: θ is causal only under unconfoundedness given the controls.
- No cost data: margin assumed at 30% of shelf price; vendor funding assumed at 50% of the discount.
- Baseline is a rolling non-promo mean; no-back-to-back is a proxy for pull-forward; cannibalization is handled by a category cap, not cross-elasticities.

## License & citation
Released under the [MIT License](LICENSE). You're welcome to use and adapt it; please keep the copyright
notice and credit the original by linking to this repository:

> Samina Yasmin (2026). *Promo Optimization Engine*. https://github.com/Saminay786/promo-optimization-engine

GitHub's **"Cite this repository"** button (from [CITATION.cff](CITATION.cff)) gives the same citation in APA and BibTeX.

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
| 4 | `src/experiment.py` | store-level A/B test design for the calendar: A/A validity check, power simulation on real store-weeks, CUPED, minimum detectable effect |

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
python -m src.experiment              # optional: A/B test design (~1 min)
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

## Extension: Bayesian marketing-mix model (MMM)
A top-down complement to the product-level DML: weekly **category** sales explained by three promo levers
(discount rate, mailer placements, display placements) with **adstock** (carryover) and **saturation**
(diminishing returns), fitted with PyMC-Marketing across 8 categories in one model.

| Stage | File | What it does |
|---|---|---|
| M0 | `src/mmm_data.py` | category × week dataset (weeks 20–101, the 115 stores covered by `causal_data`) |
| M1 | `src/mmm.py` | Bayesian MMM: per-category lever strength; carryover and saturation shape shared across categories |
| M2 | `src/mmm_report.py` | contribution decomposition, ROI per lever, budget reallocation scenario, cross-check vs DML |

```bash
pip install -r requirements-mmm.txt   # Python >= 3.12
python -m src.mmm_data && python -m src.mmm && python -m src.mmm_report
```

**Results** (94% credible intervals; levers measured above each category's quietest week):

| Finding | Value |
|---|---|
| Convergence | 0 divergences, max r-hat 1.008, min ESS 611 |
| In-sample fit | R² 0.44–0.64 in 7 of 8 categories (deli meats 0.21); MAPE 4–11% |
| Discount share of units | 23% soft drinks, 19% frozen pizza; 4–9% elsewhere |
| Discount ROI ($ margin per $ retailer discount) | below 1 in 7 of 8 categories (e.g. soft drinks 0.69 [0.50–0.89]); only frozen pizza > 1 (1.24 [0.71–1.75]) |
| Display break-even cost | $0.005–0.08 per placement (product × store × week); mailer ~$0.001–0.002 |
| Reallocation (20% of discount spend → display) | frozen meat dinners improves with ≥ 81% probability up to $0.05/placement; frozen pizza loses at ≥ $0.03 |
| MMM vs DML discount effect (4 overlapping categories) | same ordering; MMM ≈ 43–55% of DML (bread, cheese, soft drinks; deli meats 19%), consistent with within-category cannibalisation |

Data issues found and handled: `causal_data.csv` only lists promoted product-store-weeks (so shares of its rows
measure promo *mix*, not amount; levers use placement counts); household recruitment inflates weeks 1–19;
discount dollars track sales mechanically (discount *rate* is used); levers are never zero (measured above
each category's minimum, so base vs promo is identifiable).

## Extension: how would we test it? (A/B test design)
The optimiser *predicts* a gain; a retailer would want to *see* it before rolling out. Proposed test:
randomise stores 50/50, treatment stores run the optimised calendar, control stores keep current practice;
metric = weekly promo margin per store on the planned products. Stores (not households) are the unit because
a promo calendar applies to everyone shopping in a store. The data has 582 stores; the planned products sell in 115,
of which 84 average ≥ $15 weekly margin. Using their real store-week history (weeks 20–101), 1,000 simulated tests per scenario:

| Question | Answer |
|---|---|
| Is the analysis valid? (A/A tests, no effect injected) | false-positive rate 3.7–5.6% vs 5% target |
| Does CUPED (pre-period adjustment) help? | yes: at a 10% true lift, power 63% vs 19% without it (8-week test) |
| Minimum detectable effect, 80% power (CUPED) | 14.8% (4 weeks), **13.4% (8 weeks)**, 12.3% (12 weeks); never reached without CUPED |
| Optimiser's predicted gain vs current practice | 3.2% of baseline margin (≈1.6% if the MMM's cannibalisation estimate holds) |
| Power to detect 3% with 84 stores, 8 weeks | ~10%: a test this size would almost always come back "no significant effect" |
| Stores needed to detect it (8 weeks, CUPED) | ~1,500 (≈6,000 at the MMM-adjusted 1.6%) |

Takeaway: the predicted gain is too small to confirm with a store test of this size, so a "no effect" result
would not mean the calendar failed. Note the panel only records ~2,500 households (~100 per store), so real stores
have far more customers and much lower relative noise; the *method* transfers, the store count does not.
Options: test on full-store sales, run longer, pool more categories into the metric, or use a switchback design.

## Limitations
- Observational data: θ is causal only under unconfoundedness given the controls.
- No cost data: margin assumed at 30% of shelf price; vendor funding assumed at 50% of the discount.
- Baseline is a rolling non-promo mean; no-back-to-back is a proxy for pull-forward; cannibalization is handled by a category cap, not cross-elasticities.
- The MMM cross-check suggests product-level θ is roughly 2× the category-level gain, so the optimizer's incremental margin is likely overstated; scaling θ by a category-level factor or modelling cross-effects is the next step.
- Experiment design: lift is injected as a uniform % on the whole store metric (a real calendar changes only some product-weeks); panel stores see only panel households' purchases.
- MMM: 82 weekly observations per category; mailer and display are correlated (0.6–0.7), so their split is uncertain; mailer/display have no cost data (break-even costs reported instead); standard adstock cannot represent the negative carryover of pull-forward.

## License & citation
Released under the [MIT License](LICENSE). You're welcome to use and adapt it; please keep the copyright
notice and credit the original by linking to this repository:

> Samina Yasmin (2026). *Promo Optimization Engine*. https://github.com/Saminay786/promo-optimization-engine

GitHub's **"Cite this repository"** button (from [CITATION.cff](CITATION.cff)) gives the same citation in APA and BibTeX.

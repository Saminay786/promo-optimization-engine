"""Stage 4: how would we PROVE the optimised calendar works? Design a store-level A/B test.

Treatment stores run the optimised promo calendar, control stores keep business-as-usual.
Metric: weekly promo margin per store on the planned products. Using REAL historical store-week
data (no effect injected = A/A test, known effect injected = power), we answer:
  1. Is the analysis valid?        A/A false-positive rate should be ~5%.
  2. How small an effect can we see? Minimum detectable effect (MDE) at 80% power.
  3. Does CUPED help?               Adjusting for each store's pre-period margin removes store-size noise.
  4. Is the optimiser's predicted gain detectable with the stores we have?

Run:  python -m src.experiment   (needs data/raw + data/processed/params.csv)
Out:  data/processed/experiment_power.csv, data/processed/experiment_summary.json
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    EXP_ALPHA,
    EXP_LIFTS,
    EXP_MIN_WEEKLY_MARGIN,
    EXP_POWER,
    EXP_PRE_WEEKS,
    EXP_SIMS,
    EXP_TEST_WEEKS,
    EXP_WEEKS,
    MARGIN_RATE,
    PROCESSED,
    RAW,
    VENDOR_FUNDING,
)


def store_week_margin(products: pd.Series) -> pd.DataFrame:
    """Rows = stores, columns = weeks, values = retailer margin on the given products.
    Same margin definition as the optimiser: 30% of shelf value minus the retailer-funded discount."""
    cols = ["PRODUCT_ID", "STORE_ID", "WEEK_NO", "QUANTITY", "SALES_VALUE", "RETAIL_DISC", "COUPON_MATCH_DISC"]
    tx = pd.read_csv(RAW / "transaction_data.csv", usecols=cols)
    tx = tx[tx.PRODUCT_ID.isin(products) & (tx.QUANTITY > 0)]
    shelf = tx.SALES_VALUE - tx.RETAIL_DISC - tx.COUPON_MATCH_DISC     # discounts are stored as negatives
    tx = tx.assign(margin=MARGIN_RATE * shelf + (1 - VENDOR_FUNDING) * tx.RETAIL_DISC)
    sw = tx.pivot_table(index="STORE_ID", columns="WEEK_NO", values="margin", aggfunc="sum", fill_value=0)
    return sw.loc[:, EXP_WEEKS[0]:EXP_WEEKS[1]]


def eligible(sw: pd.DataFrame, min_weekly: float = EXP_MIN_WEEKLY_MARGIN) -> pd.DataFrame:
    """Tiny stores add noise but almost no signal, so a real test would exclude them up front."""
    return sw[sw.mean(axis=1) >= min_weekly]


def cuped(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """CUPED: remove the part of y explained by the pre-period x. Same mean, smaller variance."""
    theta = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
    return y - theta * (x - x.mean())


def one_experiment(sw: np.ndarray, rng: np.random.Generator, test_weeks: int, lift: float,
                   use_cuped: bool, pre_weeks: int = EXP_PRE_WEEKS) -> float:
    """Simulate one test on history: random start week, random 50/50 store split, inject `lift`
    into treated stores, return the Welch t-test p-value."""
    n_stores, n_weeks = sw.shape
    start = rng.integers(pre_weeks, n_weeks - test_weeks + 1)
    x = sw[:, start - pre_weeks:start].sum(axis=1)
    y = sw[:, start:start + test_weeks].sum(axis=1)
    treat = rng.permutation(n_stores) < n_stores // 2
    y = np.where(treat, y * (1 + lift), y)
    if use_cuped:
        y = cuped(y, x)
    return float(stats.ttest_ind(y[treat], y[~treat], equal_var=False).pvalue)


def power_table(sw: pd.DataFrame, test_weeks: list[int] = EXP_TEST_WEEKS, lifts: list[float] = EXP_LIFTS,
                n_sims: int = EXP_SIMS, seed: int = 0) -> pd.DataFrame:
    """Share of simulated tests that detect the effect (p < alpha). At lift = 0 this is the false-positive rate."""
    rng, arr, rows = np.random.default_rng(seed), sw.to_numpy(dtype=float), []
    for weeks in test_weeks:
        for method, use_cuped in [("difference in means", False), ("CUPED", True)]:
            for lift in lifts:
                p = [one_experiment(arr, rng, weeks, lift, use_cuped) for _ in range(n_sims)]
                rows.append((weeks, method, lift, float(np.mean(np.array(p) < EXP_ALPHA))))
    return pd.DataFrame(rows, columns=["test_weeks", "method", "lift", "power"])


def mde(power: pd.DataFrame, target: float = EXP_POWER) -> pd.DataFrame:
    """Smallest lift reaching the target power (linear interpolation between simulated lifts)."""
    rows = []
    for (weeks, method), g in power[power.lift > 0].groupby(["test_weeks", "method"]):
        g = g.sort_values("lift")
        hit = g[g.power >= target]
        if hit.empty:
            value = np.nan                                   # not reached within the simulated range
        else:
            i = g.index.get_loc(hit.index[0])
            if i == 0:
                value = hit.lift.iloc[0]
            else:
                lo, hi = g.iloc[i - 1], g.iloc[i]
                value = lo.lift + (target - lo.power) * (hi.lift - lo.lift) / (hi.power - lo.power)
        rows.append((weeks, method, value))
    return pd.DataFrame(rows, columns=["test_weeks", "method", "mde"])


def predicted_lift(params: pd.DataFrame, plan_margin: float, weeks: int) -> float:
    """Optimiser's expected incremental margin as a % of the planned products' normal margin."""
    base_margin = MARGIN_RATE * (params.baseline * params.price).sum() * weeks
    return plan_margin / base_margin


def stores_needed(n_stores: int, mde_value: float, target_lift: float) -> int:
    """MDE shrinks with 1/sqrt(stores), so this many stores would make `target_lift` detectable."""
    return int(np.ceil(n_stores * (mde_value / target_lift) ** 2))


def run() -> dict:
    from src.optimizer import Rules, compare
    params = pd.read_csv(PROCESSED / "params.csv")
    sw = eligible(store_week_margin(params.PRODUCT_ID))
    power = power_table(sw)
    power.to_csv(PROCESSED / "experiment_power.csv", index=False)
    m = mde(power)

    rules = Rules(budget=5000)
    out = compare(params, rules)
    pred = predicted_lift(params, out["mip_minus_heuristic"], rules.weeks)   # test = optimiser vs current practice
    mde_8 = float(m.query("test_weeks == 8 and method == 'CUPED'").mde.iloc[0])
    aa = power[power.lift == 0]
    summary = {
        "eligible_stores": len(sw),
        "false_positive_rate": {f"{r.test_weeks}w {r.method}": round(r.power, 3) for r in aa.itertuples()},
        # None = 80% power not reached even at the largest simulated lift
        "mde": {f"{r.test_weeks}w {r.method}": None if np.isnan(r.mde) else round(float(r.mde), 4)
                for r in m.itertuples()},
        "predicted_lift": round(pred, 4),
        "predicted_lift_mmm_adjusted": round(pred / 2, 4),     # MMM: category-level gain ~half the product-level
        "stores_needed_8w_cuped": stores_needed(len(sw), mde_8, pred),
        "stores_needed_8w_cuped_mmm_adjusted": stores_needed(len(sw), mde_8, pred / 2),
    }
    (PROCESSED / "experiment_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))

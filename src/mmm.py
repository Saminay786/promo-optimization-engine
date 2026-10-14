"""MMM stage 1: Bayesian marketing-mix model per category (PyMC-Marketing).

units[c, t] = base[c] + traffic + seasonality
              + sum over levers of  beta[c,l] * saturation( adstock( lever[c, t] ) )

Run:  python -m src.mmm
Out:  data/processed/mmm_model.nc (+ printed diagnostics)
"""
import json
import warnings

import arviz as az
import numpy as np
import pandas as pd
from pymc_extras.prior import Prior
from pymc_marketing.mmm import MMM, DataDerivedScaling, GeometricAdstock, LogisticSaturation, Scaling

from src.config import PROCESSED
from src.mmm_data import LEVERS

CONTROLS = ["households_z"]         # panel traffic (standardised): drives sales but is not a promo lever
SAMPLER = {"nuts_sampler": "nutpie", "draws": 1000, "tune": 1000, "chains": 4, "target_accept": 0.95}
MODEL_PATH = PROCESSED / "mmm_model.nc"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED / "mmm_weekly.parquet")
    # Standardise the control: raw households (~1,275, CV 3%) is nearly constant, so its effect and the
    # intercept trade off perfectly (negative intercepts + divergences). Centred, the intercept = typical week.
    df["households_z"] = (df.households - df.households.mean()) / df.households.std()
    # Levers are NEVER off (every week has some discount, mailer and display), so "zero promo" is outside the
    # data and base vs promo can't be separated. Measure each lever ABOVE its category's quietest week instead:
    # contributions then mean "extra sales vs the least-promoted week".
    for lever in LEVERS:
        df[lever] = df[lever] - df.groupby("category")[lever].transform("min")
    X = df[["date", "category", *LEVERS, *CONTROLS]]
    return X, df.units


def build_model() -> MMM:
    return MMM(date_column="date", channel_columns=LEVERS, control_columns=CONTROLS,
               target_column="units", dims=("category",),
               # carryover (how fast effects fade) and saturation shape (where returns flatten) are SHARED across
               # categories; each category keeps its own strength (beta). 82 weeks can't support 9 free
               # lever parameters per category.
               adstock=GeometricAdstock(l_max=4, priors={"alpha": Prior("Beta", alpha=1, beta=3, dims="channel")}),
               saturation=LogisticSaturation(priors={
                   "lam": Prior("Gamma", alpha=3, beta=1, dims="channel"),
                   "beta": Prior("HalfNormal", sigma=2, dims=("category", "channel"))}),
               model_config={"intercept": Prior("HalfNormal", sigma=2, dims="category")},  # base sales >= 0
               yearly_seasonality=2,
               # scale each category by ITS OWN max (default pools all categories into one global max,
               # so small categories like deli meats get squeezed and the priors misfit -> divergences)
               scaling=Scaling(target=DataDerivedScaling(dims=(), method="max"),
                               channel=DataDerivedScaling(dims=(), method="max")))


def fit_model(X: pd.DataFrame, y: pd.Series) -> MMM:
    mmm = build_model()
    mmm.fit(X, y, progressbar=False, random_seed=42, sample_kwargs=SAMPLER)
    return mmm


def diagnostics(mmm: MMM, X: pd.DataFrame, y: pd.Series) -> dict:
    """Convergence (r-hat, ESS, divergences) and in-sample fit per category."""
    post = mmm.idata["posterior"]
    params = [v for v in post.data_vars if not v.endswith("_contribution")]
    summ = az.summary(mmm.idata, var_names=params)
    divergences = int(np.asarray(mmm.idata["sample_stats"]["diverging"]).sum())
    pp = mmm.sample_posterior_predictive(X, extend_idata=False, combined=True, progressbar=False)
    target_scale = mmm.get_scales_as_xarray()["target_scale"]   # model works on y / scale internally
    pred = (pp["y"] * target_scale).mean("sample").to_dataframe(name="pred").reset_index()
    fit = X[["date", "category"]].assign(units=y.values).merge(pred, on=["date", "category"])
    per_cat = {}
    for cat, g in fit.groupby("category"):
        r2 = 1 - ((g.units - g.pred) ** 2).sum() / ((g.units - g.units.mean()) ** 2).sum()
        per_cat[cat] = {"r2": round(float(r2), 3),
                        "mape": round(float((abs(g.units - g.pred) / g.units).mean()), 3)}
    return {"max_rhat": round(float(summ["r_hat"].max()), 3), "min_ess_bulk": int(summ["ess_bulk"].min()),
            "divergences": divergences, "n_draws": int(post.sizes["draw"] * post.sizes["chain"]),
            "fit": per_cat}


def run() -> MMM:
    warnings.filterwarnings("ignore", category=FutureWarning)
    X, y = load_data()
    mmm = fit_model(X, y)
    mmm.save(str(MODEL_PATH))
    diag = diagnostics(mmm, X, y)
    (PROCESSED / "mmm_diagnostics.json").write_text(json.dumps(diag, indent=1))
    print(json.dumps(diag, indent=1))
    return mmm


if __name__ == "__main__":
    run()

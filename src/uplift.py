"""Stage 1: how much EXTRA volume does a discount cause?

Compares 3 estimates of theta (effect of +1.0 discount depth on log units):
  1. naive  : within-product OLS, ignores displays/mailers  -> biased upward
  2. DML    : EconML LinearDML, controls for confounders     -> causal estimate, per category
  3. placebo: DML with shuffled treatment                   -> should be ~0

Run:  python -m src.uplift
Out:  data/processed/theta.csv  (+ MLflow run in ./mlruns)
"""
import mlflow
import numpy as np
import pandas as pd
from econml.dml import LinearDML
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

from src.config import PROCESSED, PROMO_THRESHOLD, TOP_CATEGORIES

W_COLS = ["display_share", "mailer_share", "lag1", "lag2", "week_of_year",
          "reg_price", "cat_promo", "prod_mean"]          # confounders / controls


def lgbm():
    return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                         min_child_samples=20, verbose=-1, random_state=42)


def prepare(panel: pd.DataFrame):
    df = panel.copy()
    df["prod_mean"] = df.groupby("PRODUCT_ID").log_units.transform("mean")   # product fixed effect
    top = df.COMMODITY_DESC.value_counts().index[:TOP_CATEGORIES]
    df["cat"] = df.COMMODITY_DESC.where(df.COMMODITY_DESC.isin(top), "OTHER")
    X = pd.get_dummies(df.cat, prefix="cat", drop_first=True).astype(float)  # heterogeneity
    return df, df.log_units.to_numpy(), df.depth.to_numpy(), X, df[W_COLS]


def naive_theta(df: pd.DataFrame) -> float:
    """Within-product slope of log units on depth, NO controls."""
    y = df.log_units - df.groupby("PRODUCT_ID").log_units.transform("mean")
    t = df.depth - df.groupby("PRODUCT_ID").depth.transform("mean")
    return float(LinearRegression().fit(t.to_frame(), y).coef_[0])


def fit_dml(Y, T, X, W):
    est = LinearDML(model_y=lgbm(), model_t=lgbm(), cv=5, random_state=42)
    est.fit(Y, T, X=X, W=W, cache_values=True)
    return est


def run():
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    df, Y, T, X, W = prepare(panel)

    with mlflow.start_run(run_name="uplift_dml"):
        theta_naive = naive_theta(df)

        est = fit_dml(Y, T, X, W)
        ate = float(est.ate(X))
        lo, hi = est.ate_interval(X, alpha=0.05)
        _, t_res, _, _ = est.residuals_
        resid_ratio = float(np.var(t_res) / np.var(T))   # overlap check: share of T variation left

        # Placebo: shuffle depth WITHIN each product -> true effect is zero by construction
        T_pl = df.groupby("PRODUCT_ID").depth.transform(
            lambda s: s.sample(frac=1, random_state=1).to_numpy()).to_numpy()
        theta_placebo = float(fit_dml(Y, T_pl, X, W).ate(X))

        # Per-category theta -> every product gets its category's theta
        cats = df[["cat"]].drop_duplicates().reset_index(drop=True)
        Xc = pd.get_dummies(cats.cat, prefix="cat").reindex(columns=X.columns, fill_value=0).astype(float)
        cats["theta"] = est.effect(Xc, T0=0, T1=1)
        theta = df[["PRODUCT_ID", "COMMODITY_DESC", "cat"]].drop_duplicates("PRODUCT_ID").merge(cats, on="cat")
        theta.to_csv(PROCESSED / "theta.csv", index=False)

        # Resume number [X]: share of promo-week units that would have sold anyway
        promo = df[df.depth >= PROMO_THRESHOLD]
        non_incremental = float(promo.baseline.sum() / promo.units.sum())

        metrics = {"theta_naive": theta_naive, "theta_dml_ate": ate, "dml_ci_low": float(lo),
                   "dml_ci_high": float(hi), "theta_placebo": theta_placebo,
                   "treatment_resid_var_ratio": resid_ratio, "non_incremental_share": non_incremental}
        mlflow.log_params({"n_rows": len(df), "cv_folds": 5, "W": ",".join(W_COLS)})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(PROCESSED / "theta.csv"))

    for k, v in metrics.items():
        print(f"{k:28s} {v: .3f}")
    return metrics


if __name__ == "__main__":
    run()

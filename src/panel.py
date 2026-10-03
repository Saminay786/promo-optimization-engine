"""Stage 0: raw transactions -> one row per (product, week) with treatment + confounders.

Run:  python -m src.panel
Out:  data/processed/panel.parquet
"""
import numpy as np
import pandas as pd

from src.config import BASELINE_WINDOW, PROCESSED, PROMO_THRESHOLD, RAW, TOP_N_PRODUCTS

TX_COLS = ["PRODUCT_ID", "STORE_ID", "WEEK_NO", "QUANTITY", "SALES_VALUE",
           "RETAIL_DISC", "COUPON_MATCH_DISC"]
# Not real merchandise (gasoline, gift cards, misc transactions): never promoted like groceries.
NON_MERCH_DEPTS = ["KIOSK-GAS", "MISC SALES TRAN", "MISC. TRANS."]


def load_transactions() -> pd.DataFrame:
    tx = pd.read_csv(RAW / "transaction_data.csv", usecols=TX_COLS)
    tx = tx[tx.QUANTITY > 0]
    prod = pd.read_csv(RAW / "product.csv", usecols=["PRODUCT_ID", "DEPARTMENT"])
    non_merch = prod.loc[prod.DEPARTMENT.isin(NON_MERCH_DEPTS), "PRODUCT_ID"]
    tx = tx[~tx.PRODUCT_ID.isin(non_merch)]
    keep = tx.groupby("PRODUCT_ID").SALES_VALUE.sum().nlargest(TOP_N_PRODUCTS).index
    return tx[tx.PRODUCT_ID.isin(keep)]


def weekly_sales(tx: pd.DataFrame) -> pd.DataFrame:
    # Discounts are stored as NEGATIVE numbers, so subtracting them adds them back.
    tx = tx.assign(shelf_value=tx.SALES_VALUE - tx.RETAIL_DISC - tx.COUPON_MATCH_DISC)
    g = (tx.groupby(["PRODUCT_ID", "WEEK_NO"])
           .agg(units=("QUANTITY", "sum"), shelf_value=("shelf_value", "sum"),
                retail_disc=("RETAIL_DISC", "sum"), n_stores=("STORE_ID", "nunique"))
           .reset_index())
    g["price"] = g.shelf_value / g.units                               # shelf price per unit
    g["depth"] = (-g.retail_disc / g.shelf_value).clip(0, 0.9)         # TREATMENT
    return g


def promo_support(products, weeks_df: pd.DataFrame) -> pd.DataFrame:
    """Share of selling stores with a display / mailer (the main CONFOUNDERS).
    causal_data.csv is huge, so read it in chunks and keep only our products."""
    parts = []
    for ch in pd.read_csv(RAW / "causal_data.csv", chunksize=2_000_000,
                          dtype={"display": str, "mailer": str}):
        ch = ch[ch.PRODUCT_ID.isin(products)]
        ch = ch.assign(disp=(ch.display != "0").astype(int), mail=(ch.mailer != "0").astype(int))
        parts.append(ch.groupby(["PRODUCT_ID", "WEEK_NO"])[["disp", "mail"]].sum())
    c = pd.concat(parts).groupby(level=[0, 1]).sum().reset_index()
    out = weeks_df.merge(c, on=["PRODUCT_ID", "WEEK_NO"], how="left").fillna({"disp": 0, "mail": 0})
    out["display_share"] = (out.disp / out.n_stores).clip(0, 1)
    out["mailer_share"] = (out.mail / out.n_stores).clip(0, 1)
    return out.drop(columns=["disp", "mail"])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    prod = pd.read_csv(RAW / "product.csv", usecols=["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC"])
    df = df.merge(prod, on="PRODUCT_ID", how="left").sort_values(["PRODUCT_ID", "WEEK_NO"])
    g = df.groupby("PRODUCT_ID")
    df["log_units"] = np.log1p(df.units)
    df["lag1"] = g.log_units.shift(1)
    df["lag2"] = g.log_units.shift(2)
    df["week_of_year"] = df.WEEK_NO % 52
    df["reg_price"] = g.price.transform("median")        # typical shelf price
    # How promotional is the rest of the category this week? (cannibalization pressure)
    cat = df.groupby(["COMMODITY_DESC", "WEEK_NO"]).depth.agg(["sum", "count"])
    df = df.join(cat, on=["COMMODITY_DESC", "WEEK_NO"])
    df["cat_promo"] = (df["sum"] - df.depth) / (df["count"] - 1).clip(lower=1)
    df = df.drop(columns=["sum", "count"])
    # Baseline = what would have sold WITHOUT a promo: rolling mean of past non-promo weeks.
    df["non_promo_units"] = df.units.where(df.depth < PROMO_THRESHOLD)
    df["baseline"] = df.groupby("PRODUCT_ID").non_promo_units.transform(
        lambda s: s.shift(1).rolling(BASELINE_WINDOW, min_periods=3).mean())
    df["baseline"] = df.groupby("PRODUCT_ID").baseline.ffill()
    return df.drop(columns="non_promo_units").dropna(subset=["lag1", "lag2", "baseline"])


def build_panel() -> pd.DataFrame:
    tx = load_transactions()
    wk = weekly_sales(tx)
    wk = promo_support(wk.PRODUCT_ID.unique(), wk)
    panel = add_features(wk)
    panel.to_parquet(PROCESSED / "panel.parquet", index=False)
    print(f"panel: {len(panel):,} rows | {panel.PRODUCT_ID.nunique()} products | "
          f"promo weeks: {(panel.depth >= PROMO_THRESHOLD).mean():.1%}")
    return panel


if __name__ == "__main__":
    build_panel()

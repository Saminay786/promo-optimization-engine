"""MMM stage 0: raw files -> one row per (category, week) with target, levers and a traffic control.

Run:  python -m src.mmm_data
Out:  data/processed/mmm_weekly.parquet
"""
import pandas as pd

from src.config import MMM_CATEGORIES, MMM_START_DATE, MMM_WEEKS, PROCESSED, RAW
from src.panel import NON_MERCH_DEPTS

LEVERS = ["disc_rate", "mailer_n", "display_n"]


def causal_stores() -> set:
    """Stores covered by causal_data.csv (display/mailer are only observed there)."""
    stores = set()
    for ch in pd.read_csv(RAW / "causal_data.csv", usecols=["STORE_ID"], chunksize=4_000_000):
        stores |= set(ch.STORE_ID.unique())
    return stores


def weekly_sales(stores: set, prod: pd.DataFrame) -> pd.DataFrame:
    tx = pd.read_csv(RAW / "transaction_data.csv",
                     usecols=["household_key", "PRODUCT_ID", "STORE_ID", "WEEK_NO", "QUANTITY",
                              "SALES_VALUE", "RETAIL_DISC", "COUPON_MATCH_DISC"])
    non_merch = prod.loc[prod.DEPARTMENT.isin(NON_MERCH_DEPTS), "PRODUCT_ID"]
    tx = tx[(tx.QUANTITY > 0) & tx.STORE_ID.isin(stores) & tx.WEEK_NO.between(*MMM_WEEKS)
            & ~tx.PRODUCT_ID.isin(non_merch)]
    # Traffic control: active households in the whole store that week (panel size, not promo driven)
    traffic = tx.groupby("WEEK_NO").household_key.nunique().rename("households")
    tx = tx.merge(prod[["PRODUCT_ID", "COMMODITY_DESC"]], on="PRODUCT_ID")
    tx = tx[tx.COMMODITY_DESC.isin(MMM_CATEGORIES)]
    tx = tx.assign(shelf=tx.SALES_VALUE - tx.RETAIL_DISC - tx.COUPON_MATCH_DISC)
    g = (tx.groupby(["COMMODITY_DESC", "WEEK_NO"])
           .agg(units=("QUANTITY", "sum"), sales=("SALES_VALUE", "sum"), shelf=("shelf", "sum"),
                disc=("RETAIL_DISC", lambda s: -s.sum()))
           .reset_index())
    g["disc_rate"] = g.disc / g.shelf                   # LEVER 1: share of shelf value given away
    g["shelf_price"] = g.shelf / g.units                # to convert units back to $ later
    return g.merge(traffic, on="WEEK_NO")


def weekly_support(prod: pd.DataFrame) -> pd.DataFrame:
    """Mailer / display placements (product-store slots) per category-week.
    causal_data.csv ONLY lists promoted product-store-weeks (no row has display=0 AND mailer=0),
    so a share of its rows measures the promo MIX, not the amount. Use counts instead."""
    cat_of = prod.set_index("PRODUCT_ID").COMMODITY_DESC
    parts = []
    for ch in pd.read_csv(RAW / "causal_data.csv", chunksize=4_000_000, dtype={"display": str, "mailer": str}):
        ch = ch[ch.WEEK_NO.between(*MMM_WEEKS)]
        ch = ch.assign(COMMODITY_DESC=ch.PRODUCT_ID.map(cat_of))
        ch = ch[ch.COMMODITY_DESC.isin(MMM_CATEGORIES)]
        ch = ch.assign(mailer_n=(ch.mailer != "0").astype(int),     # LEVER 2: mailer placements
                       display_n=(ch.display != "0").astype(int))   # LEVER 3: display placements
        parts.append(ch.groupby(["COMMODITY_DESC", "WEEK_NO"])[["mailer_n", "display_n"]].sum())
    return pd.concat(parts).groupby(level=[0, 1]).sum().reset_index()


def build_mmm_data() -> pd.DataFrame:
    prod = pd.read_csv(RAW / "product.csv", usecols=["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC"])
    df = weekly_sales(causal_stores(), prod).merge(weekly_support(prod), on=["COMMODITY_DESC", "WEEK_NO"], how="left")
    df[["mailer_n", "display_n"]] = df[["mailer_n", "display_n"]].fillna(0)
    df["date"] = pd.Timestamp(MMM_START_DATE) + pd.to_timedelta((df.WEEK_NO - 1) * 7, unit="D")
    df = df.rename(columns={"COMMODITY_DESC": "category"}).sort_values(["category", "WEEK_NO"])
    df.to_parquet(PROCESSED / "mmm_weekly.parquet", index=False)
    print(f"mmm data: {len(df):,} rows | {df.category.nunique()} categories | weeks {df.WEEK_NO.min()}-{df.WEEK_NO.max()}")
    return df


if __name__ == "__main__":
    build_mmm_data()

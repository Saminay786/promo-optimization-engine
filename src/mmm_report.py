"""MMM stage 2: contribution decomposition and ROI per lever (with uncertainty).

Mailer/display have no cost data, so instead of an assumed-cost ROI we report the BREAK-EVEN cost per
placement (ROI at any cost c = break-even / c). Every number is computed per posterior draw, then summarised as mean and 94% interval (3%-97%).
Levers are measured ABOVE each category's quietest week (see src/mmm.py), so "contribution" means
"extra units vs the least-promoted week".

Run:  python -m src.mmm_report
Out:  data/processed/mmm_contributions.csv, mmm_roi.csv, mmm_reallocation.csv, mmm_vs_dml.csv
"""
import numpy as np
import pandas as pd
import xarray as xr
from pymc_marketing.mmm import MMM

from src.config import MARGIN_RATE, MMM_DISPLAY_COSTS, MMM_REALLOC_SHARE, PROCESSED, VENDOR_FUNDING
from src.mmm import MODEL_PATH


def summarise(draws: xr.DataArray) -> dict:
    x = np.asarray(draws).ravel()
    return {"mean": float(x.mean()), "lo": float(np.quantile(x, 0.03)), "hi": float(np.quantile(x, 0.97))}


def contribution_units(mmm: MMM) -> xr.DataArray:
    """Extra units per (chain, draw, date, category, channel), on the original unit scale."""
    post = mmm.idata["posterior"]
    return post["channel_contribution"] * mmm.get_scales_as_xarray()["target_scale"]


def weekly_inputs() -> pd.DataFrame:
    """Raw weekly data plus each lever measured above its category's minimum (same as the model saw)."""
    df = pd.read_parquet(PROCESSED / "mmm_weekly.parquet")
    for lever in ["disc_rate", "mailer_n", "display_n"]:
        df[f"{lever}_above"] = df[lever] - df.groupby("category")[lever].transform("min")
    return df


def decomposition(contrib: xr.DataArray, df: pd.DataFrame) -> pd.DataFrame:
    total = contrib.sum("date")                                   # (chain, draw, category, channel)
    units = df.groupby("category").units.sum()
    rows = []
    for cat in total.category.values:
        for ch in total.channel.values:
            s = summarise(total.sel(category=cat, channel=ch))
            rows.append({"category": cat, "lever": ch, "extra_units": s["mean"], "lo": s["lo"], "hi": s["hi"],
                         "share_of_units": s["mean"] / units[cat]})
    return pd.DataFrame(rows)


def roi_table(contrib: xr.DataArray, df: pd.DataFrame) -> pd.DataFrame:
    total = contrib.sum("date")
    g = df.groupby("category")
    price = g.apply(lambda d: d.shelf.sum() / d.units.sum())                  # avg shelf price per unit
    extra_disc_spend = g.apply(lambda d: (d.disc_rate_above * d.shelf).sum()) * (1 - VENDOR_FUNDING)
    placements = {lv: g[f"{lv}_above"].sum() for lv in ["mailer_n", "display_n"]}
    rows = []
    for cat in total.category.values:
        margin_draws = total.sel(category=cat) * price[cat] * MARGIN_RATE     # extra gross margin $ per draw
        d = summarise(margin_draws.sel(channel="disc_rate") / extra_disc_spend[cat])
        rows.append({"category": cat, "lever": "disc_rate", "metric": "ROI ($ margin per $ retailer discount)",
                     "mean": d["mean"], "lo": d["lo"], "hi": d["hi"]})
        for lv in ["mailer_n", "display_n"]:
            per_k = summarise(total.sel(category=cat, channel=lv) / placements[lv][cat] * 1000)
            rows.append({"category": cat, "lever": lv, "metric": "extra units per 1,000 placements", **per_k})
            be = summarise(margin_draws.sel(channel=lv) / placements[lv][cat])
            rows.append({"category": cat, "lever": lv, "metric": "break-even $ per placement", **be})
    return pd.DataFrame(rows)


def response(mmm: MMM, cat: str, lever: str, level: float) -> xr.DataArray:
    """Weekly extra units (per draw) from holding a lever at `level` (above its minimum) every week.
    Adstock weights are normalised to sum to 1, so in steady state the carried-over level equals `level`."""
    post, sc = mmm.idata["posterior"], mmm.get_scales_as_xarray()
    x = level / float(sc["channel_scale"].sel(category=cat, channel=lever))
    lam = post["saturation_lam"].sel(channel=lever)
    beta = post["saturation_beta"].sel(category=cat, channel=lever)
    return beta * (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x)) * float(sc["target_scale"].sel(category=cat))


def reallocation(mmm: MMM, df: pd.DataFrame) -> pd.DataFrame:
    """Scenario: move MMM_REALLOC_SHARE of each category's extra discount spend into display placements.
    Total spend is unchanged; evaluated at the average week, for each assumed display cost."""
    rows = []
    for cat, d in df.groupby("category"):
        rate, disp = d.disc_rate_above.mean(), d.display_n_above.mean()
        shelf_week, price = d.shelf.mean(), d.shelf.sum() / d.units.sum()
        moved = MMM_REALLOC_SHARE * rate * shelf_week * (1 - VENDOR_FUNDING)      # $ per week moved
        lost = response(mmm, cat, "disc_rate", (1 - MMM_REALLOC_SHARE) * rate) - response(mmm, cat, "disc_rate", rate)
        for cost in MMM_DISPLAY_COSTS:
            extra = moved / cost                                                    # extra placements per week
            gained = response(mmm, cat, "display_n", disp + extra) - response(mmm, cat, "display_n", disp)
            units = lost + gained
            s, m = summarise(units), summarise(units * price * MARGIN_RATE)
            rows.append({"category": cat, "display_cost": cost, "moved_per_week": moved,
                         "extra_placements": extra, "beyond_observed": disp + extra > d.display_n_above.max(),
                         "d_units_mean": s["mean"], "d_units_lo": s["lo"], "d_units_hi": s["hi"],
                         "d_margin_mean": m["mean"], "d_margin_lo": m["lo"], "d_margin_hi": m["hi"],
                         "p_improve": float((np.asarray(units) > 0).mean())})
    return pd.DataFrame(rows)


def compare_dml(mmm: MMM, df: pd.DataFrame, step: float = 0.01) -> pd.DataFrame:
    """Put MMM and DML on the same scale: semi-elasticity d log(units) / d discount depth.
    DML theta is PRODUCT level (includes sales stolen from sister products); the MMM is CATEGORY level
    (stealing nets out), so cannibalisation predicts theta_mmm < theta_dml."""
    dml = pd.read_csv(PROCESSED / "theta.csv").groupby("cat").theta.first()
    rows = []
    for cat, d in df.groupby("category"):
        if cat not in dml.index:                    # DML pooled this category into "OTHER"
            continue
        rate = d.disc_rate_above.mean()
        slope = (response(mmm, cat, "disc_rate", rate + step) - response(mmm, cat, "disc_rate", rate)) / step
        s = summarise(slope / d.units.mean())
        rows.append({"category": cat, "theta_dml": float(dml[cat]), "theta_mmm": s["mean"],
                     "mmm_lo": s["lo"], "mmm_hi": s["hi"], "ratio_mmm_to_dml": s["mean"] / float(dml[cat])})
    return pd.DataFrame(rows)


def run():
    mmm = MMM.load(str(MODEL_PATH))
    df = weekly_inputs()
    contrib = contribution_units(mmm)
    check = float(contrib.sum(("date", "category", "channel")).mean())
    stored = float(mmm.idata["posterior"]["total_media_contribution_original_scale"].mean())
    print(f"check: my total {check:,.0f} vs model's stored total {stored:,.0f}")
    dec = decomposition(contrib, df)
    roi = roi_table(contrib, df)
    realloc = reallocation(mmm, df)
    dec.to_csv(PROCESSED / "mmm_contributions.csv", index=False)
    roi.to_csv(PROCESSED / "mmm_roi.csv", index=False)
    realloc.to_csv(PROCESSED / "mmm_reallocation.csv", index=False)
    vs = compare_dml(mmm, df)
    vs.to_csv(PROCESSED / "mmm_vs_dml.csv", index=False)
    pd.set_option("display.width", 200)
    print(dec.round(3).to_string(index=False))
    print(roi.round(3).to_string(index=False))
    print(realloc.round(3).to_string(index=False))
    print(vs.round(3).to_string(index=False))
    return dec, roi, realloc, vs


if __name__ == "__main__":
    run()

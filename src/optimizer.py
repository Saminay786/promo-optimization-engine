"""Stage 2: choose which product gets which discount in which week.

Model (0-1 integer program, solved with OR-Tools CP-SAT):
  x[i,k,t] = 1 if product i runs discount depth k in week t
  max  sum u[i,k,t] * x[i,k,t]                       (incremental margin)
  s.t. sum_k x[i,k,t] <= 1                for all i,t (one depth per product-week)
       sum c[i,k,t] * x[i,k,t] <= B                   (promo budget)
       sum_{i,k} x[i,k,t] <= S             for all t  (promo slots per week)
       sum_{k,t} x[i,k,t] <= F             for all i  (max promo weeks per product)
       sum_{i in g,k} x[i,k,t] <= M        for all g,t (category cap: cannibalization)
       sum_k x[i,k,t] + x[i,k,t+1] <= 1    for all i,t (no back-to-back: pull-forward)

Run:  python -m src.optimizer   -> builds data/processed/params.csv and prints MIP vs heuristic
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from src.config import DEPTHS, MARGIN_RATE, PROCESSED, VENDOR_FUNDING

SCALE = 100  # CP-SAT only accepts integers -> work in cents


@dataclass
class Rules:
    budget: float
    slots_per_week: int = 10
    weeks: int = 8
    max_promos_per_product: int = 2
    category_cap: int = 2


def build_params() -> pd.DataFrame:
    """One row per product: latest baseline, shelf price, category, theta."""
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    theta = pd.read_csv(PROCESSED / "theta.csv")[["PRODUCT_ID", "theta"]]
    last = (panel.sort_values("WEEK_NO").groupby("PRODUCT_ID").tail(1)
            [["PRODUCT_ID", "COMMODITY_DESC", "baseline", "reg_price"]])
    params = last.merge(theta, on="PRODUCT_ID").rename(columns={"reg_price": "price"})
    params.to_csv(PROCESSED / "params.csv", index=False)
    return params


def candidates(params: pd.DataFrame, weeks: int) -> pd.DataFrame:
    """Every (product, depth, week) option with its incremental margin u and cost c."""
    rows = []
    for r in params.itertuples():
        for d in DEPTHS:
            extra = r.baseline * (np.exp(r.theta * d) - 1)         # incremental units q
            unit_margin = MARGIN_RATE * r.price
            own_disc = r.price * d * (1 - VENDOR_FUNDING)            # discount the retailer pays per unit
            # gain on extra units  MINUS  discount given away on units that would have sold anyway
            u = extra * (unit_margin - own_disc) - r.baseline * own_disc
            c = (r.baseline + extra) * own_disc                      # retailer's promo spend
            for t in range(weeks):
                rows.append((r.PRODUCT_ID, r.COMMODITY_DESC, d, t, u, c))
    return pd.DataFrame(rows, columns=["product", "category", "depth", "week", "u", "c"])


def solve(params: pd.DataFrame, rules: Rules, time_limit_s: float = 20.0) -> pd.DataFrame:
    cand = candidates(params, rules.weeks)
    cand = cand[cand.u > 0].reset_index(drop=True)          # never pick a money-losing promo
    m = cp_model.CpModel()
    x = [m.new_bool_var(f"x{j}") for j in range(len(cand))]
    u = (cand.u * SCALE).round().astype(int).tolist()
    c = (cand.c * SCALE).round().astype(int).tolist()

    def total(idx):
        return sum(x[j] for j in idx)

    for idx in cand.groupby(["product", "week"]).groups.values():
        m.add(total(idx) <= 1)                                           # one depth
    m.add(sum(c[j] * x[j] for j in range(len(x))) <= int(rules.budget * SCALE))  # budget
    for idx in cand.groupby("week").groups.values():
        m.add(total(idx) <= rules.slots_per_week)                        # weekly slots
    for idx in cand.groupby("product").groups.values():
        m.add(total(idx) <= rules.max_promos_per_product)                # frequency
    for idx in cand.groupby(["category", "week"]).groups.values():
        m.add(total(idx) <= rules.category_cap)                          # cannibalization
    by_pw = cand.groupby(["product", "week"]).groups
    for (p, t), idx in by_pw.items():                                    # no back-to-back
        nxt = by_pw.get((p, t + 1))
        if nxt is not None:
            m.add(total(idx) + total(nxt) <= 1)

    m.maximize(sum(u[j] * x[j] for j in range(len(x))))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    status = solver.solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"No solution: {solver.status_name(status)}")
    chosen = [j for j in range(len(x)) if solver.value(x[j])]
    return cand.loc[chosen].sort_values(["week", "product"]).reset_index(drop=True)


def heuristic(params: pd.DataFrame, rules: Rules) -> pd.DataFrame:
    """What a planner might do by hand: deepest discount on the biggest sellers until money runs out.
    Respects the same rules, so the comparison with the MIP is fair."""
    cand = candidates(params, rules.weeks)
    cand = cand[cand.depth == max(DEPTHS)]
    ranked = params.assign(rev=params.baseline * params.price).sort_values("rev", ascending=False)
    spent, picks, used = 0.0, [], {}
    for t in range(rules.weeks):
        per_cat, slots = {}, 0
        for p in ranked.PRODUCT_ID:
            row = cand[(cand["product"] == p) & (cand.week == t)].iloc[0]
            weeks_used = used.get(p, [])
            if (slots >= rules.slots_per_week or len(weeks_used) >= rules.max_promos_per_product
                    or (t - 1) in weeks_used or per_cat.get(row.category, 0) >= rules.category_cap
                    or spent + row.c > rules.budget):
                continue
            picks.append(row); spent += row.c; slots += 1
            used.setdefault(p, []).append(t); per_cat[row.category] = per_cat.get(row.category, 0) + 1
    return pd.DataFrame(picks, columns=cand.columns).reset_index(drop=True)


def compare(params: pd.DataFrame, rules: Rules) -> dict:
    mip, heur = solve(params, rules), heuristic(params, rules)
    mip_u, heur_u = mip.u.sum(), heur.u.sum()
    return {"mip_margin": round(float(mip_u), 2), "heuristic_margin": round(float(heur_u), 2),
            "mip_spend": round(float(mip.c.sum()), 2), "heuristic_spend": round(float(heur.c.sum()), 2),
            "mip_minus_heuristic": round(float(mip_u - heur_u), 2),
            "uplift_vs_heuristic_pct": round(float(100 * (mip_u - heur_u) / heur_u), 1) if heur_u > 0 else None,
            "n_promos_mip": len(mip), "n_promos_heuristic": len(heur),
            "plan": mip}


if __name__ == "__main__":
    params = build_params()
    out = compare(params, Rules(budget=5000))
    plan = out.pop("plan")
    print(plan.head(15).to_string(index=False))
    print(out)

import numpy as np
import pytest

from src.config import MARGIN_RATE, VENDOR_FUNDING
from src.optimizer import Rules, candidates, compare, heuristic, solve

RULES = Rules(budget=300, slots_per_week=2, weeks=4, max_promos_per_product=2, category_cap=1)


def test_margin_math_matches_hand_calculation(params):
    row = candidates(params.iloc[[0]], weeks=1).query("depth == 0.2").iloc[0]
    b, p, th, d = 50, 4.0, 2.5, 0.2
    extra = b * (np.exp(th * d) - 1)
    own = p * d * (1 - VENDOR_FUNDING)
    assert row.u == pytest.approx(extra * (MARGIN_RATE * p - own) - b * own)
    assert row.c == pytest.approx((b + extra) * own)


def test_solution_respects_every_rule(params):
    plan = solve(params, RULES)
    assert plan.c.sum() <= RULES.budget + 1e-6
    assert plan.groupby("week").size().max() <= RULES.slots_per_week
    assert plan.groupby("product").size().max() <= RULES.max_promos_per_product
    assert plan.groupby(["category", "week"]).size().max() <= RULES.category_cap
    assert not plan.duplicated(["product", "week"]).any()
    for _, weeks in plan.groupby("product").week:
        assert not (np.diff(sorted(weeks)) == 1).any()       # no back-to-back weeks


def test_mip_beats_or_ties_heuristic(params):
    assert solve(params, RULES).u.sum() >= heuristic(params, RULES).u.sum() - 1e-6


def test_zero_budget_means_no_promos(params):
    assert solve(params, Rules(budget=0.01, weeks=2)).empty


def test_compare_handles_unaffordable_budget(params):
    out = compare(params, Rules(budget=0.01, weeks=2))       # heuristic picks nothing -> must not crash
    assert out["heuristic_margin"] == 0 and out["plan"].empty

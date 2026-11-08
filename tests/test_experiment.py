import numpy as np
import pandas as pd
import pytest

from src.config import MARGIN_RATE
from src.experiment import cuped, eligible, mde, power_table, predicted_lift, stores_needed


@pytest.fixture
def store_weeks():
    """Fake store x week margins: stores differ a lot in size (that's what CUPED exploits) + weekly noise."""
    rng = np.random.default_rng(1)
    size = rng.lognormal(mean=3.5, sigma=0.5, size=60)
    data = size[:, None] * rng.lognormal(0, 0.3, size=(60, 40))
    return pd.DataFrame(data, columns=range(20, 60))


def test_aa_false_positive_rate_is_near_alpha(store_weeks):
    fpr = power_table(store_weeks, test_weeks=[4], lifts=[0.0], n_sims=600).power
    assert (fpr.between(0.02, 0.09)).all()          # valid analysis: ~5% false alarms when nothing changed


def test_cuped_keeps_mean_and_cuts_variance():
    rng = np.random.default_rng(2)
    x = rng.lognormal(3, 0.6, 500)
    y = x * rng.lognormal(0, 0.2, 500)
    adj = cuped(y, x)
    assert adj.mean() == pytest.approx(y.mean())
    assert adj.var() < 0.5 * y.var()


def test_power_rises_with_lift_and_cuped_helps(store_weeks):
    p = power_table(store_weeks, test_weeks=[4], lifts=[0.05, 0.15], n_sims=300).set_index(["method", "lift"]).power
    assert p["CUPED", 0.15] > p["CUPED", 0.05]
    assert p["CUPED", 0.05] > p["difference in means", 0.05]


def test_mde_interpolates_between_simulated_lifts():
    power = pd.DataFrame({"test_weeks": 8, "method": "CUPED", "lift": [0.0, 0.02, 0.04], "power": [0.05, 0.6, 1.0]})
    assert mde(power).mde.iloc[0] == pytest.approx(0.02 + 0.2 / 0.4 * 0.02)


def test_eligibility_drops_small_stores(store_weeks):
    assert len(eligible(store_weeks, min_weekly=1e9)) == 0
    assert len(eligible(store_weeks, min_weekly=0)) == len(store_weeks)


def test_predicted_lift_and_stores_needed_math(params):
    base = MARGIN_RATE * (params.baseline * params.price).sum() * 8
    assert predicted_lift(params, plan_margin=base * 0.03, weeks=8) == pytest.approx(0.03)
    assert stores_needed(100, mde_value=0.06, target_lift=0.03) == 400    # half the effect -> 4x the stores

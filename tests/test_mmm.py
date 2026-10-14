"""MMM extension tests. Skipped unless requirements-mmm.txt is installed (pymc-marketing needs Python >= 3.12)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pymc_marketing")   # must come first: xarray etc. are only installed with the MMM extras
import xarray as xr

from src import mmm as mmm_mod
from src.mmm_report import response, summarise


@pytest.fixture
def fake_model():
    """Minimal stand-in exposing what response() reads: posterior draws and the model's scales."""
    lam = xr.DataArray(np.array([[[0.5, 1.0]]]), dims=("chain", "draw", "channel"),
                       coords={"channel": ["disc_rate", "display_n"]})
    beta = xr.DataArray(np.array([[[[0.2, 0.4]]]]), dims=("chain", "draw", "category", "channel"),
                        coords={"category": ["A"], "channel": ["disc_rate", "display_n"]})
    scales = {"channel_scale": xr.DataArray([[0.5, 100.0]], dims=("category", "channel"),
                                            coords={"category": ["A"], "channel": ["disc_rate", "display_n"]}),
              "target_scale": xr.DataArray([1000.0], dims=("category",), coords={"category": ["A"]})}
    return SimpleNamespace(idata={"posterior": {"saturation_lam": lam, "saturation_beta": beta}},
                           get_scales_as_xarray=lambda: scales)


def test_response_matches_logistic_formula(fake_model):
    x = 50 / 100.0                                                    # level / channel scale
    expected = 0.4 * (1 - np.exp(-1.0 * x)) / (1 + np.exp(-1.0 * x)) * 1000
    assert response(fake_model, "A", "display_n", 50).item() == pytest.approx(expected)


def test_response_is_zero_at_zero_and_has_diminishing_returns(fake_model):
    r = [response(fake_model, "A", "display_n", v).item() for v in (0, 100, 200, 300)]
    assert r[0] == 0
    gains = np.diff(r)
    assert (gains > 0).all() and (np.diff(gains) < 0).all()          # increasing, but each step adds less


def test_summarise_interval_brackets_mean():
    s = summarise(xr.DataArray(np.random.default_rng(0).normal(5, 1, (4, 500)), dims=("chain", "draw")))
    assert s["lo"] < s["mean"] < s["hi"] and s["mean"] == pytest.approx(5, abs=0.1)


def test_load_data_shifts_levers_and_standardises_control(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"date": list(pd.date_range("2020-01-05", periods=6, freq="W")) * 2,
                       "category": ["A"] * 6 + ["B"] * 6, "units": rng.integers(50, 100, 12),
                       "disc_rate": rng.uniform(0.1, 0.3, 12), "mailer_n": rng.integers(100, 900, 12),
                       "display_n": rng.integers(10, 90, 12), "households": rng.integers(1200, 1300, 12)})
    df.to_parquet(tmp_path / "mmm_weekly.parquet")
    monkeypatch.setattr(mmm_mod, "PROCESSED", tmp_path)
    X, y = mmm_mod.load_data()
    for lever in ["disc_rate", "mailer_n", "display_n"]:
        assert (X.groupby("category")[lever].min() == 0).all()       # quietest week of each category = 0
    assert X.households_z.mean() == pytest.approx(0, abs=1e-9) and X.households_z.std() == pytest.approx(1)
    assert len(y) == 12

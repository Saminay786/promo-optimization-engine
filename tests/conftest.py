import pandas as pd
import pytest


@pytest.fixture
def params():
    """Tiny hand-made catalogue so tests run in CI without the real data."""
    return pd.DataFrame({
        "PRODUCT_ID": [1, 2, 3, 4, 5, 6],
        "COMMODITY_DESC": ["A", "A", "A", "B", "B", "C"],
        "baseline": [50, 40, 30, 60, 20, 45],
        "price": [4.0, 5.0, 3.0, 6.0, 2.0, 4.5],
        "theta": [2.5, 2.8, 2.2, 2.6, 1.0, 2.9],
    })

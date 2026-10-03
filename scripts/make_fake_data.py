"""Creates SMALL fake CSVs with the exact dunnhumby column names, so you can run the
whole pipeline before downloading the real data. The true discount effect is known
(TRUE_THETA), so you can check that DML recovers it while the naive model is biased.

Run:  python scripts/make_fake_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)
N_PROD, N_STORE, N_WEEK, N_CAT = 160, 15, 102, 12
TRUE_THETA = rng.uniform(1.5, 3.0, N_CAT)   # effect of +1.0 depth on log(units), per category

prod = pd.DataFrame({
    "PRODUCT_ID": np.arange(1000, 1000 + N_PROD),
    "DEPARTMENT": "GROCERY",
    "COMMODITY_DESC": [f"CAT_{c:02d}" for c in rng.integers(0, N_CAT, N_PROD)],
})
cat_idx = prod.COMMODITY_DESC.str[-2:].astype(int).to_numpy()
base = rng.uniform(0.5, 3.0, N_PROD)
price = rng.uniform(1.0, 8.0, N_PROD)

tx_rows, causal_rows = [], []
for w in range(1, N_WEEK + 1):
    season = 0.2 * np.sin(2 * np.pi * w / 52)
    promo = rng.random(N_PROD) < 0.15
    depth = np.where(promo, rng.choice([0.1, 0.2, 0.3], N_PROD), 0.0)
    for s in range(N_STORE):
        # Displays mostly happen DURING promos -> confounding (naive model over-credits discount)
        display = rng.random(N_PROD) < np.where(promo, 0.7, 0.05)
        mailer = rng.random(N_PROD) < np.where(promo, 0.5, 0.02)
        lam = base * np.exp(TRUE_THETA[cat_idx] * depth + 0.5 * display + 0.3 * mailer + season)
        q = rng.poisson(lam)
        for i in np.nonzero(q)[0]:
            full = q[i] * price[i]
            tx_rows.append((1000 + i, 300 + s, w, q[i], full * (1 - depth[i]), -full * depth[i], 0.0))
        for i in range(N_PROD):
            causal_rows.append((1000 + i, 300 + s, w, "1" if display[i] else "0", "A" if mailer[i] else "0"))

pd.DataFrame(tx_rows, columns=["PRODUCT_ID", "STORE_ID", "WEEK_NO", "QUANTITY", "SALES_VALUE",
                               "RETAIL_DISC", "COUPON_MATCH_DISC"]).to_csv(OUT / "transaction_data.csv", index=False)
pd.DataFrame(causal_rows, columns=["PRODUCT_ID", "STORE_ID", "WEEK_NO", "display", "mailer"]).to_csv(
    OUT / "causal_data.csv", index=False)
prod.to_csv(OUT / "product.csv", index=False)
pd.DataFrame({"COMMODITY_DESC": [f"CAT_{c:02d}" for c in range(N_CAT)], "true_theta": TRUE_THETA}).to_csv(
    OUT / "_true_theta_FAKE.csv", index=False)
print("fake data written to data/raw/")

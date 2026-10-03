"""Central config. Change numbers here, not inside the code."""
from pathlib import Path

RAW = Path("data/raw")              # put the 3 dunnhumby CSVs here
PROCESSED = Path("data/processed")  # pipeline outputs land here
PROCESSED.mkdir(parents=True, exist_ok=True)

TOP_N_PRODUCTS = 150      # keep top-N products by revenue (keeps data small)
PROMO_THRESHOLD = 0.05    # discount depth >= 5% counts as a promo week
BASELINE_WINDOW = 8       # baseline = mean units of last 8 NON-promo weeks
TOP_CATEGORIES = 10       # uplift is estimated separately for top-10 categories

DEPTHS = [0.10, 0.20, 0.30]   # discount options the optimizer can choose
MARGIN_RATE = 0.30            # ASSUMPTION: no cost data, margin = 30% of shelf price
VENDOR_FUNDING = 0.50         # ASSUMPTION: share of the discount paid by the supplier (common in retail promos)

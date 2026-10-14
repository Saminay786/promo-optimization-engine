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

# --- Optional extension: Bayesian MMM (src/mmm_data.py, src/mmm.py) ---
MMM_WEEKS = (20, 101)         # before wk 20 households were still joining the panel; causal data ends at wk 101
MMM_CATEGORIES = ["SOFT DRINKS", "BAG SNACKS", "BAKED BREAD/BUNS/ROLLS", "CHEESE", "COLD CEREAL",
                  "FROZEN PIZZA", "FRZN MEAT/MEAT DINNERS", "DELI MEATS"]   # categories with real weekly lever variation
MMM_START_DATE = "2019-01-06"  # dunnhumby has week numbers only; week 1 is mapped to this (arbitrary) Sunday
MMM_REALLOC_SHARE = 0.20      # scenario: move 20% of the extra discount spend into displays
MMM_DISPLAY_COSTS = [0.01, 0.03, 0.05, 0.10]   # ASSUMPTION grid: $ per display placement (no cost data)

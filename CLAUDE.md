# Context for Claude Code
Portfolio project: promo uplift (Double ML) + promo calendar optimisation (CP-SAT IP) + FastAPI/Docker/CI.
Data: dunnhumby "The Complete Journey" CSVs in data/raw (transaction_data, causal_data, product). Not committed.
Run order: src.panel -> src.uplift -> src.optimizer -> src.api. Assumptions live in src/config.py.
Rules: keep functions small and typed; every optimizer rule must have a pytest check; run `ruff check` and `pytest -q` before committing.
The owner is learning: when asked to change code, explain the change in 2-3 lines.

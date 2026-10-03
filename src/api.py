"""Stage 3: serve the optimizer as an API.

Run locally:  uvicorn src.api:app --reload      then open http://127.0.0.1:8000/docs
"""
from functools import lru_cache
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from src.config import PROCESSED
from src.optimizer import Rules, compare

app = FastAPI(title="Promo Optimization Engine")


class OptimizeRequest(BaseModel):
    budget: float = Field(gt=0, description="Retailer promo spend allowed over the horizon")
    slots_per_week: int = Field(10, ge=1, le=50)
    weeks: int = Field(8, ge=1, le=13)
    max_promos_per_product: int = Field(2, ge=1)
    category_cap: int = Field(2, ge=1)


class PromoLine(BaseModel):
    product: int
    category: str
    depth: float
    week: int
    incremental_margin: float
    spend: float


class OptimizeResponse(BaseModel):
    expected_incremental_margin: float
    total_spend: float
    heuristic_margin: float
    mip_minus_heuristic: float
    plan: list[PromoLine]


@lru_cache
def get_params() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "params.csv")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest, params: Annotated[pd.DataFrame, Depends(get_params)]):
    out = compare(params, Rules(**req.model_dump()))
    plan = out["plan"].rename(columns={"u": "incremental_margin", "c": "spend"})
    return OptimizeResponse(
        expected_incremental_margin=out["mip_margin"], total_spend=out["mip_spend"],
        heuristic_margin=out["heuristic_margin"], mip_minus_heuristic=out["mip_minus_heuristic"],
        plan=plan.round(2).to_dict(orient="records"))

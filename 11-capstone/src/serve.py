from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from .predictor import predict as predict_any  # type: ignore
except Exception:  # pragma: no cover
    from src.predictor import predict as predict_any  # type: ignore


class AnimeRequest(BaseModel):
    type: Optional[str] = None
    season: Optional[str] = None
    year: Optional[int] = None
    episodes: Optional[int] = None
    source: Optional[str] = None
    rating: Optional[str] = None
    status: Optional[str] = None

    genres: List[str] = Field(default_factory=list)
    themes: List[str] = Field(default_factory=list)
    demographics: List[str] = Field(default_factory=list)
    studios: List[str] = Field(default_factory=list)


class PredictionResponse(BaseModel):
    hit_probability: float
    hit: bool
    backend: Optional[str] = None
    threshold: Optional[float] = None


app = FastAPI(title="Anime Hit Prediction API", version="1.0")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: AnimeRequest) -> PredictionResponse:
    try:
        payload = req.model_dump()
    except Exception:
        payload = req.dict()

    try:
        out = predict_any(payload)
        return PredictionResponse(**out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    environment: str


class ServiceStatus(BaseModel):
    enabled: bool
    available: bool
    loaded: bool
    detail: str | None = None


class ModelsInfoResponse(BaseModel):
    sentiment: ServiceStatus
    classifier: ServiceStatus


class SentimentRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Lista de textos a analizar")
    add_probabilities: bool = False


class SentimentItem(BaseModel):
    text: str
    label: str
    score: float
    probabilities: dict[str, float] | None = None


class SentimentResponse(BaseModel):
    items: list[SentimentItem]
    model_name: str


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Lista de textos para el clasificador")
    return_probabilities: bool = False


class PredictionItem(BaseModel):
    text: str
    prediction: Any
    probabilities: dict[str, float] | None = None


class PredictResponse(BaseModel):
    items: list[PredictionItem]
    model_loaded_from: str | None = None


class AnalyzeRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    add_sentiment_probabilities: bool = False
    add_prediction_probabilities: bool = False
    require_prediction: bool = False


class AnalyzeItem(BaseModel):
    text: str
    sentiment: SentimentItem
    prediction: PredictionItem | None = None


class AnalyzeResponse(BaseModel):
    items: list[AnalyzeItem]

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from src.api.config import ApiSettings, load_settings
from src.api.schemas import (
    AnalyzeItem,
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
    ModelsInfoResponse,
    PredictRequest,
    PredictResponse,
    PredictionItem,
    SentimentItem,
    SentimentRequest,
    SentimentResponse,
    ServiceStatus,
)
from src.api.services import (
    ClassifierArtifacts,
    ClassifierService,
    SentimentService,
    ServiceUnavailableError,
)
from src.modeling.sentiment_analisis import InferenceConfig


def _build_services(settings: ApiSettings) -> tuple[SentimentService, ClassifierService]:
    sentiment_cfg = InferenceConfig(
        model_name=settings.sentiment_model_name,
        batch_size=settings.sentiment_batch_size,
        max_length=settings.sentiment_max_length,
        stride=settings.sentiment_stride,
        device=settings.sentiment_device,
        num_workers=settings.sentiment_num_workers,
        progress=False,
        add_proba_cols=False,
    )
    sentiment = SentimentService(sentiment_cfg)
    classifier = ClassifierService(
        ClassifierArtifacts(
            model_path=settings.classifier_model_path,
            vectorizer_path=settings.classifier_vectorizer_path,
        ),
        enabled=settings.classifier_enabled,
    )
    return sentiment, classifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    sentiment_service, classifier_service = _build_services(settings)

    app.state.settings = settings
    app.state.sentiment_service = sentiment_service
    app.state.classifier_service = classifier_service

    if settings.sentiment_eager_load:
        sentiment_service.load()
    if settings.classifier_eager_load and settings.classifier_enabled:
        try:
            classifier_service.load()
        except Exception:
            # No derribar la API si el clasificador aún no existe.
            pass

    yield


_BOOT_SETTINGS = load_settings()

app = FastAPI(
    title=_BOOT_SETTINGS.app_name,
    description="API para análisis de sentimiento y predicción con modelos NLP.",
    version=_BOOT_SETTINGS.app_version,
    lifespan=lifespan,
)


@app.exception_handler(ServiceUnavailableError)
async def _service_unavailable_handler(_: Request, exc: ServiceUnavailableError) -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": str(exc)})


def _get_settings() -> ApiSettings:
    return cast(ApiSettings, app.state.settings)


def _get_sentiment_service() -> SentimentService:
    return cast(SentimentService, app.state.sentiment_service)


def _get_classifier_service() -> ClassifierService:
    return cast(ClassifierService, app.state.classifier_service)


def _service_status_from_metadata(meta: dict, enabled_default: bool = True) -> ServiceStatus:
    return ServiceStatus(
        enabled=bool(meta.get("enabled", enabled_default)),
        available=bool(meta.get("available", meta.get("loaded", False))),
        loaded=bool(meta.get("loaded", False)),
        detail=meta.get("detail"),
    )


def _validate_texts(texts: list[str]) -> list[str]:
    if not texts:
        raise HTTPException(status_code=422, detail="Debe enviar al menos un texto")
    cleaned = [str(t) for t in texts]
    if any(t.strip() == "" for t in cleaned):
        raise HTTPException(status_code=422, detail="No se permiten textos vacíos")
    return cleaned


@app.get("/", tags=["meta"])
def root() -> dict:
    settings = _get_settings()
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.app_env,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    settings = _get_settings()
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
    )


@app.get("/health/ready", tags=["meta"])
def readiness(response: Response) -> dict:
    sentiment_meta = _get_sentiment_service().metadata()
    classifier_meta = _get_classifier_service().metadata()
    degraded = bool(sentiment_meta.get("detail"))
    if classifier_meta.get("enabled"):
        degraded = degraded or bool(classifier_meta.get("detail"))
    response.status_code = (
        status.HTTP_503_SERVICE_UNAVAILABLE if degraded else status.HTTP_200_OK
    )
    return {
        "status": "degraded" if degraded else "ready",
        "sentiment": sentiment_meta,
        "classifier": classifier_meta,
    }


@app.get("/v1/models", response_model=ModelsInfoResponse, tags=["meta"])
def models_info() -> ModelsInfoResponse:
    sentiment_meta = _get_sentiment_service().metadata()
    classifier_meta = _get_classifier_service().metadata()
    return ModelsInfoResponse(
        sentiment=_service_status_from_metadata(
            {"enabled": True, "available": sentiment_meta.get("loaded", False), **sentiment_meta}
        ),
        classifier=_service_status_from_metadata(classifier_meta),
    )


@app.post("/v1/models/warmup", response_model=ModelsInfoResponse, tags=["meta"])
def warmup_models() -> ModelsInfoResponse:
    sentiment_service = _get_sentiment_service()
    classifier_service = _get_classifier_service()

    sentiment_service.load()
    if classifier_service.enabled:
        try:
            classifier_service.load()
        except ServiceUnavailableError:
            # El clasificador puede ser opcional en despliegues iniciales.
            pass

    return models_info()


@app.post("/v1/sentiment", response_model=SentimentResponse, tags=["sentiment"])
def predict_sentiment(request: SentimentRequest) -> SentimentResponse:
    texts = _validate_texts(request.texts)
    service = _get_sentiment_service()
    items_raw = service.predict_texts(texts, add_probabilities=request.add_probabilities)
    items = [SentimentItem(**item) for item in items_raw]
    return SentimentResponse(items=items, model_name=service.cfg.model_name)


@app.post("/v1/predict", response_model=PredictResponse, tags=["prediction"])
def predict_classifier(request: PredictRequest) -> PredictResponse:
    texts = _validate_texts(request.texts)
    service = _get_classifier_service()
    items_raw = service.predict_texts(texts, return_probabilities=request.return_probabilities)
    items = [PredictionItem(**item) for item in items_raw]
    return PredictResponse(items=items, model_loaded_from=service.artifacts.model_path)


@app.post("/v1/analyze", response_model=AnalyzeResponse, tags=["analysis"])
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    texts = _validate_texts(request.texts)

    sentiment_service = _get_sentiment_service()
    sentiment_items_raw = sentiment_service.predict_texts(
        texts,
        add_probabilities=request.add_sentiment_probabilities,
    )
    sentiment_items = [SentimentItem(**item) for item in sentiment_items_raw]

    classifier_service = _get_classifier_service()
    prediction_items_map: list[PredictionItem | None] = [None] * len(texts)

    if classifier_service.enabled:
        try:
            prediction_items_raw = classifier_service.predict_texts(
                texts,
                return_probabilities=request.add_prediction_probabilities,
            )
            prediction_items_map = [PredictionItem(**item) for item in prediction_items_raw]
        except ServiceUnavailableError as exc:
            if request.require_prediction:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

    items = [
        AnalyzeItem(text=texts[i], sentiment=sentiment_items[i], prediction=prediction_items_map[i])
        for i in range(len(texts))
    ]
    return AnalyzeResponse(items=items)

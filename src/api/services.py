from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from src.modeling.sentiment_analisis import (
    ChunksDataset,
    InferenceConfig,
    _prepare_tokenizer_and_model,
    _softmax,
)


class ServiceUnavailableError(RuntimeError):
    pass


@dataclass
class ClassifierArtifacts:
    model_path: str | None
    vectorizer_path: str | None


class SentimentService:
    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg
        self._tokenizer = None
        self._model = None
        self._device: str | None = None
        self._lock = threading.RLock()
        self._loaded = False
        self._load_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._loaded and self._model is not None and self._tokenizer is not None

    def load(self) -> None:
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            try:
                tok, mdl, device = _prepare_tokenizer_and_model(self.cfg)
                # warmup mínimo para evitar primer request lento/errores de padding
                mini = tok(["hola mundo"], return_tensors="pt", padding=True, truncation=True, max_length=64)
                with torch.no_grad():
                    _ = mdl(**{k: v.to(device) for k, v in mini.items()})
                self._tokenizer = tok
                self._model = mdl
                self._device = device
                self._loaded = True
                self._load_error = None
            except Exception as exc:
                self._tokenizer = None
                self._model = None
                self._device = None
                self._loaded = False
                self._load_error = str(exc)
                raise ServiceUnavailableError(self._load_error) from exc

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.cfg.model_name,
            "device": self._device,
            "available": self.loaded,
            "loaded": self.loaded,
            "batch_size": self.cfg.batch_size,
            "max_length": self.cfg.max_length,
            "stride": self.cfg.stride,
            "detail": self._load_error,
        }

    def predict_texts(self, texts: list[str], add_probabilities: bool = False) -> list[dict[str, Any]]:
        if not texts:
            return []
        self.load()
        with self._lock:
            return self._predict_locked(texts, add_probabilities=add_probabilities)

    def _predict_locked(self, texts: list[str], add_probabilities: bool = False) -> list[dict[str, Any]]:
        tok = self._tokenizer
        mdl = self._model
        device = self._device
        assert tok is not None and mdl is not None and device is not None

        enc = tok(
            texts,
            truncation=True,
            max_length=self.cfg.max_length,
            stride=self.cfg.stride,
            return_overflowing_tokens=True,
            return_tensors=None,
        )
        sample_map = enc.pop("overflow_to_sample_mapping")

        ds = ChunksDataset(enc, sample_map)
        collator = DataCollatorWithPadding(tokenizer=tok)

        def collate_fn(batch):
            sample_idx = torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long)
            model_inputs = {k: [b[k] for b in batch] for k in batch[0].keys() if k != "sample_idx"}
            model_inputs = collator(model_inputs)
            model_inputs["sample_idx"] = sample_idx
            return model_inputs

        dl = DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.cfg.num_workers,
            pin_memory=(self.cfg.pin_memory and (device == "cuda")),
        )

        raw_id2label = getattr(mdl.config, "id2label", {}) or {}
        id2label = {int(k): str(v) for k, v in raw_id2label.items()} if raw_id2label else {}
        num_labels = int(getattr(mdl.config, "num_labels", 2))
        if not id2label:
            id2label = {i: f"LABEL_{i}" for i in range(num_labels)}

        sum_logits = np.zeros((len(texts), num_labels), dtype=np.float32)
        count_chunks = np.zeros((len(texts),), dtype=np.int32)

        with torch.no_grad():
            for batch in dl:
                sample_idx = batch.pop("sample_idx").cpu().numpy()
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                out = mdl(**batch)
                logits = out.logits.detach().cpu().numpy()
                for i, doc_id in enumerate(sample_idx):
                    sum_logits[doc_id] += logits[i]
                    count_chunks[doc_id] += 1

        mean_logits = sum_logits / np.maximum(count_chunks[:, None], 1)
        probs = _softmax(mean_logits)
        pred_ids = probs.argmax(axis=1)

        results: list[dict[str, Any]] = []
        for idx, text in enumerate(texts):
            pred_id = int(pred_ids[idx])
            label = id2label.get(pred_id, str(pred_id))
            item: dict[str, Any] = {
                "text": text,
                "label": label,
                "score": float(probs[idx, pred_id]),
            }
            if add_probabilities:
                item["probabilities"] = {
                    id2label.get(j, str(j)): float(probs[idx, j])
                    for j in range(num_labels)
                }
            results.append(item)
        return results


class ClassifierService:
    def __init__(self, artifacts: ClassifierArtifacts, enabled: bool = True):
        self.artifacts = artifacts
        self.enabled = enabled
        self._model = None
        self._vectorizer = None
        self._loaded = False
        self._load_error: str | None = None
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._loaded and self._model is not None

    @property
    def available(self) -> bool:
        return self.enabled and self.loaded

    def _validate_paths(self) -> None:
        if not self.artifacts.model_path:
            raise ServiceUnavailableError("CLASSIFIER_MODEL_PATH no configurado")
        if not Path(self.artifacts.model_path).exists():
            raise ServiceUnavailableError(f"No existe el modelo: {self.artifacts.model_path}")
        if self.artifacts.vectorizer_path and not Path(self.artifacts.vectorizer_path).exists():
            raise ServiceUnavailableError(f"No existe el vectorizer: {self.artifacts.vectorizer_path}")

    def load(self) -> None:
        if not self.enabled:
            self._load_error = "Servicio deshabilitado por configuración"
            return
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            try:
                self._validate_paths()
                self._model = joblib.load(self.artifacts.model_path)
                self._vectorizer = joblib.load(self.artifacts.vectorizer_path) if self.artifacts.vectorizer_path else None
                self._loaded = True
                self._load_error = None
            except Exception as exc:
                self._loaded = False
                self._model = None
                self._vectorizer = None
                self._load_error = str(exc)
                raise ServiceUnavailableError(self._load_error) from exc

    def ensure_available(self) -> None:
        if not self.enabled:
            raise ServiceUnavailableError("Servicio de predicción deshabilitado")
        if not self.loaded:
            self.load()
        if not self.loaded:
            raise ServiceUnavailableError(self._load_error or "Servicio no disponible")

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "loaded": self.loaded,
            "available": self.available,
            "model_path": self.artifacts.model_path,
            "vectorizer_path": self.artifacts.vectorizer_path,
            "detail": self._load_error,
        }

    def predict_texts(self, texts: list[str], return_probabilities: bool = False) -> list[dict[str, Any]]:
        if not texts:
            return []
        self.ensure_available()
        with self._lock:
            return self._predict_locked(texts, return_probabilities=return_probabilities)

    def _predict_locked(self, texts: list[str], return_probabilities: bool = False) -> list[dict[str, Any]]:
        model = self._model
        vectorizer = self._vectorizer
        assert model is not None

        X = vectorizer.transform(texts) if vectorizer is not None else texts
        preds = model.predict(X)
        prob_dicts: list[dict[str, float] | None] = [None] * len(texts)

        if return_probabilities:
            probs, classes = self._predict_probabilities(model, X)
            if probs is not None and classes is not None:
                class_names = [str(c) for c in classes]
                prob_dicts = [
                    {class_names[j]: float(probs[i, j]) for j in range(len(class_names))}
                    for i in range(probs.shape[0])
                ]

        out: list[dict[str, Any]] = []
        for i, text in enumerate(texts):
            out.append(
                {
                    "text": text,
                    "prediction": self._to_python_scalar(preds[i]),
                    "probabilities": prob_dicts[i],
                }
            )
        return out

    def _predict_probabilities(self, model, X):
        classes = self._get_classes(model)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            return np.asarray(probs), classes

        if hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(X))
            if scores.ndim == 1:
                scores = np.column_stack([-scores, scores])
                if classes is None:
                    classes = np.array([0, 1])
            probs = _softmax(scores.astype(np.float32))
            return probs, classes

        return None, classes

    def _get_classes(self, model):
        classes = getattr(model, "classes_", None)
        if classes is not None:
            return np.asarray(classes)
        # sklearn Pipeline (si no proxya classes_)
        final_estimator = getattr(model, "steps", None)
        if final_estimator:
            est = final_estimator[-1][1]
            classes = getattr(est, "classes_", None)
            if classes is not None:
                return np.asarray(classes)
        return None

    @staticmethod
    def _to_python_scalar(value: Any) -> Any:
        if isinstance(value, (np.generic,)):
            return value.item()
        return value

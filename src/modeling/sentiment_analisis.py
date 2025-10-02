from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import os
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

try:
    from tqdm.auto import tqdm
    _USE_TQDM = True
except Exception:
    _USE_TQDM = False


if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "enable_flash_sdp"):
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


@dataclass
class InferenceConfig:
    model_name: str = "ignacio-ave/beto-sentiment-analysis-spanish"
    text_col: str = "texto_tfidf"
    out_label_col: str = "sentimiento"
    out_score_col: str = "score"
    batch_size: int = 32
    max_length: int = 512
    stride: int = 128
    device: Optional[str] = None  # "cuda", "cpu" o None (auto)
    num_workers: int = 0          # DataLoader workers
    pin_memory: bool = True
    add_proba_cols: bool = False  # si True, agrega una columna por clase con probas
    progress: bool = True         # mostrar barra de progreso si hay tqdm
    padding_side: str = "right"
    truncation_side: str = "right"


class ChunksDataset(Dataset):
    """
    Dataset de 'chunks' tokenizados con mapeo al índice de documento original.
    """
    def __init__(self, encodings: Dict[str, List[List[int]]], sample_map: List[int]):
        self.encodings = encodings
        self.sample_map = sample_map

    def __len__(self):
        return len(self.sample_map)

    def __getitem__(self, idx: int):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["sample_idx"] = self.sample_map[idx]
        return item


def _prepare_tokenizer_and_model(cfg: InferenceConfig):
    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True, model_max_length=cfg.max_length)
    tok.padding_side = cfg.padding_side
    tok.truncation_side = cfg.truncation_side
    if tok.pad_token is None:
        # fallback razonable
        tok.pad_token = getattr(tok, "eos_token", None) or getattr(tok, "cls_token", None)

    mdl = AutoModelForSequenceClassification.from_pretrained(cfg.model_name)
    mdl.config.pad_token_id = tok.pad_token_id

    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mdl = mdl.to(device).eval()
    return tok, mdl, device


def _softmax(logits: np.ndarray) -> np.ndarray:
    # estabilidad numérica
    z = logits - logits.max(axis=-1, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis=-1, keepdims=True)
    return z


def predict_dataframe(
    df: pd.DataFrame,
    cfg: InferenceConfig = InferenceConfig(),):

    """
    Realiza inferencia sobre el DataFrame y devuelve una copia con columnas nuevas.

    - Usa chunking con (max_length, stride).
    - Agrega columnas: cfg.out_label_col y cfg.out_score_col
    - Opcionalmente agrega columnas de probabilidad por clase si cfg.add_proba_cols=True.
    """
    assert cfg.text_col in df.columns, f"No existe la columna de texto '{cfg.text_col}'."
    texts = df[cfg.text_col].fillna("").astype(str).tolist()

    tok, mdl, device = _prepare_tokenizer_and_model(cfg)

    enc = tok(
        texts,
        truncation=True,
        max_length=cfg.max_length,
        stride=cfg.stride,
        return_overflowing_tokens=True,
        return_tensors=None,)
    
    sample_map = enc.pop("overflow_to_sample_mapping")

    _mini = tok(["hola mundo"], return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        _ = mdl(**{k: v.to(device) for k, v in _mini.items()})

    ds = ChunksDataset(enc, sample_map)
    collator = DataCollatorWithPadding(tokenizer=tok)
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        sample_idx = torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long)
        model_inputs = {k: [b[k] for k in batch] for k in batch[0].keys() if k != "sample_idx"}
        model_inputs = collator(model_inputs)
        model_inputs["sample_idx"] = sample_idx
        return model_inputs

    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.pin_memory and (device == "cuda")))

    id2label = mdl.config.id2label
    num_labels = mdl.config.num_labels

    sum_logits = np.zeros((len(texts), num_labels), dtype=np.float32)
    count_chunks = np.zeros((len(texts),), dtype=np.int32)

    iterator = dl
    if cfg.progress and _USE_TQDM:
        iterator = tqdm(dl, desc="Inferencia", leave=False)

    with torch.no_grad():
        for batch in iterator:
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
    labels = [id2label[int(i)] for i in pred_ids]
    scores = probs[np.arange(len(pred_ids)), pred_ids].astype(float).tolist()

    out_df = df.copy()
    out_df[cfg.out_label_col] = labels
    out_df[cfg.out_score_col] = scores

    if cfg.add_proba_cols:
        for j in range(num_labels):
            lab = id2label[j]
            colname = f"proba__{lab}"
            out_df[colname] = probs[:, j].astype(float)

    return out_df


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inferencia de sentimiento por lotes (HF SequenceClassification).")
    p.add_argument("--input_csv", type=str, required=True, help="Ruta al CSV de entrada.")
    p.add_argument("--output_csv", type=str, required=True, help="Ruta al CSV de salida.")
    p.add_argument("--model_name", type=str, default=InferenceConfig.model_name)
    p.add_argument("--text_col", type=str, default=InferenceConfig.text_col)
    p.add_argument("--out_label_col", type=str, default=InferenceConfig.out_label_col)
    p.add_argument("--out_score_col", type=str, default=InferenceConfig.out_score_col)
    p.add_argument("--batch_size", type=int, default=InferenceConfig.batch_size)
    p.add_argument("--max_length", type=int, default=InferenceConfig.max_length)
    p.add_argument("--stride", type=int, default=InferenceConfig.stride)
    p.add_argument("--device", type=str, default=None, help="'cpu' o 'cuda'")
    p.add_argument("--num_workers", type=int, default=InferenceConfig.num_workers)
    p.add_argument("--no_pin_memory", action="store_true", help="Desactiva pin_memory.")
    p.add_argument("--add_proba_cols", action="store_true", help="Agrega columnas de probabilidad por clase.")
    p.add_argument("--no_progress", action="store_true", help="Oculta barra de progreso.")
    return p


def main():
    ap = _build_argparser()
    args = ap.parse_args()
    df = pd.read_csv(args.input_csv)

    cfg = InferenceConfig(
        model_name=args.model_name,
        text_col=args.text_col,
        out_label_col=args.out_label_col,
        out_score_col=args.out_score_col,
        batch_size=args.batch_size,
        max_length=args.max_length,
        stride=args.stride,
        device=args.device,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        add_proba_cols=args.add_proba_cols,
        progress=not args.no_progress)

    out_df = predict_dataframe(df, cfg)
    out_df.to_csv(args.output_csv, index=False)
    print(f"[OK] Guardado en: {args.output_csv}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.sentiment_analisis import InferenceConfig, predict_csv


def build_argparser() -> argparse.ArgumentParser:
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


def main() -> None:
    args = build_argparser().parse_args()
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
        progress=not args.no_progress,
    )
    predict_csv(args.input_csv, args.output_csv, cfg=cfg)
    print(f"[OK] Guardado en: {args.output_csv}")


if __name__ == "__main__":
    main()

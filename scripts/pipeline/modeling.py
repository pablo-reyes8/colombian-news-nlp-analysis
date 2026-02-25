from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import numpy as np
import pandas as pd
from scipy import sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.clasification_models import (
    entrenar_mlp_randomsearch,
    entrenar_rf_randomsearch,
    entrenar_svm_randomsearch,
)


def _load_X(path: str):
    return sp.load_npz(path)


def _load_y(path: str, label_col: str) -> np.ndarray:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(p)
    elif suffix == ".parquet":
        df = pd.read_parquet(p)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(p)
    else:
        raise ValueError("Formato no soportado para y. Usa .csv/.parquet/.xlsx")
    if label_col not in df.columns:
        raise ValueError(f"No existe la columna de labels '{label_col}' en {path}")
    return df[label_col].to_numpy()


def _save_outputs(model, cv_results: pd.DataFrame, args: argparse.Namespace, model_name: str) -> None:
    model_path = Path(args.output_model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    if args.output_cv:
        cv_path = Path(args.output_cv)
        cv_path.parent.mkdir(parents=True, exist_ok=True)
        cv_results.to_csv(cv_path, index=False)

    if args.output_metadata:
        meta_path = Path(args.output_metadata)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "model_type": model_name,
            "x_nnz": int(getattr(args, "_x_nnz", -1)),
            "x_shape": list(getattr(args, "_x_shape", ())),
            "labels_count": int(getattr(args, "_y_count", 0)),
            "classes": [str(c) for c in getattr(model, "classes_", [])] if hasattr(model, "classes_") else None,
        }
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Modelo guardado en: {args.output_model}")
    if args.output_cv:
        print(f"[OK] Resultados CV guardados en: {args.output_cv}")
    if args.output_metadata:
        print(f"[OK] Metadata guardada en: {args.output_metadata}")


def _run_train(args: argparse.Namespace) -> None:
    X = _load_X(args.x_npz)
    y = _load_y(args.y_file, args.label_col)
    args._x_shape = tuple(int(v) for v in X.shape)
    args._x_nnz = int(X.nnz) if sp.issparse(X) else int(np.count_nonzero(X))
    args._y_count = int(len(y))

    common = dict(
        test_size=args.test_size,
        random_state=args.random_state,
        n_iter=args.n_iter,
        cv_splits=args.cv_splits,
        n_jobs=args.n_jobs,
        scoring=args.scoring,
        verbose=args.verbose,
    )

    if args.cmd == "train-svm":
        model, cv = entrenar_svm_randomsearch(
            X,
            y,
            usar_class_weight_balanced=not args.no_class_weight_balanced,
            **common,
        )
        _save_outputs(model, cv, args, "svm")
        return

    if args.cmd == "train-rf":
        model, cv = entrenar_rf_randomsearch(
            X,
            y,
            considerar_class_weight=not args.no_class_weight,
            **common,
        )
        _save_outputs(model, cv, args, "rf")
        return

    if args.cmd == "train-mlp":
        model, cv = entrenar_mlp_randomsearch(
            X,
            y,
            max_iter=args.max_iter,
            **common,
        )
        _save_outputs(model, cv, args, "mlp")
        return

    raise RuntimeError(f"Comando no soportado: {args.cmd}")


def _add_common_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--x-npz", required=True, help="Matriz sparse .npz (scipy)")
    p.add_argument("--y-file", required=True, help="CSV/Parquet/XLSX con labels")
    p.add_argument("--label-col", required=True, help="Columna de labels")
    p.add_argument("--output-model", required=True, help="Ruta de salida .joblib")
    p.add_argument("--output-cv", help="Ruta para resultados CV (.csv)")
    p.add_argument("--output-metadata", help="Ruta para metadata (.json)")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-iter", type=int, default=30)
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--scoring", default="f1_macro")
    p.add_argument("--verbose", type=int, default=1)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CLI de entrenamiento de modelos (SVM/RF/MLP)")
    sp_parser = p.add_subparsers(dest="cmd", required=True)

    p_svm = sp_parser.add_parser("train-svm", help="Entrena LinearSVC con RandomizedSearchCV")
    _add_common_train_args(p_svm)
    p_svm.add_argument("--no-class-weight-balanced", action="store_true")

    p_rf = sp_parser.add_parser("train-rf", help="Entrena RandomForest con RandomizedSearchCV")
    _add_common_train_args(p_rf)
    p_rf.add_argument("--no-class-weight", action="store_true")
    p_rf.set_defaults(n_iter=40)

    p_mlp = sp_parser.add_parser("train-mlp", help="Entrena MLP con RandomizedSearchCV")
    _add_common_train_args(p_mlp)
    p_mlp.add_argument("--max-iter", type=int, default=200)

    return p


def main() -> None:
    args = build_argparser().parse_args()
    _run_train(args)


if __name__ == "__main__":
    main()

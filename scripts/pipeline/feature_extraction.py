from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import pandas as pd
from scipy import sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.main_preprocessing import preprocess_text_column
from src.embeddings.bow import BowConfig, BowFeaturizerDF
from src.embeddings.tf_idf import fit_transform_embeddings, transform_embeddings


def _read_df(path: str) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    raise ValueError("Formato no soportado. Usa .csv, .parquet o .xlsx")


def _write_df(df: pd.DataFrame, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        df.to_csv(p, index=False)
    elif suffix == ".parquet":
        df.to_parquet(p, index=False)
    elif suffix in {".xlsx", ".xls"}:
        df.to_excel(p, index=False)
    else:
        raise ValueError("Formato no soportado. Usa .csv, .parquet o .xlsx")


def _save_sparse_matrix(X, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(p, X)


def _save_json(data: dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _add_common_preprocess_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", required=True, help="CSV/Parquet/XLSX de entrada")
    p.add_argument("--output", required=True, help="CSV/Parquet/XLSX de salida")
    p.add_argument("--input-col", required=True, help="Columna de texto crudo")
    p.add_argument("--output-col", default="texto_tfidf", help="Columna de texto procesado")
    p.add_argument("--spacy-model", default="es_core_news_sm")
    p.add_argument("--remove-accents", dest="remove_accents", action="store_true")
    p.add_argument("--no-remove-accents", dest="remove_accents", action="store_false")
    p.set_defaults(remove_accents=True)
    p.add_argument("--remove-stopwords", dest="remove_stopwords", action="store_true")
    p.add_argument("--no-remove-stopwords", dest="remove_stopwords", action="store_false")
    p.set_defaults(remove_stopwords=True)
    p.add_argument("--remove-numbers", dest="remove_numbers", action="store_true")
    p.add_argument("--no-remove-numbers", dest="remove_numbers", action="store_false")
    p.set_defaults(remove_numbers=True)
    p.add_argument("--remove-punct", dest="remove_punct", action="store_true")
    p.add_argument("--no-remove-punct", dest="remove_punct", action="store_false")
    p.set_defaults(remove_punct=True)
    p.add_argument("--min-token-len", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-process", type=int, default=1)
    p.add_argument("--return-tokens", action="store_true")


def _cmd_preprocess(args: argparse.Namespace) -> None:
    df = _read_df(args.input)
    out = preprocess_text_column(
        df=df,
        input_col=args.input_col,
        output_col=args.output_col,
        spacy_model=args.spacy_model,
        remove_accents=args.remove_accents,
        remove_numbers=args.remove_numbers,
        remove_punct=args.remove_punct,
        remove_stopwords=args.remove_stopwords,
        min_token_len=args.min_token_len,
        batch_size=args.batch_size,
        n_process=args.n_process,
        return_tokens=args.return_tokens,
    )
    _write_df(out, args.output)
    print(f"[OK] Preprocesamiento guardado en: {args.output}")
    print(f"[INFO] Filas: {len(out)}")


def _cmd_tfidf_fit(args: argparse.Namespace) -> None:
    df = _read_df(args.input)
    X, v_word, v_char = fit_transform_embeddings(
        df_train=df,
        col=args.text_col,
        word_ngram_range=(args.word_ngram_min, args.word_ngram_max),
        char_ngram_range=(args.char_ngram_min, args.char_ngram_max),
        min_df=args.min_df,
        max_df=args.max_df,
        sublinear_tf=not args.no_sublinear_tf,
        norm=args.norm,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_sparse_matrix(X, str(out_dir / "X_tfidf.npz"))
    joblib.dump(v_word, out_dir / "tfidf_word_vectorizer.joblib")
    joblib.dump(v_char, out_dir / "tfidf_char_vectorizer.joblib")
    _save_json(
        {
            "rows": int(X.shape[0]),
            "cols": int(X.shape[1]),
            "text_col": args.text_col,
            "word_ngram_range": [args.word_ngram_min, args.word_ngram_max],
            "char_ngram_range": [args.char_ngram_min, args.char_ngram_max],
            "min_df": args.min_df,
            "max_df": args.max_df,
            "sublinear_tf": not args.no_sublinear_tf,
            "norm": args.norm,
        },
        str(out_dir / "metadata.json"),
    )
    print(f"[OK] Features TF-IDF guardadas en: {out_dir}")
    print(f"[INFO] X shape: {X.shape}")


def _cmd_tfidf_transform(args: argparse.Namespace) -> None:
    df = _read_df(args.input)
    v_word = joblib.load(args.word_vectorizer)
    v_char = joblib.load(args.char_vectorizer)
    X = transform_embeddings(df_test=df, v_word=v_word, v_char=v_char, col=args.text_col)
    _save_sparse_matrix(X, args.output_matrix)
    print(f"[OK] Matriz transformada guardada en: {args.output_matrix}")
    print(f"[INFO] X shape: {X.shape}")


def _cmd_bow_fit(args: argparse.Namespace) -> None:
    df = _read_df(args.input)
    cfg = BowConfig(
        ngram_range=(args.ngram_min, args.ngram_max),
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        binary=args.binary,
    )
    bow = BowFeaturizerDF(cfg)
    _, X = bow.fit_transform_add_column(df, text_col=args.text_col, out_col="bow", tolist=False)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_sparse_matrix(X, str(out_dir / "X_bow.npz"))
    bow.save_vectorizer(str(out_dir / "bow_vectorizer.joblib"))
    _save_json(
        {
            "rows": int(X.shape[0]),
            "cols": int(X.shape[1]),
            "text_col": args.text_col,
            "ngram_range": [args.ngram_min, args.ngram_max],
            "max_features": args.max_features,
            "min_df": args.min_df,
            "max_df": args.max_df,
            "binary": args.binary,
        },
        str(out_dir / "metadata.json"),
    )
    print(f"[OK] Features BOW guardadas en: {out_dir}")
    print(f"[INFO] X shape: {X.shape}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CLI de feature extraction (preprocesamiento, TF-IDF y BOW)")
    sp = p.add_subparsers(dest="cmd", required=True)

    p_pre = sp.add_parser("preprocess", help="Preprocesa una columna de texto con spaCy")
    _add_common_preprocess_args(p_pre)

    p_tfidf_fit = sp.add_parser("tfidf-fit", help="Ajusta vectorizadores TF-IDF y guarda matriz/features")
    p_tfidf_fit.add_argument("--input", required=True)
    p_tfidf_fit.add_argument("--text-col", default="texto_tfidf")
    p_tfidf_fit.add_argument("--output-dir", required=True)
    p_tfidf_fit.add_argument("--word-ngram-min", type=int, default=1)
    p_tfidf_fit.add_argument("--word-ngram-max", type=int, default=2)
    p_tfidf_fit.add_argument("--char-ngram-min", type=int, default=3)
    p_tfidf_fit.add_argument("--char-ngram-max", type=int, default=5)
    p_tfidf_fit.add_argument("--min-df", type=int, default=3)
    p_tfidf_fit.add_argument("--max-df", type=float, default=0.9)
    p_tfidf_fit.add_argument("--norm", default="l2")
    p_tfidf_fit.add_argument("--no-sublinear-tf", action="store_true")

    p_tfidf_transform = sp.add_parser("tfidf-transform", help="Transforma textos con TF-IDF ya ajustado")
    p_tfidf_transform.add_argument("--input", required=True)
    p_tfidf_transform.add_argument("--text-col", default="texto_tfidf")
    p_tfidf_transform.add_argument("--word-vectorizer", required=True)
    p_tfidf_transform.add_argument("--char-vectorizer", required=True)
    p_tfidf_transform.add_argument("--output-matrix", required=True)

    p_bow_fit = sp.add_parser("bow-fit", help="Ajusta BOW y guarda matriz/features")
    p_bow_fit.add_argument("--input", required=True)
    p_bow_fit.add_argument("--text-col", default="texto_tfidf")
    p_bow_fit.add_argument("--output-dir", required=True)
    p_bow_fit.add_argument("--ngram-min", type=int, default=1)
    p_bow_fit.add_argument("--ngram-max", type=int, default=1)
    p_bow_fit.add_argument("--max-features", type=int)
    p_bow_fit.add_argument("--min-df", type=float, default=1)
    p_bow_fit.add_argument("--max-df", type=float, default=1.0)
    p_bow_fit.add_argument("--binary", action="store_true")

    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.cmd == "preprocess":
        _cmd_preprocess(args)
    elif args.cmd == "tfidf-fit":
        _cmd_tfidf_fit(args)
    elif args.cmd == "tfidf-transform":
        _cmd_tfidf_transform(args)
    elif args.cmd == "bow-fit":
        _cmd_bow_fit(args)
    else:
        raise RuntimeError(f"Comando no soportado: {args.cmd}")


if __name__ == "__main__":
    main()

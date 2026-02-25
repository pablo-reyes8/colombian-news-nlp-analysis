from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.scrapping_elcolombiano.scraper_full import (
    DEFAULT_COLUMNISTAS,
    DEFAULT_REFERENCE_DATE_TEXT,
    DEFAULT_SECTION_URLS,
    run_pipeline_elcolombiano,
    scrapear_noticias_columnistas,
    scrapear_noticias_estaticas,
)
from data.scrapping_elcolombiano.scrapping_columnosta import scrape_columnista
from data.scrapping_elcolombiano.scrapping_one_new import scrape_noticia
from data.scrapping_elcolombiano.utils_scrapping import collect_links_from_default_sections


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def _save_df(df: pd.DataFrame, path_str: str) -> None:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif suffix == ".json":
        df.to_json(path, orient="records", force_ascii=False, indent=2)
    else:
        raise ValueError("Extensión no soportada. Usa .csv, .parquet o .json")


def _save_links(links: list[Any], path_str: str) -> None:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".json":
        path.write_text(json.dumps(links, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    if suffix == ".txt":
        path.write_text("\n".join(str(x) for x in links), encoding="utf-8")
        return
    if suffix == ".csv":
        if links and isinstance(links[0], (tuple, list)) and len(links[0]) >= 2:
            pd.DataFrame(links, columns=["url", "categoria"]).to_csv(path, index=False)
        else:
            pd.DataFrame({"url": links}).to_csv(path, index=False)
        return
    raise ValueError("Extensión no soportada. Usa .json, .txt o .csv")


def _save_record(record: dict, path_str: str) -> None:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        pd.DataFrame([record]).to_csv(path, index=False)


def _print_sections() -> None:
    print(f"Secciones estáticas del notebook: {len(DEFAULT_SECTION_URLS)}")
    for url in DEFAULT_SECTION_URLS:
        print(f"- {url}")
    print(f"\nColumnistas por defecto: {len(DEFAULT_COLUMNISTAS)}")
    for url in DEFAULT_COLUMNISTAS:
        print(f"- {url}")


def _print_df_summary(name: str, df: pd.DataFrame) -> None:
    print(f"[INFO] {name}: {len(df)} registros")
    if "categoria" in df.columns and not df.empty:
        counts = df["categoria"].value_counts(dropna=False)
        print(f"[INFO] {name} por categoría:")
        for categoria, total in counts.items():
            print(f"- {categoria}: {int(total)}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CLI de scraping para El Colombiano")
    sp = p.add_subparsers(dest="cmd", required=True)

    sp.add_parser("list", help="Listar secciones y columnistas por defecto del notebook")

    p_one = sp.add_parser("one-news", help="Scrapear una noticia individual")
    p_one.add_argument("url")
    p_one.add_argument("--output")

    p_col = sp.add_parser("one-columnista", help="Scrapear una nota de columnista")
    p_col.add_argument("url")
    p_col.add_argument("--output")

    p_links = sp.add_parser("links-static", help="Extraer links de las secciones estáticas del notebook")
    p_links.add_argument("--strict", dest="strict", action="store_true")
    p_links.add_argument("--no-strict", dest="strict", action="store_false")
    p_links.set_defaults(strict=True)
    p_links.add_argument("--output")

    p_static = sp.add_parser("static", help="Pipeline de noticias estáticas")
    p_static.add_argument("--output", required=True)
    p_static.add_argument("--links-output")
    p_static.add_argument("--strict", dest="strict", action="store_true")
    p_static.add_argument("--no-strict", dest="strict", action="store_false")
    p_static.set_defaults(strict=True)
    p_static.add_argument("--no-deduplicate", action="store_true")
    p_static.add_argument("--hoy", default=DEFAULT_REFERENCE_DATE_TEXT)

    p_cols = sp.add_parser("columnistas", help="Pipeline de columnistas")
    p_cols.add_argument("--output", required=True)
    p_cols.add_argument("--links-output")
    p_cols.add_argument("--max-clicks", type=int, default=80)
    p_cols.add_argument("--wait-secs", type=int, default=12)
    p_cols.add_argument("--max-workers", type=int, default=12)
    p_cols.add_argument("--sleep-jitter", type=float, default=0.0)
    p_cols.add_argument("--hoy", default=DEFAULT_REFERENCE_DATE_TEXT)

    p_all = sp.add_parser("all", help="Pipeline completo (estático + columnistas)")
    p_all.add_argument("--out-static")
    p_all.add_argument("--out-columnistas")
    p_all.add_argument("--out-links-static")
    p_all.add_argument("--out-links-columnistas")
    p_all.add_argument("--strict-static", action="store_true", default=True)
    p_all.add_argument("--no-strict-static", dest="strict_static", action="store_false")
    p_all.add_argument("--no-deduplicate-static", action="store_true")
    p_all.add_argument("--hoy", default=DEFAULT_REFERENCE_DATE_TEXT)
    p_all.add_argument("--max-clicks-columnistas", type=int, default=80)
    p_all.add_argument("--wait-secs-columnistas", type=int, default=12)
    p_all.add_argument("--max-workers-columnistas", type=int, default=12)
    p_all.add_argument("--sleep-jitter-columnistas", type=float, default=0.0)

    return p


def main() -> None:
    args = build_argparser().parse_args()

    if args.cmd == "list":
        _print_sections()
        return

    if args.cmd == "one-news":
        rec = scrape_noticia(args.url)
        print(json.dumps(rec, ensure_ascii=False, indent=2))
        if args.output:
            _save_record(rec, args.output)
        return

    if args.cmd == "one-columnista":
        rec = scrape_columnista(args.url)
        print(json.dumps(rec, ensure_ascii=False, indent=2))
        if args.output:
            _save_record(rec, args.output)
        return

    if args.cmd == "links-static":
        links = collect_links_from_default_sections(strict=args.strict)
        print(f"[INFO] Total links: {len(links)}")
        if args.output:
            _save_links(links, args.output)
            print(f"[OK] Links guardados en: {args.output}")
        return

    if args.cmd == "static":
        df, links = scrapear_noticias_estaticas(
            strict=args.strict,
            deduplicate=not args.no_deduplicate,
            hoy=args.hoy,
        )
        _save_df(df, args.output)
        if args.links_output:
            _save_links(links, args.links_output)
        print(f"[OK] Guardado en: {args.output}")
        _print_df_summary("noticias_estatico", df)
        return

    if args.cmd == "columnistas":
        df, links = scrapear_noticias_columnistas(
            max_clicks=args.max_clicks,
            wait_secs=args.wait_secs,
            max_workers=args.max_workers,
            sleep_jitter=args.sleep_jitter,
            hoy=args.hoy,
        )
        _save_df(df, args.output)
        if args.links_output:
            _save_links(links, args.links_output)
        print(f"[OK] Guardado en: {args.output}")
        _print_df_summary("noticias_columnistas", df)
        return

    if args.cmd == "all":
        out = run_pipeline_elcolombiano(
            strict_static_links=args.strict_static,
            deduplicate_static_links=not args.no_deduplicate_static,
            hoy=args.hoy,
            max_clicks_columnistas=args.max_clicks_columnistas,
            wait_secs_columnistas=args.wait_secs_columnistas,
            max_workers_columnistas=args.max_workers_columnistas,
            sleep_jitter_columnistas=args.sleep_jitter_columnistas,
        )
        if args.out_static and "noticias_estatico" in out:
            _save_df(out["noticias_estatico"], args.out_static)
        if args.out_columnistas and "noticias_columnistas" in out:
            _save_df(out["noticias_columnistas"], args.out_columnistas)
        if args.out_links_static and "links_estatico" in out:
            _save_links(out["links_estatico"], args.out_links_static)
        if args.out_links_columnistas and "links_columnistas" in out:
            _save_links(out["links_columnistas"], args.out_links_columnistas)

        for key, value in out.items():
            if isinstance(value, pd.DataFrame):
                _print_df_summary(key, value)
            elif hasattr(value, "__len__"):
                print(f"[INFO] {key}: {len(value)}")
            else:
                print(f"[INFO] {key}: {type(value).__name__}")
        return


if __name__ == "__main__":
    main()

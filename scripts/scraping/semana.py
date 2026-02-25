from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.scrapping_semana.data.scrapping_data import (
    DEFAULT_BASE_SEMANA_URL,
    DEFAULT_NUM_VER_MAS_CLICKS,
    DEFAULT_PATTERN_CATALOG_CANONICAL,
    DEFAULT_PATTERN_CATALOG_SOUP,
    DEFAULT_TAGS,
    DEFAULT_TRIM_PATTERN,
    build_semana_notebook_config,
    consolidar_noticias_semana,
    guardar_dataframe_semana,
    listar_catalogo_semana,
    resolver_catalogo_semana,
    run_semana_catalog_pipeline,
    run_semana_notebook_pipeline_and_consolidate,
)


def _split_csv_arg(val: str | None) -> list[str] | None:
    if val is None:
        return None
    return [x.strip() for x in val.split(",") if x.strip()]


def _print_categories(categories: list[str]) -> None:
    print(f"Total categorías: {len(categories)}")
    for category in categories:
        print(f"- {category}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CLI de scraping para Semana")
    sp = p.add_subparsers(dest="cmd", required=True)

    p_list = sp.add_parser("list-categories", help="Listar catálogos de categorías del notebook")
    p_list.add_argument("--catalog", choices=["default", "complete", "discarded"], default="default")

    p_cfg = sp.add_parser("show-config", help="Mostrar configuración equivalente al notebook")
    p_cfg.add_argument("--drive-path", required=True)
    p_cfg.add_argument("--categories", help="Lista separada por comas")
    p_cfg.add_argument("--base-url", default=DEFAULT_BASE_SEMANA_URL)
    p_cfg.add_argument("--pattern-catalog-soup", default=DEFAULT_PATTERN_CATALOG_SOUP)
    p_cfg.add_argument("--pattern-catalog-canonical", default=DEFAULT_PATTERN_CATALOG_CANONICAL)
    p_cfg.add_argument("--trim-pattern", default=DEFAULT_TRIM_PATTERN)
    p_cfg.add_argument("--tags", default=",".join(DEFAULT_TAGS))
    p_cfg.add_argument("--num-ver-mas-clicks", type=int, default=DEFAULT_NUM_VER_MAS_CLICKS)

    p_scrape = sp.add_parser("scrape", help="Ejecutar scraping por categorías")
    p_scrape.add_argument("--drive-path", required=True)
    p_scrape.add_argument("--categories", help="Lista separada por comas; por defecto usa catálogo default")
    p_scrape.add_argument("--base-url", default=DEFAULT_BASE_SEMANA_URL)
    p_scrape.add_argument("--pattern-catalog-soup", default=DEFAULT_PATTERN_CATALOG_SOUP)
    p_scrape.add_argument("--pattern-catalog-canonical", default=DEFAULT_PATTERN_CATALOG_CANONICAL)
    p_scrape.add_argument("--trim-pattern", default=DEFAULT_TRIM_PATTERN)
    p_scrape.add_argument("--tags", help="Tags separados por coma", default=",".join(DEFAULT_TAGS))
    p_scrape.add_argument("--num-ver-mas-clicks", type=int, default=DEFAULT_NUM_VER_MAS_CLICKS)

    p_merge = sp.add_parser("merge", help="Consolidar parquets extraídos")
    p_merge.add_argument("--input-dir", required=True)
    p_merge.add_argument("--output")
    p_merge.add_argument("--initial-date", default="2025-08-01")
    p_merge.add_argument("--final-date", default="2025-08-30")
    p_merge.add_argument("--no-filter-dates", action="store_true")
    p_merge.add_argument("--original-col", default="datePublished")
    p_merge.add_argument("--formatted-col", default="datePublishedFormatted")

    p_pipe = sp.add_parser("pipeline", help="Ejecutar scraping por categorías y luego consolidar")
    p_pipe.add_argument("--drive-path", required=True)
    p_pipe.add_argument("--output", required=True)
    p_pipe.add_argument("--categories", help="Lista separada por comas; por defecto usa catálogo default")
    p_pipe.add_argument("--base-url", default=DEFAULT_BASE_SEMANA_URL)
    p_pipe.add_argument("--pattern-catalog-soup", default=DEFAULT_PATTERN_CATALOG_SOUP)
    p_pipe.add_argument("--pattern-catalog-canonical", default=DEFAULT_PATTERN_CATALOG_CANONICAL)
    p_pipe.add_argument("--trim-pattern", default=DEFAULT_TRIM_PATTERN)
    p_pipe.add_argument("--tags", default=",".join(DEFAULT_TAGS))
    p_pipe.add_argument("--num-ver-mas-clicks", type=int, default=DEFAULT_NUM_VER_MAS_CLICKS)
    p_pipe.add_argument("--initial-date", default="2025-08-01")
    p_pipe.add_argument("--final-date", default="2025-08-30")
    p_pipe.add_argument("--no-filter-dates", action="store_true")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    if args.cmd == "list-categories":
        _print_categories(listar_catalogo_semana(modo=args.catalog))
        return

    if args.cmd == "show-config":
        cfg = build_semana_notebook_config(
            drive_path=args.drive_path,
            categories=resolver_catalogo_semana(_split_csv_arg(args.categories)),
            base_Semana_url=args.base_url,
            pattern_catalog_soup=args.pattern_catalog_soup,
            pattern_catalog_canonical=args.pattern_catalog_canonical,
            trim_pattern=args.trim_pattern,
            tag_ls=_split_csv_arg(args.tags),
            num_of_ver_mas_clicks=args.num_ver_mas_clicks,
        )
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return

    if args.cmd == "scrape":
        processed = run_semana_catalog_pipeline(
            drive_path=args.drive_path,
            categories=resolver_catalogo_semana(_split_csv_arg(args.categories)),
            base_Semana_url=args.base_url,
            pattern_catalog_soup=args.pattern_catalog_soup,
            pattern_catalog_canonical=args.pattern_catalog_canonical,
            trim_pattern=args.trim_pattern,
            tag_ls=_split_csv_arg(args.tags) or DEFAULT_TAGS,
            num_of_ver_mas_clicks=args.num_ver_mas_clicks,
        )
        print(f"[OK] Categorías procesadas: {len(processed)}")
        return

    if args.cmd == "merge":
        df = consolidar_noticias_semana(
            drive_folder_path=args.input_dir,
            original_published_date_col=args.original_col,
            formatted_published_date_col=args.formatted_col,
            initial_published_date=args.initial_date,
            final_published_date=args.final_date,
            filter_df_by_dates_flag=not args.no_filter_dates,
        )
        if args.output:
            guardar_dataframe_semana(df, args.output)
            print(f"[OK] Guardado en: {args.output}")
        else:
            print(df.head())
        print(f"[INFO] Registros: {len(df)}")
        return

    if args.cmd == "pipeline":
        df = run_semana_notebook_pipeline_and_consolidate(
            drive_path=args.drive_path,
            categories=resolver_catalogo_semana(_split_csv_arg(args.categories)),
            base_Semana_url=args.base_url,
            pattern_catalog_soup=args.pattern_catalog_soup,
            pattern_catalog_canonical=args.pattern_catalog_canonical,
            trim_pattern=args.trim_pattern,
            tag_ls=_split_csv_arg(args.tags) or DEFAULT_TAGS,
            num_of_ver_mas_clicks=args.num_ver_mas_clicks,
            initial_published_date=args.initial_date,
            final_published_date=args.final_date,
            filter_df_by_dates_flag=not args.no_filter_dates,
        )
        guardar_dataframe_semana(df, args.output)
        print(f"[OK] Guardado en: {args.output}")
        print(f"[INFO] Registros: {len(df)}")
        return


if __name__ == "__main__":
    main()

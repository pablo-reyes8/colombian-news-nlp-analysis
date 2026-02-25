from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.scrapping_espectador.main_function import (
    CATEGORIAS_ELESPECTADOR_ARCHIVO,
    CATEGORIAS_ELESPECTADOR_MENU,
    build_driver,
    guardar_dataframe_elespectador,
    listar_categorias_elespectador,
    resolver_categorias_elespectador,
    run_elespectador_archivo_pipeline,
    scrape_categoria_archivo,
)


def _split_csv_arg(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _print_categories(categories: dict[str, str]) -> None:
    print(f"Total categorías: {len(categories)}")
    for name, url in categories.items():
        print(f"- {name}: {url}")


def _print_summary(df: pd.DataFrame) -> None:
    print(f"[INFO] Registros totales: {len(df)}")
    if "Categoria" in df.columns and not df.empty:
        counts = df["Categoria"].value_counts(dropna=False)
        print("[INFO] Registros por categoría:")
        for categoria, total in counts.items():
            print(f"- {categoria}: {int(total)}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CLI de scraping para El Espectador")
    sp = p.add_subparsers(dest="cmd", required=True)

    p_list = sp.add_parser("list-categories", help="Listar categorías configuradas en data/")
    p_list.add_argument("--source", choices=["archivo", "menu"], default="archivo")

    p_cat = sp.add_parser("category", help="Scrapear una categoría de archivo")
    p_cat.add_argument("categoria", help="Nombre exacto de la categoría")
    p_cat.add_argument("--num-paginas", type=int, default=12)
    p_cat.add_argument("--output", required=True)
    p_cat.add_argument("--no-headless", action="store_true")
    p_cat.add_argument("--url", help="URL personalizada del archivo (sobrescribe la del diccionario)")

    p_all = sp.add_parser("all", help="Scrapear todas o un subconjunto de categorías de archivo")
    p_all.add_argument("--categories", help="Lista separada por comas (subset del diccionario de archivo)")
    p_all.add_argument("--num-paginas", type=int, default=12)
    p_all.add_argument("--sleep-secs", type=float, default=3.0)
    p_all.add_argument("--output", required=True)
    p_all.add_argument("--no-headless", action="store_true")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    if args.cmd == "list-categories":
        _print_categories(listar_categorias_elespectador(source=args.source))
        return

    if args.cmd == "category":
        if args.url is None and args.categoria not in CATEGORIAS_ELESPECTADOR_ARCHIVO:
            raise KeyError(
                f"Categoría '{args.categoria}' no encontrada. Usa 'list-categories --source archivo'."
            )
        nav = build_driver(headless=not args.no_headless)
        try:
            df = scrape_categoria_archivo(
                nav,
                categoria=args.categoria,
                url=args.url,
                num_paginas=args.num_paginas,
            )
        finally:
            nav.quit()
        guardar_dataframe_elespectador(df, args.output)
        print(f"[OK] Guardado en: {args.output}")
        _print_summary(df)
        return

    if args.cmd == "all":
        categorias_map = resolver_categorias_elespectador(
            categorias=_split_csv_arg(args.categories),
            source="archivo",
        )
        df = run_elespectador_archivo_pipeline(
            categorias=categorias_map.keys(),
            num_paginas=args.num_paginas,
            sleep_secs=args.sleep_secs,
            headless=not args.no_headless,
        )
        guardar_dataframe_elespectador(df, args.output)
        print(f"[OK] Guardado en: {args.output}")
        _print_summary(df)
        return


if __name__ == "__main__":
    main()

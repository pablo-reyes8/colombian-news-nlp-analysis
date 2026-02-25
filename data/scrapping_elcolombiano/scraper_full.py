"""Pipeline de scraping de El Colombiano (estático + columnistas) extraído del notebook."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

from src.scraping.elcolombiano.scraper_full import build_session, _safe_scrape_any, scrapping_all
from src.scraping.elcolombiano.scrapping_columnosta import columnistas as DEFAULT_COLUMNISTAS
from data.scrapping_elcolombiano.utils_scrapping import (
    DEFAULT_SECTION_URLS,
    build_links_tuplas,
    collect_links_from_default_sections,
    collect_links_from_sections,
)
from data.scrapping_elcolombiano.scrapper_columnistas_selenium import (
    collect_columnistas_links,
    scrapping_columnistas,
)


DEFAULT_REFERENCE_DATE_TEXT = "4 de Septiembre de 2025"


def preparar_df(lista, hoy: str = DEFAULT_REFERENCE_DATE_TEXT) -> pd.DataFrame:
    """Limpieza final del DataFrame como en el notebook (títulos/cuerpos vacíos, textos y fechas relativas)."""
    df = pd.DataFrame(lista)
    if df.empty:
        return df

    df = df.replace("", np.nan)
    df_limpio = df.dropna(subset=["titulo", "cuerpo"]).copy()

    pat = r"^\s*(?:\d+\s*y\s*\d+\s*|no\s+){4,}\s*"
    df_limpio["cuerpo"] = (
        df_limpio["cuerpo"]
        .astype(str)
        .str.replace(pat, "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    if "hora" in df_limpio.columns:
        df_limpio.loc[:, "hora"] = df_limpio["hora"].apply(
            lambda x: hoy if "hace" in str(x) else x
        )
    return df_limpio


def extraer_links_noticias_estaticas(
    links_tuplas: Iterable[tuple[str, str]] | None = None,
    strict: bool = True,
    deduplicate: bool = True,
) -> list[tuple[str, str]]:
    """Extrae links de noticias desde secciones estáticas de El Colombiano."""
    if links_tuplas is None:
        return collect_links_from_default_sections(strict=strict, deduplicate=deduplicate)
    return collect_links_from_sections(links_tuplas, strict=strict, deduplicate=deduplicate)


def scrapear_noticias_estaticas(
    links_tuplas: Iterable[tuple[str, str]] | None = None,
    strict: bool = True,
    deduplicate: bool = True,
    limpiar: bool = True,
    hoy: str = DEFAULT_REFERENCE_DATE_TEXT,
):
    """Pipeline del notebook para scraping de noticias (no columnistas)."""
    links = extraer_links_noticias_estaticas(links_tuplas=links_tuplas, strict=strict, deduplicate=deduplicate)
    noticias = scrapping_all(links)
    if limpiar:
        return preparar_df(noticias, hoy=hoy), links
    return pd.DataFrame(noticias), links


def scrapear_noticias_columnistas(
    listing_urls: Iterable[str] = DEFAULT_COLUMNISTAS,
    max_clicks: int = 80,
    wait_secs: int = 12,
    max_workers: int = 12,
    sleep_jitter: float = 0.0,
    limpiar: bool = True,
    hoy: str = DEFAULT_REFERENCE_DATE_TEXT,
    save_links_pickle_path: str | None = None,
):
    """Pipeline del notebook para columnas: obtener links con Selenium y luego scrapear en paralelo."""
    links_columnistas = collect_columnistas_links(
        columnistas_urls=listing_urls,
        max_clicks=max_clicks,
        wait_secs=wait_secs,
        save_pickle_path=save_links_pickle_path,
    )
    noticias_columnistas = scrapping_columnistas(
        links_columnistas,
        max_workers=max_workers,
        sleep_jitter=sleep_jitter,
    )
    if limpiar:
        return preparar_df(noticias_columnistas, hoy=hoy), links_columnistas
    return pd.DataFrame(noticias_columnistas), links_columnistas


def run_pipeline_elcolombiano(
    include_static: bool = True,
    include_columnistas: bool = True,
    static_links_tuplas: Iterable[tuple[str, str]] | None = None,
    columnistas_listing_urls: Iterable[str] = DEFAULT_COLUMNISTAS,
    strict_static_links: bool = True,
    deduplicate_static_links: bool = True,
    hoy: str = DEFAULT_REFERENCE_DATE_TEXT,
    max_clicks_columnistas: int = 80,
    wait_secs_columnistas: int = 12,
    max_workers_columnistas: int = 12,
    sleep_jitter_columnistas: float = 0.0,
):
    """Ejecuta el pipeline completo del notebook y retorna un diccionario con resultados."""
    out: dict[str, object] = {}

    if include_static:
        noticias_estatico, links_estatico = scrapear_noticias_estaticas(
            links_tuplas=static_links_tuplas,
            strict=strict_static_links,
            deduplicate=deduplicate_static_links,
            limpiar=True,
            hoy=hoy,
        )
        out["noticias_estatico"] = noticias_estatico
        out["links_estatico"] = links_estatico

    if include_columnistas:
        noticias_columnistas, links_columnistas = scrapear_noticias_columnistas(
            listing_urls=columnistas_listing_urls,
            max_clicks=max_clicks_columnistas,
            wait_secs=wait_secs_columnistas,
            max_workers=max_workers_columnistas,
            sleep_jitter=sleep_jitter_columnistas,
            limpiar=True,
            hoy=hoy,
        )
        out["noticias_columnistas"] = noticias_columnistas
        out["links_columnistas"] = links_columnistas

    return out


__all__ = [
    "DEFAULT_SECTION_URLS",
    "DEFAULT_COLUMNISTAS",
    "DEFAULT_REFERENCE_DATE_TEXT",
    "build_session",
    "_safe_scrape_any",
    "scrapping_all",
    "build_links_tuplas",
    "collect_links_from_sections",
    "collect_links_from_default_sections",
    "preparar_df",
    "extraer_links_noticias_estaticas",
    "scrapear_noticias_estaticas",
    "scrapear_noticias_columnistas",
    "run_pipeline_elcolombiano",
]

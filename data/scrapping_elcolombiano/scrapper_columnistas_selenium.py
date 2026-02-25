"""Pipeline de columnistas de El Colombiano (links + scraping) extraído del notebook."""
from __future__ import annotations

from pathlib import Path
import pickle
import sys
from typing import Iterable

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

from src.scraping.elcolombiano.scrapper_columnistas_selenium import (
    BAD_SUBSTR,
    _is_article,
    _normalize,
    get_listing_article_links_from_html,
    get_links_after_clicks,
    _safe_scrape,
    scrapping_columnistas,
)
from src.scraping.elcolombiano.scrapping_columnosta import columnistas as DEFAULT_COLUMNISTAS


def collect_columnistas_links(
    columnistas_urls: Iterable[str] = DEFAULT_COLUMNISTAS,
    max_clicks: int = 80,
    wait_secs: int = 12,
    save_pickle_path: str | None = None,
) -> list[str]:
    """Replica el loop del notebook para obtener links de artículos de columnistas."""
    total_noticias_columnistas: list[str] = []
    for url in columnistas_urls:
        links = get_links_after_clicks(url, max_clicks=max_clicks, wait_secs=wait_secs)
        print("Total:", len(links))
        total_noticias_columnistas.extend(links)

    if save_pickle_path:
        with open(save_pickle_path, "wb") as f:
            pickle.dump(total_noticias_columnistas, f)

    return total_noticias_columnistas


def load_columnistas_links_pickle(path: str) -> list[str]:
    with open(path, "rb") as f:
        return list(pickle.load(f))


def scrape_columnistas_from_listing_urls(
    listing_urls: Iterable[str] = DEFAULT_COLUMNISTAS,
    max_clicks: int = 80,
    wait_secs: int = 12,
    max_workers: int = 12,
    sleep_jitter: float = 0.0,
    save_links_pickle_path: str | None = None,
):
    """Obtiene links desde listados y luego scrapea el contenido de columnistas."""
    links = collect_columnistas_links(
        columnistas_urls=listing_urls,
        max_clicks=max_clicks,
        wait_secs=wait_secs,
        save_pickle_path=save_links_pickle_path,
    )
    noticias_columnistas = scrapping_columnistas(links, max_workers=max_workers, sleep_jitter=sleep_jitter)
    return noticias_columnistas, links


__all__ = [
    "BAD_SUBSTR",
    "DEFAULT_COLUMNISTAS",
    "_is_article",
    "_normalize",
    "get_listing_article_links_from_html",
    "get_links_after_clicks",
    "_safe_scrape",
    "scrapping_columnistas",
    "collect_columnistas_links",
    "load_columnistas_links_pickle",
    "scrape_columnistas_from_listing_urls",
]

"""Funciones de scraping de una noticia individual de El Colombiano (extraídas del notebook)."""
from __future__ import annotations

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

from src.scraping.elcolombiano.scrapping_one_new import _clean, _join_paragraphs, scrape_noticia

__all__ = ["_clean", "_join_paragraphs", "scrape_noticia"]

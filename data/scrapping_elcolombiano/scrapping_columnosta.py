"""Funciones de scraping de columnas en El Colombiano (extraídas del notebook)."""
from __future__ import annotations

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

from src.scraping.elcolombiano.scrapping_columnosta import (
    columnistas,
    _clean,
    _join_paragraphs,
    scrape_columnista,
)

__all__ = ["columnistas", "_clean", "_join_paragraphs", "scrape_columnista"]

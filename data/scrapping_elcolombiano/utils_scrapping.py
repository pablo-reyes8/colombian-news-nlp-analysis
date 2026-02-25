"""Utilidades de extracción de links para El Colombiano + helpers del pipeline del notebook."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

from src.scraping.elcolombiano.utils_scrapping import (
    SOCIAL_HOSTS,
    BAD_SUBSTR,
    BAD_PATH_PREFIX,
    GOOD_SECTIONS,
    ART_ID_REGEX,
    _normalize,
    _is_internal_or_elcolombiano,
    _looks_like_article,
    get_all_news_links,
)

DEFAULT_SECTION_URLS = [
    "https://www.elcolombiano.com/medellin",
    "https://www.elcolombiano.com/colombia",
    "https://www.elcolombiano.com/negocios",
    "https://www.elcolombiano.com/deportes",
    "https://www.elcolombiano.com/antioquia",
    "https://www.elcolombiano.com/cultura",
    "https://www.elcolombiano.com/tendencias",
    "https://www.elcolombiano.com/tecnologia",
    "https://www.elcolombiano.com/entretenimiento",
    "https://www.elcolombiano.com/colombia/politica",
    "https://www.elcolombiano.com/colombia/salud",
    "https://www.elcolombiano.com/colombia/educacion",
    "https://www.elcolombiano.com/colombia/paz-y-derechos-humanos",
    "https://www.elcolombiano.com/negocios/empresas",
    "https://www.elcolombiano.com/negocios/finanzas",
    "https://www.elcolombiano.com/negocios/agro",
    "https://www.elcolombiano.com/deportes/futbol",
    "https://www.elcolombiano.com/deportes/formula-1",
    "https://www.elcolombiano.com/deportes/atletico-nacional",
    "https://www.elcolombiano.com/deportes/independiente-medellin",
    "https://www.elcolombiano.com/antioquia/seguridad",
    "https://www.elcolombiano.com/antioquia/movilidad",
    "https://www.elcolombiano.com/antioquia/obras",
    "https://www.elcolombiano.com/cultura/cine",
    "https://www.elcolombiano.com/cultura/literatura",
    "https://www.elcolombiano.com/cultura/musica",
    "https://www.elcolombiano.com/cultura/mascotas",
    "https://www.elcolombiano.com/tecnologia/ciencia",
    "https://www.elcolombiano.com/entretenimiento/motores",
    "https://www.elcolombiano.com/entretenimiento/turismo",
    "https://www.elcolombiano.com/medio-ambiente",
    "https://www.elcolombiano.com/tecnologia/gadgets",
    "https://www.elcolombiano.com/tecnologia/videojuegos",
    "https://www.elcolombiano.com/tecnologia/aplicaciones",
    "https://www.elcolombiano.com/redes-sociales/trending-topic",
    "https://www.elcolombiano.com/entretenimiento/moda",
    "https://www.elcolombiano.com/entretenimiento/farandula",
    "https://www.elcolombiano.com/entretenimiento/television",
]


def build_links_tuplas(section_urls: Iterable[str] = DEFAULT_SECTION_URLS) -> list[tuple[str, str]]:
    """Construye pares (url_seccion, categoria) como en el notebook."""
    return [(link, str(link).rstrip("/").split("/")[-1]) for link in section_urls]


def collect_links_from_sections(
    links_tuplas: Iterable[tuple[str, str]],
    strict: bool = True,
    deduplicate: bool = True,
) -> list[tuple[str, str]]:
    """Extrae links de noticias para múltiples secciones y añade su categoría."""
    links_totales: list[tuple[str, str]] = []
    for page_url, categoria in links_tuplas:
        temp_links = get_all_news_links(page_url, strict=strict)
        links_totales.extend((noticia, categoria) for noticia in temp_links)

    if not deduplicate:
        return links_totales

    seen: set[tuple[str, str]] = set()
    unique_links: list[tuple[str, str]] = []
    for pair in links_totales:
        if pair not in seen:
            seen.add(pair)
            unique_links.append(pair)
    return unique_links


def collect_links_from_default_sections(strict: bool = True, deduplicate: bool = True) -> list[tuple[str, str]]:
    return collect_links_from_sections(build_links_tuplas(), strict=strict, deduplicate=deduplicate)


__all__ = [
    "SOCIAL_HOSTS",
    "BAD_SUBSTR",
    "BAD_PATH_PREFIX",
    "GOOD_SECTIONS",
    "ART_ID_REGEX",
    "_normalize",
    "_is_internal_or_elcolombiano",
    "_looks_like_article",
    "get_all_news_links",
    "DEFAULT_SECTION_URLS",
    "build_links_tuplas",
    "collect_links_from_sections",
    "collect_links_from_default_sections",
]

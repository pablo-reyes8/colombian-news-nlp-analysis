"""Pipeline de scraping de El Espectador extraído del notebook (funciones + configuración)."""
from __future__ import annotations

import os
from pathlib import Path
import re
import time
from typing import Iterable, Mapping

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


# -----------------------------------------------------------------------------
# Configuración de categorías (notebook)
# -----------------------------------------------------------------------------
CATEGORIAS_ELESPECTADOR_MENU: dict[str, str] = {
    "Política": "https://www.elespectador.com/politica/?utm_source=interno&utm_medium=boton&utm_campaign=menu_hamburguesa&utm_content=boton_menu_hamburguesa",
    "Judicial": "https://www.elespectador.com/judicial/?utm_source=interno&utm_medium=boton&utm_campaign=menu_hamburguesa&utm_content=boton_menu_hamburguesa",
    "Economía": "https://www.elespectador.com/economia/?utm_source=interno&utm_medium=boton&utm_campaign=menu_hamburguesa&utm_content=boton_menu_hamburguesa",
    "Mundo": "https://www.elespectador.com/mundo/?utm_source=interno&utm_medium=boton&utm_campaign=menu_hamburguesa&utm_content=boton_menu_hamburguesa",
    "Bogotá": "https://www.elespectador.com/bogota/?utm_source=interno&utm_medium=boton&utm_campaign=menu_hamburguesa&utm_content=boton_menu_hamburguesa",
    "Entretenimiento": "https://www.elespectador.com/entretenimiento/?utm_source=interno&utm_medium=boton&utm_campaign=menu_hamburguesa&utm_content=boton_menu_hamburguesa",
    "Deportes": "https://www.elespectador.com/deportes/?utm_source=interno&utm_medium=boton&utm_campaign=menu_hamburguesa&utm_content=boton_menu_hamburguesa",
    "Colombia": "https://www.elespectador.com/colombia/?utm_source=interno&utm_medium=boton&utm_campaign=menu_hamburguesa&utm_content=boton_menu_hamburguesa",
    "El Magazín Cultural": "https://www.elespectador.com/el-magazin-cultural/?utm_source=interno&utm_medium=boton&utm_campaign=menu_hamburguesa&utm_content=boton_menu_hamburguesa",
    "Salud": "https://www.elespectador.com/salud/?utm_source=interno&utm_medium=boton&utm_campaign=menu_hamburguesa&utm_content=boton_menu_hamburguesa",
    "Ambiente": "https://www.elespectador.com/ambiente/?utm_source=interno&utm_medium=boton&utm_campaign=menu_hamburguesa&utm_content=boton_menu_hamburguesa",
    "Investigación": "https://www.elespectador.com/investigacion/?utm_source=interno&utm_medium=boton&utm_campaign=menu_hamburguesa&utm_content=boton_menu_hamburguesa",
    "Educación": "https://www.elespectador.com/educacion/?utm_source=interno&utm_medium=boton&utm_campaign=menu_hamburguesa&utm_content=boton_menu_hamburguesa",
    "Ciencia": "https://www.elespectador.com/ciencia/?utm_source=interno&utm_medium=boton&utm_campaign=menu_hamburguesa&utm_content=boton_menu_hamburguesa",
    "Género y Diversidad": "https://www.elespectador.com/genero-y-diversidad/?utm_source=interno&utm_medium=boton&utm_campaign=menu_hamburguesa&utm_content=boton_menu_hamburguesa",
    "Tecnología": "https://www.elespectador.com/tecnologia/?utm_source=interno&utm_medium=boton&utm_campaign=menu_hamburguesa&utm_content=boton_menu_hamburguesa",
    "Actualidad": "https://www.elespectador.com/actualidad/?utm_source=interno&utm_medium=boton&utm_campaign=menu_hamburguesa&utm_content=boton_menu_hamburguesa",
    "Reportajes": "https://www.elespectador.com/reportajes/?utm_source=interno&utm_medium=boton&utm_campaign=menu_hamburguesa&utm_content=boton_menu_hamburguesa",
}

CATEGORIAS_ELESPECTADOR_ARCHIVO: dict[str, str] = {
    "Política": "https://www.elespectador.com/archivo/politica/",
    "Judicial": "https://www.elespectador.com/archivo/judicial/",
    "Economía": "https://www.elespectador.com/archivo/economia/",
    "Mundo": "https://www.elespectador.com/archivo/mundo/",
    "Bogotá": "https://www.elespectador.com/archivo/bogota/",
    "Entretenimiento": "https://www.elespectador.com/archivo/entretenimiento/",
    "Deportes": "https://www.elespectador.com/archivo/deportes/",
    "Colombia": "https://www.elespectador.com/archivo/colombia/",
    "El Magazín Cultural": "https://www.elespectador.com/archivo/el-magazin-cultural/",
    "Salud": "https://www.elespectador.com/archivo/salud/",
    "Ambiente": "https://www.elespectador.com/archivo/ambiente/",
    "Investigación": "https://www.elespectador.com/archivo/investigacion/",
    "Educación": "https://www.elespectador.com/archivo/educacion/",
    "Ciencia": "https://www.elespectador.com/archivo/ciencia/",
    "Género y Diversidad": "https://www.elespectador.com/archivo/genero-y-diversidad/",
    "Tecnología": "https://www.elespectador.com/archivo/tecnologia/",
    "Actualidad": "https://www.elespectador.com/archivo/actualidad/",
    "Reportajes": "https://www.elespectador.com/archivo/reportajes/",
}


# -----------------------------------------------------------------------------
# Función base del notebook (autocontenida)
# -----------------------------------------------------------------------------
def extract_article_data(nav, url, num_paginas=2):
    """
    Scrapea noticias de un archivo de El Espectador, navegando a través de las páginas.
    """
    nav.get(url)
    data = []

    for page_num in range(num_paginas):
        print(f"Scraping página {page_num + 1}...")

        try:
            WebDriverWait(nav, 20).until(
                EC.presence_of_element_located((By.ID, "sectionLayout"))
            )

            ini = nav.find_element(By.ID, "sectionLayout")
            ini2 = ini.find_element(By.ID, "main-layout-12-13")
            bloques = ini2.find_elements(By.XPATH, ".//div[contains(@class, 'Card-HomeEE_lateral')]")

            if not bloques:
                print("🔎 No se encontraron bloques de noticias en esta página. Terminando.")
                break

            links_to_visit = []
            for bloque in bloques:
                links_elements = bloque.find_elements(By.XPATH, ".//h2[contains(@class, 'Card-Title_xs')]/a")
                for link_elem in links_elements:
                    links_to_visit.append(link_elem.get_attribute("href"))

            for href in links_to_visit:
                try:
                    nav.execute_script("window.open(arguments[0]);", href)
                    nav.switch_to.window(nav.window_handles[1])

                    WebDriverWait(nav, 30).until(
                        EC.presence_of_element_located((By.XPATH, "//h1[contains(@class, 'ArticleHeader-Title')]"))
                    )

                    titulo = nav.find_element(By.XPATH, "//h1[contains(@class, 'ArticleHeader-Title')]").text

                    fecha = ""
                    fecha_xpaths = [
                        "//time[@datetime]", "//time[contains(@class, 'date')]",
                        "//meta[@property='article:published_time']", "//meta[@name='cXenseParse:recs:publishtime']",
                        "//div[contains(@class, 'ArticleHeader-Date')]", "//span[contains(@class, 'ArticleHeader-Date')]",
                        "//div[contains(@class, 'Datetime')]", "//div[contains(@class, 'VideoHeader-Date')]",
                        "//span[contains(@class, 'date')]", "//p[contains(@class, 'PublishedDate')]"
                    ]
                    for xp in fecha_xpaths:
                        try:
                            elem = nav.find_element(By.XPATH, xp)
                            if elem.tag_name == "meta":
                                fecha = elem.get_attribute("content")
                            elif elem.tag_name == "time":
                                fecha = elem.get_attribute("datetime") or elem.text.strip()
                            else:
                                fecha = elem.text.strip()
                            if fecha:
                                break
                        except Exception:
                            continue

                    if not fecha:
                        html = nav.page_source
                        patrones = [
                            r"\d{1,2}\s+de\s+[a-zA-Záéíóúñ]+\s+de\s+\d{4}",
                            r"\d{4}-\d{2}-\d{2}",
                            r"[A-Z][a-z]{2,8}\s+\d{1,2},\s+\d{4}",
                        ]
                        for patron in patrones:
                            match = re.search(patron, html)
                            if match:
                                fecha = match.group(0)
                                break

                    try:
                        categoria = nav.find_element(By.XPATH, "//div[@class='' and string-length(text()) > 0]").text
                    except Exception:
                        categoria = ""
                    try:
                        hook = nav.find_element(By.XPATH, "//h2[contains(@class,'ArticleHeader-Hook')]/div").text
                    except Exception:
                        hook = ""
                    try:
                        parrafos = " ".join(
                            [p.text for p in nav.find_elements(By.XPATH, "//div[contains(@class,'ArticleBody-Content')]/p")]
                        )
                    except Exception:
                        parrafos = ""

                    cuerpo = (hook + " " + parrafos).strip()

                    data.append({
                        "Titulo": titulo,
                        "Link": href,
                        "Fecha": fecha,
                        "Categoria": categoria,
                        "Cuerpo": cuerpo,
                    })

                except (TimeoutException, WebDriverException) as e:
                    print(f"❌ Error al procesar el enlace: {href}. Error: {e}")
                except Exception as e:
                    print(f"❌ Error desconocido al procesar el enlace: {href}. Error: {e}")
                finally:
                    nav.close()
                    nav.switch_to.window(nav.window_handles[0])

        except (TimeoutException, WebDriverException) as e:
            print(f"⚠️ Ocurrió un error al cargar la página principal o los bloques de noticias: {e}")
            break
        except Exception as e:
            print(f"⚠️ Ocurrió un error inesperado en el bucle de la página: {e}")
            break

        try:
            WebDriverWait(nav, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//div[contains(@class, 'Pagination-Nav')]/a[text()='Siguiente']"))
            ).click()
            time.sleep(3)
        except TimeoutException:
            print("No se encontró el botón de 'Siguiente' o no es cliqueable. Fin de la paginación.")
            break
        except Exception:
            print("No hay más páginas disponibles o la paginación ha cambiado.")
            break

    return pd.DataFrame(data)


# -----------------------------------------------------------------------------
# Helpers de selección / driver
# -----------------------------------------------------------------------------
def build_driver(headless: bool = True) -> webdriver.Chrome:
    """Crea un driver Selenium similar al usado en el notebook."""
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    for candidate in [
        os.getenv("CHROME_BIN"),
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ]:
        if candidate and Path(candidate).exists():
            options.binary_location = candidate
            break
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)


def listar_categorias_elespectador(source: str = "archivo") -> dict[str, str]:
    source = source.lower().strip()
    if source == "archivo":
        return dict(CATEGORIAS_ELESPECTADOR_ARCHIVO)
    if source == "menu":
        return dict(CATEGORIAS_ELESPECTADOR_MENU)
    raise ValueError("source debe ser 'archivo' o 'menu'")


def resolver_categorias_elespectador(
    categorias: Iterable[str] | None = None,
    source: str = "archivo",
) -> dict[str, str]:
    disponibles = listar_categorias_elespectador(source=source)
    if categorias is None:
        return disponibles

    seleccionadas: dict[str, str] = {}
    faltantes: list[str] = []
    for categoria in categorias:
        if categoria in disponibles:
            seleccionadas[categoria] = disponibles[categoria]
        else:
            faltantes.append(categoria)

    if faltantes:
        raise KeyError(
            "Categorías no encontradas: " + ", ".join(faltantes) + f". Usa alguna de: {', '.join(disponibles.keys())}"
        )
    return seleccionadas


# -----------------------------------------------------------------------------
# Pipeline (replica del cuaderno)
# -----------------------------------------------------------------------------
def scrape_categoria_archivo(
    nav: webdriver.Chrome,
    categoria: str,
    url: str | None = None,
    num_paginas: int = 12,
) -> pd.DataFrame:
    target_url = url or CATEGORIAS_ELESPECTADOR_ARCHIVO[categoria]
    df_categoria = extract_article_data(nav, target_url, num_paginas=num_paginas)
    if not df_categoria.empty:
        df_categoria = df_categoria.copy()
    df_categoria["Categoria"] = categoria
    return df_categoria


def scrape_varias_categorias_archivo(
    categorias: Mapping[str, str],
    num_paginas: int = 12,
    sleep_secs: float = 3.0,
    nav: webdriver.Chrome | None = None,
    headless: bool = True,
) -> pd.DataFrame:
    own_driver = nav is None
    if own_driver:
        nav = build_driver(headless=headless)

    all_dataframes: list[pd.DataFrame] = []
    try:
        for categoria, url in categorias.items():
            print(f"\n--- Iniciando scraping de la categoría: {categoria} ---")
            df_categoria = scrape_categoria_archivo(nav, categoria=categoria, url=url, num_paginas=num_paginas)
            all_dataframes.append(df_categoria)
            if sleep_secs:
                time.sleep(sleep_secs)
    finally:
        if own_driver and nav is not None:
            nav.quit()

    if not all_dataframes:
        return pd.DataFrame()
    return pd.concat(all_dataframes, ignore_index=True)


def scrape_todas_categorias_archivo(
    categorias: Mapping[str, str] | None = None,
    num_paginas: int = 12,
    sleep_secs: float = 3.0,
    nav: webdriver.Chrome | None = None,
    headless: bool = True,
) -> pd.DataFrame:
    categorias_resueltas = dict(categorias or CATEGORIAS_ELESPECTADOR_ARCHIVO)
    return scrape_varias_categorias_archivo(
        categorias=categorias_resueltas,
        num_paginas=num_paginas,
        sleep_secs=sleep_secs,
        nav=nav,
        headless=headless,
    )


def run_elespectador_archivo_pipeline(
    categorias: Iterable[str] | None = None,
    num_paginas: int = 12,
    sleep_secs: float = 3.0,
    headless: bool = True,
) -> pd.DataFrame:
    categorias_map = resolver_categorias_elespectador(categorias=categorias, source="archivo")
    return scrape_varias_categorias_archivo(
        categorias=categorias_map,
        num_paginas=num_paginas,
        sleep_secs=sleep_secs,
        headless=headless,
    )


# -----------------------------------------------------------------------------
# Persistencia
# -----------------------------------------------------------------------------
def guardar_dataframe_elespectador(df: pd.DataFrame, path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".xlsx":
        df.to_excel(output, index=False)
    elif suffix == ".csv":
        df.to_csv(output, index=False)
    elif suffix == ".parquet":
        df.to_parquet(output, index=False)
    elif suffix == ".json":
        df.to_json(output, orient="records", force_ascii=False, indent=2)
    else:
        raise ValueError("Extensión no soportada. Usa .xlsx, .csv, .parquet o .json")


__all__ = [
    "CATEGORIAS_ELESPECTADOR_MENU",
    "CATEGORIAS_ELESPECTADOR_ARCHIVO",
    "build_driver",
    "listar_categorias_elespectador",
    "resolver_categorias_elespectador",
    "extract_article_data",
    "scrape_categoria_archivo",
    "scrape_varias_categorias_archivo",
    "scrape_todas_categorias_archivo",
    "run_elespectador_archivo_pipeline",
    "guardar_dataframe_elespectador",
]

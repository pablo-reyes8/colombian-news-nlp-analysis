"""Pipeline de scraping de Semana extraído del notebook (autocontenido)."""
from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Iterable, Literal

import bs4
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


logger = logging.getLogger()
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Funciones base del notebook (autocontenidas)
# -----------------------------------------------------------------------------
def get_content_from_news(text: str, tag_ls: list, category: str, url: str) -> dict:
  content_dict = {}
  for tag in tag_ls:
    tag_pattern = r'\"{}\": \"(.*?)\",'.format(tag)
    if len(re.findall(tag_pattern, text)) != 0:
      content_dict[tag] = re.findall(tag_pattern, text)[0]
      logger.info(f"Successfully added: {tag}: {content_dict[tag]}")
    else:
      logger.exception(f"Error getting: {tag}")
      content_dict[tag] = None
  content_dict["scrappedCategory"] = category
  content_dict["sourceUrl"] = url
  return content_dict


def get_selenium_driver() -> webdriver:
  chrome_options = Options()
  chrome_options.add_argument("--headless=new")
  chrome_options.add_argument("--no-sandbox")
  chrome_options.add_argument("--disable-dev-shm-usage")
  chrome_options.add_argument("--disable-blink-features=AutomationControlled")
  for candidate in [
      os.getenv("CHROME_BIN"),
      "/usr/bin/google-chrome",
      "/usr/bin/chromium",
      "/usr/bin/chromium-browser",
  ]:
    if candidate and os.path.exists(candidate):
      chrome_options.binary_location = candidate
      break
  driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
  return driver


def access_to_Semana_category_news_catalog(
    driver: webdriver,
    category: str,
    base_Semana_url: str = "https://www.semana.com/",
    ver_mas_clicks_num: int = 10) -> str:

  web_route = str(base_Semana_url) + str(category) + "/"
  print(web_route)
  driver.get(web_route)
  logger.info('Waiting 20 seconds')
  time.sleep(20)

  try:
    recibir_notification_button = driver.find_element(By.ID, "onesignal-slidedown-cancel-button")
    recibir_notification_button.click()
    logger.info('Successfully clicked "recibir notification" button')
  except Exception:
    logger.info('No "recibir notification" button found')
  time.sleep(1)

  try:
    consent_use_personal_data_button = driver.find_element(By.CLASS_NAME, "fc-button-label")
    consent_use_personal_data_button.click()
    logger.info('Successfully clicked "consent use personal data" button')
  except Exception:
    logger.info('No "consent use personal data" button found')
  time.sleep(1)

  button_flag = False
  for n in range(ver_mas_clicks_num):
    botton_name = 'Más Columnas' if category == 'opinion' else 'Más contenido'

    try:
      if not button_flag:
        ver_mas_link = WebDriverWait(driver, 10).until(
          EC.element_to_be_clickable((By.XPATH, f"//a[text()='{botton_name}']"))
        )
        driver.execute_script("arguments[0].click();", ver_mas_link)
        button_type = 'executing script with <a>'
      else:
        ver_mas_link = driver.find_element(By.XPATH, f"//button[text()='{botton_name}']")
        driver.execute_script("arguments[0].click();", ver_mas_link)
        button_type = 'executing script with <button>'

      logger.info(f"[{n+1}] Successfully clicked '{botton_name}' in {web_route} by {button_type}")
    except Exception:
      logger.error(f"[{n+1}] Error clicking '{botton_name}' in {web_route}")
      button_flag = True
      if n > 5:
        logger.info(f"Stopping clicking '{botton_name}' in {web_route}")
        break

    time.sleep(1)

  return driver.page_source


def get_rendered_screenshot(driver: webdriver) -> None:
  driver.get_screenshot_as_file("screen.png")
  img = mpimg.imread("screen.png")
  fig, ax = plt.subplots(figsize=(10, 7))
  ax.imshow(img)
  ax.axis("off")
  return None


def get_news_urls_from_catalog(
    web_text: str,
    pattern_catalog_canonical: str,
    pattern_in_catalog_soup: str,) -> list:

  try:
    news_urls_by_canonical_regex = re.findall(pattern_catalog_canonical, web_text, flags=re.DOTALL)
    logger.info(f"Successfully news urls from catalog using canonical regex: {len(news_urls_by_canonical_regex)}")
  except Exception:
    news_urls_by_canonical_regex = []
    logger.exception("Error getting news urls from catalog using canonical regex")

  soup = bs4.BeautifulSoup(web_text)
  news_others_urls_by_bs4_regex = []
  for class_ in [
        'card-title h4',
        'styles__Top2Titulo-sc-1mj7fj3-4 eeZjUb',
        'mb-5 flex gap-[10px] border-b border-solid border-[#ddd] pb-5 md:gap-4 lg:mb-7 lg:border-none lg:pb-0',
        'col-span-1',
        'lg:col-span-2',
        'card-media',]:
    try:
      other_urls_soup = soup.find_all(class_=class_)
      other_bs4_pattern = pattern_in_catalog_soup.split(' ')[0] if class_ != 'card-title h4' else pattern_in_catalog_soup
      news_others_urls_by_bs4_regex_ = [
          re.findall(other_bs4_pattern, str(html_text))[0]
          for html_text in other_urls_soup
          if len(re.findall(other_bs4_pattern, str(html_text))) != 0
      ]
      logger.info(f"Successfully news urls from catalog using bs4: {len(news_others_urls_by_bs4_regex)} (class = {class_})")
    except Exception:
      news_others_urls_by_bs4_regex_ = []
      logger.exception(f"Error getting news urls from catalog using bs4 (class = {class_})")

    news_others_urls_by_bs4_regex += news_others_urls_by_bs4_regex_

  initial_num_urls = len(news_urls_by_canonical_regex) + len(news_others_urls_by_bs4_regex)
  news_urls_ls = list(set(news_urls_by_canonical_regex + news_others_urls_by_bs4_regex))
  final_num_urls = len(news_urls_ls)

  if initial_num_urls == 0:
    logger.warning("No news urls obtained from catalog")
  else:
    logger.info(f"Unique urls obtained: {final_num_urls} [{100*final_num_urls/initial_num_urls:.2f}% of total extraction]")
  return news_urls_ls


def get_valid_news_urls(news_urls_ls: list, base_Semana_url) -> list:
  valid_news_urls_ls = []
  for url in news_urls_ls:
    if not url.startswith(base_Semana_url):
      valid_news_urls_ls.append(base_Semana_url + url)
    else:
      valid_news_urls_ls.append(url)
  return valid_news_urls_ls


def get_news_content_from_url_ls(
    valid_news_urls_web: list,
    trim_pattern: str,
    tag_ls: list,
    category: str) -> list:

  news_scrapped_content_ls = []
  for index, url in enumerate(valid_news_urls_web, start=1):
    logger.info(f"[{index}/{len(valid_news_urls_web)}]")
    try:
      web = requests.get(url)
    except Exception:
      logger.exception(f"Connection error when trying to access to: {url}")
      continue

    try:
      content_dict = get_content_from_news(web.text, tag_ls, category, url)
    except Exception:
      logger.exception(f"Error getting content from: {url}")
      continue

    news_scrapped_content_ls.append(content_dict)
    logger.info(f"Successfully scrapped content from: {url}")
    time.sleep(1)

  return news_scrapped_content_ls


def run_scrapping_process(
    category: str,
    num_of_ver_mas_clicks: int,
    trim_pattern: str,
    pattern_catalog_canonical: str,
    pattern_catalog_soup: str,
    tag_ls: list,
    base_Semana_url: str,
    drive_path: str,):

  driver = get_selenium_driver()
  try:
    catalog_web = access_to_Semana_category_news_catalog(driver, category, base_Semana_url, num_of_ver_mas_clicks)
    news_urls_web = get_news_urls_from_catalog(catalog_web, pattern_catalog_canonical, pattern_catalog_soup)
    valid_news_urls_web = get_valid_news_urls(news_urls_web, base_Semana_url)
    news_content_ls = get_news_content_from_url_ls(valid_news_urls_web, trim_pattern, tag_ls, category)
  finally:
    driver.quit()

  df = pd.DataFrame(news_content_ls)
  category_ = str(category).replace("/", "-")
  Path(drive_path).mkdir(parents=True, exist_ok=True)
  df.to_parquet(f"{drive_path}/semana_{category_}_news.parquet")
  logger.info(f"Successfully saved parquet table for: '{category}' in {drive_path}")
  return None


def get_news_df(
    drive_folder_path: str,
    original_published_date_col: str = 'datePublished',
    formatted_published_date_col: str = 'datePublishedFormatted',
    initial_published_date: str = '2025-08-01',
    final_published_date: str = '2025-08-30',
    filter_df_by_dates_flag: bool = True,) -> pd.DataFrame:

  extracted_news_ls = os.listdir(drive_folder_path)
  news_ls = []
  for file in extracted_news_ls:
    news_ls.append(pd.read_parquet(os.path.join(drive_folder_path, file)))

  df = pd.concat(news_ls)
  df = df[~df[original_published_date_col].isnull()]
  df.loc[:, formatted_published_date_col] = (
      pd.to_datetime(
          df[original_published_date_col].astype(str).apply(lambda x: x.replace('T', ' '))
      ).dt.strftime('%Y-%m-%d')
  )
  if filter_df_by_dates_flag:
    df = df[
        (df[formatted_published_date_col] >= initial_published_date) &
        (df[formatted_published_date_col] <= final_published_date)
    ]

  return df


# -----------------------------------------------------------------------------
# Configuración del notebook
# -----------------------------------------------------------------------------
DEFAULT_BASE_SEMANA_URL = "https://www.semana.com/"

COMPLETE_SEMANA_CATALOG = [
    "nacion", "politica", "economia", "deportes", "tecnologia", "semana-tv", "tv",
    "semanaplay", "salud", "opinion", "educacion", "cultura", "loterias", "mundo",
    "turismo", "como", "vehiculos", "finanzas", "sostenible", "confidenciales",
    "especiales", "gente", "actualidad", "foros-semana/foros-anteriores", "mejor-colombia",
    "mujeres", "hablan-las-marcas", "semanarural", "fotos", "emprendimientos", "impresa", "empleos",
]
DISCARDED_SEMANA_CATALOG = [
    "fotos", "emprendimientos", "impresa", "empleos", "mujeres", "semanarural",
    "semanaplay", "opinion", "mejor-colombia", "economia",
]
DEFAULT_SEMANA_CATALOG = sorted(list(set(COMPLETE_SEMANA_CATALOG) - set(DISCARDED_SEMANA_CATALOG)))
DEFAULT_PATTERN_CATALOG_SOUP = r'href=\"(.+?)\" (?:target=\"_self\" )?rel=\"noopener\"'
DEFAULT_PATTERN_CATALOG_CANONICAL = r'\"canonical_url\":\"(.+?)\"'
DEFAULT_TRIM_PATTERN = r'\<script type=\"application\/ld\+json\"\>(.*?)\<\/script\>'
DEFAULT_TAGS = ["headline", "articleSection", "datePublished", "articleBody"]
DEFAULT_NUM_VER_MAS_CLICKS = 60


# -----------------------------------------------------------------------------
# Helpers de catálogo / configuración
# -----------------------------------------------------------------------------
def listar_catalogo_semana(modo: Literal["default", "complete", "discarded"] = "default") -> list[str]:
    if modo == "default":
        return list(DEFAULT_SEMANA_CATALOG)
    if modo == "complete":
        return list(COMPLETE_SEMANA_CATALOG)
    if modo == "discarded":
        return list(DISCARDED_SEMANA_CATALOG)
    raise ValueError("modo debe ser 'default', 'complete' o 'discarded'")


def resolver_catalogo_semana(categories: Iterable[str] | None = None) -> list[str]:
    if categories is None:
        return list(DEFAULT_SEMANA_CATALOG)
    disponibles = set(COMPLETE_SEMANA_CATALOG)
    categories_list = list(categories)
    faltantes = [c for c in categories_list if c not in disponibles]
    if faltantes:
        raise KeyError(
            "Categorías de Semana no reconocidas: " + ", ".join(faltantes) + ". Revisa COMPLETE_SEMANA_CATALOG."
        )
    return categories_list


def build_semana_notebook_config(
    drive_path: str,
    categories: Iterable[str] | None = None,
    base_Semana_url: str = DEFAULT_BASE_SEMANA_URL,
    pattern_catalog_soup: str = DEFAULT_PATTERN_CATALOG_SOUP,
    pattern_catalog_canonical: str = DEFAULT_PATTERN_CATALOG_CANONICAL,
    trim_pattern: str = DEFAULT_TRIM_PATTERN,
    tag_ls: list[str] | None = None,
    num_of_ver_mas_clicks: int = DEFAULT_NUM_VER_MAS_CLICKS,
) -> dict:
    resolved_categories = resolver_catalogo_semana(categories)
    return {
        "drive_path": drive_path,
        "base_Semana_url": base_Semana_url,
        "semana_catalog_ls": resolved_categories,
        "pattern_catalog_soup": pattern_catalog_soup,
        "pattern_catalog_canonical": pattern_catalog_canonical,
        "trim_pattern": trim_pattern,
        "tag_ls": list(tag_ls) if tag_ls is not None else list(DEFAULT_TAGS),
        "num_of_ver_mas_clicks": num_of_ver_mas_clicks,
    }


# -----------------------------------------------------------------------------
# Pipeline (replica del loop del notebook)
# -----------------------------------------------------------------------------
def run_semana_catalog_pipeline(
    drive_path: str,
    categories: Iterable[str] | None = None,
    base_Semana_url: str = DEFAULT_BASE_SEMANA_URL,
    pattern_catalog_soup: str = DEFAULT_PATTERN_CATALOG_SOUP,
    pattern_catalog_canonical: str = DEFAULT_PATTERN_CATALOG_CANONICAL,
    trim_pattern: str = DEFAULT_TRIM_PATTERN,
    tag_ls: list[str] | None = None,
    num_of_ver_mas_clicks: int = DEFAULT_NUM_VER_MAS_CLICKS,
) -> list[str]:
    semana_catalog_ls = resolver_catalogo_semana(categories)
    tags = list(tag_ls) if tag_ls is not None else list(DEFAULT_TAGS)

    Path(drive_path).mkdir(parents=True, exist_ok=True)
    processed: list[str] = []
    for category in semana_catalog_ls:
        run_scrapping_process(
            category=category,
            num_of_ver_mas_clicks=num_of_ver_mas_clicks,
            trim_pattern=trim_pattern,
            pattern_catalog_canonical=pattern_catalog_canonical,
            pattern_catalog_soup=pattern_catalog_soup,
            tag_ls=tags,
            base_Semana_url=base_Semana_url,
            drive_path=drive_path,
        )
        logger.info(f'Finished scrapping process for category: "{category}"')
        processed.append(category)
    return processed


def consolidar_noticias_semana(
    drive_folder_path: str,
    original_published_date_col: str = "datePublished",
    formatted_published_date_col: str = "datePublishedFormatted",
    initial_published_date: str = "2025-08-01",
    final_published_date: str = "2025-08-30",
    filter_df_by_dates_flag: bool = True,
) -> pd.DataFrame:
    return get_news_df(
        drive_folder_path=drive_folder_path,
        original_published_date_col=original_published_date_col,
        formatted_published_date_col=formatted_published_date_col,
        initial_published_date=initial_published_date,
        final_published_date=final_published_date,
        filter_df_by_dates_flag=filter_df_by_dates_flag,
    )


def run_semana_notebook_pipeline_and_consolidate(
    drive_path: str,
    categories: Iterable[str] | None = None,
    base_Semana_url: str = DEFAULT_BASE_SEMANA_URL,
    pattern_catalog_soup: str = DEFAULT_PATTERN_CATALOG_SOUP,
    pattern_catalog_canonical: str = DEFAULT_PATTERN_CATALOG_CANONICAL,
    trim_pattern: str = DEFAULT_TRIM_PATTERN,
    tag_ls: list[str] | None = None,
    num_of_ver_mas_clicks: int = DEFAULT_NUM_VER_MAS_CLICKS,
    initial_published_date: str = "2025-08-01",
    final_published_date: str = "2025-08-30",
    filter_df_by_dates_flag: bool = True,
) -> pd.DataFrame:
    run_semana_catalog_pipeline(
        drive_path=drive_path,
        categories=categories,
        base_Semana_url=base_Semana_url,
        pattern_catalog_soup=pattern_catalog_soup,
        pattern_catalog_canonical=pattern_catalog_canonical,
        trim_pattern=trim_pattern,
        tag_ls=tag_ls,
        num_of_ver_mas_clicks=num_of_ver_mas_clicks,
    )
    return consolidar_noticias_semana(
        drive_folder_path=drive_path,
        initial_published_date=initial_published_date,
        final_published_date=final_published_date,
        filter_df_by_dates_flag=filter_df_by_dates_flag,
    )


# -----------------------------------------------------------------------------
# Persistencia
# -----------------------------------------------------------------------------
def guardar_dataframe_semana(df: pd.DataFrame, path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".csv":
        df.to_csv(output, index=False)
    elif suffix == ".parquet":
        df.to_parquet(output, index=False)
    elif suffix == ".json":
        df.to_json(output, orient="records", force_ascii=False, indent=2)
    else:
        raise ValueError("Extensión no soportada. Usa .csv, .parquet o .json")


__all__ = [
    # base notebook functions
    "logger",
    "get_content_from_news",
    "get_selenium_driver",
    "access_to_Semana_category_news_catalog",
    "get_rendered_screenshot",
    "get_news_urls_from_catalog",
    "get_valid_news_urls",
    "get_news_content_from_url_ls",
    "run_scrapping_process",
    "get_news_df",
    # config / helpers
    "DEFAULT_BASE_SEMANA_URL",
    "COMPLETE_SEMANA_CATALOG",
    "DISCARDED_SEMANA_CATALOG",
    "DEFAULT_SEMANA_CATALOG",
    "DEFAULT_PATTERN_CATALOG_SOUP",
    "DEFAULT_PATTERN_CATALOG_CANONICAL",
    "DEFAULT_TRIM_PATTERN",
    "DEFAULT_TAGS",
    "DEFAULT_NUM_VER_MAS_CLICKS",
    "listar_catalogo_semana",
    "resolver_catalogo_semana",
    "build_semana_notebook_config",
    # pipelines
    "run_semana_catalog_pipeline",
    "consolidar_noticias_semana",
    "run_semana_notebook_pipeline_and_consolidate",
    # persistencia
    "guardar_dataframe_semana",
]

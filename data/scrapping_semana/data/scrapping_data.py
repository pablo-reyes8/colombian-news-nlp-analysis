import requests
import re
import time
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import bs4
import os


logger = logging.getLogger()
logger.setLevel(logging.INFO)


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

  chrome_options.binary_location = "/usr/bin/google-chrome"

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
  #get_rendered_screenshot(driver)

  try:
    recibir_notification_button = driver.find_element(By.ID, "onesignal-slidedown-cancel-button")
    recibir_notification_button.click()
    logger.info('Successfully clicked "recibir notification" button')
  except:
    logger.info('No "recibir notification" button found')
  time.sleep(1)
  #get_rendered_screenshot(driver)

  try:
    consent_use_personal_data_button = driver.find_element(By.CLASS_NAME, "fc-button-label")
    consent_use_personal_data_button.click()
    logger.info('Successfully clicked "consent use personal data" button')
  except:
    logger.info('No "consent use personal data" button found')
  time.sleep(1)
  #get_rendered_screenshot(driver)

  button_flag = False
  for n in range(ver_mas_clicks_num):

    if category == 'opinion':
      botton_name = 'Más Columnas'
    else:
      botton_name = 'Más contenido'

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
    except:
      logger.error(f"[{n+1}] Error clicking '{botton_name}' in {web_route}")
      button_flag = True
      if (n > 5):
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

  # Canonical regex way
  try:
    news_urls_by_canonical_regex = re.findall(pattern_catalog_canonical, web_text, flags=re.DOTALL)
    logger.info(f"Successfully news urls from catalog using canonical regex: {len(news_urls_by_canonical_regex)}")
  except:
    news_urls_by_canonical_regex = []
    logger.exception("Error getting news urls from catalog using canonical regex")

  # Bs4 way
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
      other_urls_soup = soup.find_all(class_ = class_)
      other_bs4_pattern = pattern_catalog_soup.split(' ')[0] if class_ != 'card-title h4' else pattern_catalog_soup
      news_others_urls_by_bs4_regex_ = [
          re.findall(other_bs4_pattern, str(html_text))[0] for html_text in other_urls_soup if len(re.findall(other_bs4_pattern, str(html_text))) != 0
      ]
      logger.info(f"Successfully news urls from catalog using bs4: {len(news_others_urls_by_bs4_regex)} (class = {class_})")
    except:
      news_others_urls_by_bs4_regex_ = []
      logger.exception(f"Error getting news urls from catalog using bs4 (class = {class_})")

    news_others_urls_by_bs4_regex += news_others_urls_by_bs4_regex_

  # Union
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
    except:
      logger.exception(f"Connection error when trying to access to: {url}")
      continue

    try:
      content_dict = get_content_from_news(web.text, tag_ls, category, url)
    except:
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
  

  # Selenium driver
  driver = get_selenium_driver()

  # Getting news urls
  catalog_web = access_to_Semana_category_news_catalog(driver, category, base_Semana_url, num_of_ver_mas_clicks)
  news_urls_web = get_news_urls_from_catalog(catalog_web, pattern_catalog_canonical, pattern_catalog_soup)
  valid_news_urls_web = get_valid_news_urls(news_urls_web, base_Semana_url)

  # Getting scrapped content
  news_content_ls = get_news_content_from_url_ls(valid_news_urls_web, trim_pattern, tag_ls, category)

  # Write parquet table
  df = pd.DataFrame(news_content_ls)
  category_ = str(category).replace("/","-")
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

  extracted_news_ls = os.listdir(drive_path)

  news_ls = []
  for file in extracted_news_ls:
    news_ls.append(pd.read_parquet(drive_path + file))

  df = pd.concat(news_ls)
  df = df[~df[original_published_date_col].isnull()]
  df.loc[:,formatted_published_date_col] = (
      pd.to_datetime(
          df[original_published_date_col]
          .astype(str)
          .apply(lambda x: x.replace('T', ' '))
      ).dt.strftime('%Y-%m-%d')
  )
  if filter_df_by_dates_flag:
    df = df[
        (df[formatted_published_date_col] >= initial_published_date) &
        (df[formatted_published_date_col] <= final_published_date)
    ]

  return df
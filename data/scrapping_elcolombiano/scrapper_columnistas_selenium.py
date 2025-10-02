
import re 
import requests 
import BeautifulSoup
import json
import pickle
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, urlunparse
import re
import time
from data.scrapping_elcolombiano.scrapping_columnosta import *
from data.scrapping_elcolombiano.scrapping_one_new import *

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
from selenium.webdriver.chrome.service import Service
from requests.adapters import HTTPAdapter, Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from data.scrapping_elcolombiano.scrape_noticia import *


BAD_SUBSTR = ("share", "sharer.php", "dialog/send", "utm_", "redirect_uri=")

# Verificar si estamos trayendo una noticia de algun columnista o otra cosa 
def _is_article(url: str, base_host: str):
    p = urlparse(url)
    host = p.netloc.lower().removeprefix("www.")

    if not (host == base_host or host.endswith("." + base_host)):
        return False
    if any(s in url for s in BAD_SUBSTR):
        return False
    return True


# Normalizar los links 
def _normalize(href: str, listing_url: str):
    absu = urljoin(listing_url, href)
    p = urlparse(absu)
    netloc = p.netloc.lower().removeprefix("www.")
    p = p._replace(scheme="https", netloc=netloc, query="", fragment="")
    if p.path != "/" and p.path.endswith("/"):
        p = p._replace(path=p.path.rstrip("/"))
    return urlunparse(p)


# Con esta funcion extrameos todos los links de una pagina de columnista 
# Esto trae los links de todos los articulos de un columnista 
def get_listing_article_links_from_html(listing_url: str, html: str):

    base_host = urlparse(listing_url).netloc.lower().removeprefix("www.")
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    containers = soup.select("div.ec-teaser-noticia-seccion-metadatos, "
        "div.container-noticia-seccion-metadatos, "
        "div.text_container_noticia_metadato")

    for box in containers:
        a = None
        for cand in box.select("a[href]"):
            if cand.select_one("span.priority-content, h3, .title__link"):
                a = cand
                break
        if not a:
            a = box.find("a", href=True)
        if not a:
            continue
        url = _normalize(a["href"].strip(), listing_url)
        if _is_article(url, base_host):
            links.add(url)

    for a in soup.select("a[href]"):
        href = a["href"].strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        url = _normalize(href, listing_url)
        if _is_article(url, base_host) and a.select_one("span.priority-content, h3"):
            links.add(url)

    return sorted(links)


# Funcion que sirve para darle al boton de 'CARGAR MAS' en las paginas de los columnistas 
# y cuando no se puedan cargar mas noticias (no tiene mas ese columnista) se traen todos los 
# links de todas las noticias mediante la funcion de arriba
def get_links_after_clicks(listing_url: str, max_clicks: int = 80, wait_secs: int = 12):

    opts = webdriver.ChromeOptions()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)

    try:
        driver.get(listing_url)
        wait = WebDriverWait(driver, wait_secs)

        def count_cards():
            return len(driver.find_elements(By.CSS_SELECTOR, "article"))

        clicks = 0
        
        while clicks < max_clicks:
            try:
                btn = wait.until(EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "div.more-button[id$='_myMoreButton']")))
            except Exception:
                break  

            before = count_cards()
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
            time.sleep(0.2)
            driver.execute_script("arguments[0].click();", btn)

            try:
                wait.until(EC.invisibility_of_element_located(
                    (By.CSS_SELECTOR, "div[id$='_loadingDiv']")))
            except Exception:
                pass

            try:
                wait.until(lambda d: count_cards() > before)
                clicks += 1
                time.sleep(0.3)
            except Exception:
                break

        html = driver.page_source
    finally:
        driver.quit()

    return get_listing_article_links_from_html(listing_url, html)



# Hacer seguro el scrapping de las noticas de columnistas
def _safe_scrape(url, session):
    try:

        return scrape_columnista(url, session=session)
    
    except requests.RequestException as e:
        print(f"[WARN] fallo en {url}: {e}")
        return None

# En paralelo y de forma segura extraemos toda la informacion de las noticias de todos 
# los links de todos los columnistas del colombiano
def scrapping_columnistas(urls, max_workers=12, sleep_jitter=0.0):

    session = build_session()
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for u in urls:
            if sleep_jitter:
                time.sleep(sleep_jitter)
            futures.append(ex.submit(_safe_scrape, u, session))

        for idx, fut in enumerate(as_completed(futures), start=1):
            item = fut.result()
            if item is not None:
                results.append(item)
                
            if idx % 100 == 0:
                print(f"[INFO] Procesadas {idx} noticias de {len(urls)}")

    return results
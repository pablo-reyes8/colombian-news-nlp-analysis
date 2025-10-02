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
from data.scrapping_elcolombiano.utils_scrapping import *

# Crear sesion para hacer seguro el scrapping
def build_session():
    s = requests.Session()

    retries = Retry(total=3,             
        backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504),allowed_methods=("GET", "HEAD"))
    
    adapter = HTTPAdapter(max_retries=retries,
        pool_connections=100, pool_maxsize=100)
    
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    s.timeout = (5, 45)  
    return s

# Scrapear noticia por link de forma segura
def _safe_scrape_any(url, categoria, session, columnistas_urls) :

    try:
        is_col = ("columnistas" in url.lower()) or (columnistas_urls and url in columnistas_urls)
        if is_col:
            item = scrape_columnista(url, session=session)  

        else:
            item = scrape_noticia(url, session=session)

        if not item:
            return None
        item["categoria"] = categoria
    
        return item
    
    except requests.RequestException as e:
        print(f"[WARN] request error en {url}: {e}")
        return None
    
    except Exception as e:
        print(f"[WARN] parse error en {url}: {e}")
        return None
    

# Scrapear todas las noticias NO ESCRITAS por columnistas
def scrapping_all(l, columnistas = None):

    session = build_session()
    out = []
    total = len(l)
    for i, (url, cat) in enumerate(l, start=1):
        item = _safe_scrape_any(url, cat, session=session, columnistas_urls=columnistas)

        if item:
            out.append(item)

        if 50 and (i % 50 == 0):
            print(f"[INFO] Procesadas {i}/{total}")

    return out
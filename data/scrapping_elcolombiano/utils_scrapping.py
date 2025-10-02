import pandas as pd
import numpy as np

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

import pickle
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, urlunparse
import re


def _normalize(base_url: str, href: str):
    absu = urljoin(base_url, href)
    p = urlparse(absu)
    p = p._replace(query="", fragment="")
    return urlunparse(p)


def _is_internal_or_elcolombiano(href: str , SOCIAL_HOSTS):
    if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
        return False
    
    if href.startswith("/"):
        return True
    
    host = urlparse(href).netloc.lower()

    if any(host.endswith(s) for s in SOCIAL_HOSTS):
        return False
    
    return host.endswith("elcolombiano.com")


def _looks_like_article(url: str, be_strict: bool) :
    p = urlparse(url)
    path = p.path

    if any(path.startswith(bad) for bad in BAD_PATH_PREFIX):
        return False
    if any(s in url for s in BAD_SUBSTR):
        return False
    
    if be_strict:
        if ART_ID_REGEX.search(path):
            return True
        if path.startswith("/opinion/columnistas/") and path.count("/") >= 3:
            return True
        return False
    parts = [p for p in path.split("/") if p]

    if len(parts) >= 2:
        if parts[0] not in GOOD_SECTIONS:
            print("Nueva sección detectada:", parts[0])
        return True
    
    return False


def get_all_news_links(page_url: str, strict: bool = True):

    r = requests.get(page_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
    r.raise_for_status()
    r.encoding = "utf-8"
    soup = BeautifulSoup(r.text, "html.parser")

    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not _is_internal_or_elcolombiano(href):
            continue
        url = _normalize(page_url, href)
        if _looks_like_article(url, be_strict=strict):
            links.add(url)

    return sorted(links)




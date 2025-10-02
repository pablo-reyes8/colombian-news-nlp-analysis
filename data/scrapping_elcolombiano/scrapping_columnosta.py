import re 
import requests 
import BeautifulSoup
import json

columnistas= ['https://www.elcolombiano.com/cronologia/noticias/meta/diego-santos',
'https://www.elcolombiano.com/cronologia/noticias/meta/luis-diego-monsalve',
'https://www.elcolombiano.com/cronologia/noticias/meta/mauricio-perfetti-del-corral',
'https://www.elcolombiano.com/cronologia/noticias/meta/oscar-dominguez-giraldo',
'https://www.elcolombiano.com/cronologia/noticias/meta/rodrigo-botero-montoya' ,
'https://www.elcolombiano.com/cronologia/noticias/meta/isabel-gutierrez-ramirez',
'https://www.elcolombiano.com/cronologia/noticias/meta/humberto-montero',
'https://www.elcolombiano.com/cronologia/noticias/meta/daniel-duque',
'https://www.elcolombiano.com/cronologia/noticias/meta/alberto-velasquez-martinez',
'https://www.elcolombiano.com/cronologia/noticias/meta/johel-moreno-s-',
'https://www.elcolombiano.com/cronologia/noticias/meta/paola-holguin',
'https://www.elcolombiano.com/cronologia/noticias/meta/david-e-santos-gomez']

# Limpiar el texto
def _clean(t):
    if t is None:
        return None
    t = " ".join(t.split())
    return t or None

# Unir los parrafos
def _join_paragraphs(parrafos):
    parrafos = [_clean(p) for p in parrafos if _clean(p)]
    return " ".join(parrafos) if parrafos else None


# Scrapper para los articulos escritos por un columnista
def scrape_columnista(url, session = None, timeout=(5, 40)):

    sess = session or requests.Session()
    r = sess.get(url, timeout=timeout)
    r.raise_for_status()
    r.encoding = "utf-8"

    soup = BeautifulSoup(r.text, "lxml")

    # Hora 
    hora_el = soup.select_one(".hora-noticia, time.hora-noticia, .fecha, time[datetime]")
    hora = _clean(hora_el.get_text(strip=True)) if hora_el else None

    # Autor
    autor_el = soup.select_one(".texto-columnista p, .autor, .byline, .firma, .author , .nombre-columnista")
    autor = _clean(autor_el.get_text(strip=True)) if autor_el else None

    # Título 
    tit_el = soup.select_one(".priority-content, h1.titulo, h1.title, h1")
    titulo = _clean(tit_el.get_text(strip=True)) if tit_el else None

    # principal
    cuerpo_div = soup.select_one("div.texto-columnista div.text, .texto-columnista .text, article .texto, article .cuerpo")
    if cuerpo_div:
        pars = [p.get_text(" ", strip=True) for p in cuerpo_div.find_all("p")]
        cuerpo = _join_paragraphs(pars)
    else:
        # fallback: buscar párrafos del artículo
        pars = [p.get_text(" ", strip=True) for p in soup.select("article p")]
        cuerpo = _join_paragraphs(pars)

    return {"hora": hora,
        "titulo": titulo,"autor": autor, "cuerpo": cuerpo,
        'link': url}
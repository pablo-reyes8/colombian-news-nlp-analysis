import re 
import requests 
import BeautifulSoup
import json

# Limpieza básica 
def _clean(t) :
    if t is None:
        return None
    t = " ".join(t.split())
    return t or None

# Para unir los parrafos por separado de la pagina
def _join_paragraphs(parrafos):
    pars = [_clean(p) for p in parrafos if _clean(p)]
    filt = []
    for p in pars:
        if p and not re.search(r"Lea también|Puede interesarle|Siga leyendo|Podcast|Video", p, re.I):
            filt.append(p)
    return " ".join(filt) if filt else None


# Scraper noticia 
def scrape_noticia(url, session = None, timeout=(5, 40)):

    sess = session or requests.Session()
    r = sess.get(url, timeout=timeout)
    r.raise_for_status()
    r.encoding = "utf-8"

    soup = BeautifulSoup(r.text, "lxml")

    # Hora 
    hora = None
    hora_el = soup.select_one(".hora-noticia, time.hora-noticia, .fecha, time[datetime]")

    if hora_el:
        hora = _clean(hora_el.get_text(" ", strip=True))


    if not hora:
        for s in soup.select('script[type="application/ld+json"]'):
            try:
                data = json.loads(s.string or "")
                items = data if isinstance(data, list) else [data]
                for it in items:
                    if isinstance(it, dict):
                        if it.get("@type") in ("NewsArticle", "Article"):
                            hora = _clean(it.get("datePublished") or it.get("dateModified"))
                            if hora:
                                break
                if hora:
                    break
            except Exception:
                pass

    # Autor 
    autor = None
    autor_el = soup.select_one(".div_author_r_texto_, .autor, .author, .byline, .firma")
    if autor_el:
        raw_text = autor_el.get_text(" ", strip=True)
        candidatos = re.findall(r"[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)+", raw_text)
        autor = _clean(candidatos[0]) if candidatos else _clean(raw_text.split("hace")[0])

    if not autor:
        meta_author = soup.find("meta", attrs={"name": "author"}) or soup.find("meta", attrs={"property": "article:author"})
        if meta_author and meta_author.get("content"):
            autor = _clean(meta_author["content"])

    # Título 
    titulo = None
    tit_el = soup.select_one(".priority-content, h1.titulo, h1.title, h1.headline, h1")
    if tit_el:
        titulo = _clean(tit_el.get_text(" ", strip=True))
    if not titulo:
        og_title = soup.find("meta", attrs={"property": "og:title"})
        if og_title and og_title.get("content"):
            titulo = _clean(og_title["content"])

    # Cuerpo 
    cuerpo = None
    cuerpo_div = soup.select_one("div.block-text, article .texto, article .cuerpo, article .content, .article-body, .texto-noticia")
    if cuerpo_div:
        pars = [p.get_text(" ", strip=True) for p in cuerpo_div.find_all("p")]
        cuerpo = _join_paragraphs(pars)
    if not cuerpo:
        pars = [p.get_text(" ", strip=True) for p in soup.select("article p")]
        cuerpo = _join_paragraphs(pars)

    return {"hora": hora, "titulo": titulo, "autor": autor, "cuerpo": cuerpo, "url": url}
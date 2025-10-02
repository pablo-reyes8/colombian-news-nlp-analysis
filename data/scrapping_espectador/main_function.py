import sys
from selenium import webdriver
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
import time
import sys
import re
import pandas as pd
import sys
import re


def extract_article_data(nav, url, num_paginas=2):
    """
    Scrapea noticias de un archivo de El Espectador, navegando a través de las páginas.
    """
    nav.get(url)
    data = []

    for page_num in range(num_paginas):
        print(f"Scraping página {page_num + 1}...")

        try:
            # Espera explícita para asegurar que los bloques de noticias estén cargados
            WebDriverWait(nav, 20).until(
                EC.presence_of_element_located((By.ID, "sectionLayout"))
            )

            ini = nav.find_element(By.ID, "sectionLayout")
            ini2 = ini.find_element(By.ID, "main-layout-12-13")
            bloques = ini2.find_elements(By.XPATH, ".//div[contains(@class, 'Card-HomeEE_lateral')]")

            if not bloques:
                print("🔎 No se encontraron bloques de noticias en esta página. Terminando.")
                break

            # Recopilar todos los enlaces primero
            links_to_visit = []
            for bloque in bloques:
                links_elements = bloque.find_elements(By.XPATH, ".//h2[contains(@class, 'Card-Title_xs')]/a")
                for link_elem in links_elements:
                    links_to_visit.append(link_elem.get_attribute("href"))

            # Iterar sobre los enlaces y extraer datos
            for href in links_to_visit:
                try:
                    nav.execute_script("window.open(arguments[0]);", href)
                    nav.switch_to.window(nav.window_handles[1])

                    # Espera explícita y más robusta
                    WebDriverWait(nav, 30).until(
                        EC.presence_of_element_located((By.XPATH, "//h1[contains(@class, 'ArticleHeader-Title')]"))
                    )

                    titulo = nav.find_element(By.XPATH, "//h1[contains(@class, 'ArticleHeader-Title')]").text

                    # Extracción robusta de fecha (mantenido del código original)
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
                            if elem.tag_name == "meta": fecha = elem.get_attribute("content")
                            elif elem.tag_name == "time": fecha = elem.get_attribute("datetime") or elem.text.strip()
                            else: fecha = elem.text.strip()
                            if fecha: break
                        except: continue

                    if not fecha:
                        html = nav.page_source
                        patrones = [r"\d{1,2}\s+de\s+[a-zA-Záéíóúñ]+\s+de\s+\d{4}", r"\d{4}-\d{2}-\d{2}", r"[A-Z][a-z]{2,8}\s+\d{1,2},\s+\d{4}"]
                        for patron in patrones:
                            match = re.search(patron, html)
                            if match:
                                fecha = match.group(0)
                                break

                    # Extracción de otros datos
                    try: categoria = nav.find_element(By.XPATH, "//div[@class='' and string-length(text()) > 0]").text
                    except: categoria = ""
                    try: hook = nav.find_element(By.XPATH, "//h2[contains(@class,'ArticleHeader-Hook')]/div").text
                    except: hook = ""
                    try: parrafos = " ".join([p.text for p in nav.find_elements(By.XPATH, "//div[contains(@class,'ArticleBody-Content')]/p")])
                    except: parrafos = ""

                    cuerpo = (hook + " " + parrafos).strip()

                    data.append({
                        "Titulo": titulo, "Link": href, "Fecha": fecha, "Categoria": categoria, "Cuerpo": cuerpo
                    })

                except (TimeoutException, WebDriverException) as e:
                    print(f"❌ Error al procesar el enlace: {href}. Error: {e}")
                except Exception as e:
                    print(f"❌ Error desconocido al procesar el enlace: {href}. Error: {e}")
                finally:
                    # Cierra la pestaña y vuelve a la principal
                    nav.close()
                    nav.switch_to.window(nav.window_handles[0])

        except (TimeoutException, WebDriverException) as e:
            print(f"⚠️ Ocurrió un error al cargar la página principal o los bloques de noticias: {e}")
            break
        except Exception as e:
            print(f"⚠️ Ocurrió un error inesperado en el bucle de la página: {e}")
            break

        # Navegación a la siguiente página
        try:
            WebDriverWait(nav, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//div[contains(@class, 'Pagination-Nav')]/a[text()='Siguiente']"))
            ).click()
            time.sleep(3) # Pausa para que la nueva página se cargue
        except TimeoutException:
            print("No se encontró el botón de 'Siguiente' o no es cliqueable. Fin de la paginación.")
            break
        except Exception:
            print("No hay más páginas disponibles o la paginación ha cambiado.")
            break

    df = pd.DataFrame(data)
    return df

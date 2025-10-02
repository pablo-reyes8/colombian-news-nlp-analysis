import matplotlib.dates as mdates
import matplotlib.cm as cm # Import cm
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np



def conteos_por_intervalo(df, freq='W'):

    s = (df.set_index('fecha')
         .sort_index().resample(freq).size().rename('n'))
    return s.reset_index()


def plot_barras_temporales(df_count, titulo="Eventos por intervalo"):
    """
    Genera un gráfico de barras temporales con colores normalizados por intensidad.

    El gráfico representa en el eje X las fechas y en el eje Y los conteos de eventos.
    La intensidad del color de cada barra se ajusta de acuerdo con la normalización
    de los valores, usando el mapa de color "viridis". Además, se muestran etiquetas
    en las barras que superan un umbral mínimo de relevancia.

    Args:
        df_count (pandas.DataFrame): DataFrame con al menos dos columnas:
            - 'fecha' (datetime): eje X, intervalo temporal.
            - 'n' (int): eje Y, número de eventos en cada fecha.
        titulo (str, optional): Título del gráfico. Por defecto "Eventos por intervalo".

    Returns:
        None: Muestra el gráfico directamente con `matplotlib.pyplot.show()`.

    Notas:
        - Se utiliza un umbral (`thresh`) igual al máximo entre 5 y el 15% del valor
          máximo de `y`, para decidir qué barras mostrar con etiquetas numéricas.
        - La escala de colores se normaliza entre el valor mínimo y máximo de `y`.
        - Se usan `matplotlib.dates` para el formateo conciso de fechas.
    """
    x = df_count['fecha']
    y = df_count['n'].astype(int)

    norm = (y - y.min()) / (y.max() - y.min() + 1e-9)
    cmap = cm.get_cmap("viridis")
    colors = cmap(norm)

    plt.figure(figsize=(12, 4.5))
    plt.bar(x, y, color=colors, edgecolor="black", linewidth=0.5)

    ax = plt.gca()
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    plt.title(titulo, fontsize=16, fontweight='bold', pad=10)
    plt.xlabel("Tiempo")
    plt.ylabel("Conteo")

    plt.grid(axis='y', linestyle='--', alpha=0.4)

    ymax = y.max()
    thresh = max(5, int(0.15 * ymax))
    for xi, yi in zip(x, y):
        if yi >= thresh:
            plt.text(xi, yi, f"{yi:,}".replace(',', '.'), ha='center', va='bottom', fontsize=9, color="black")

    plt.tight_layout()
    plt.show()


def plot_linea_area(df_count, titulo="Serie temporal de eventos", rol_window=4):
    """
    Genera un gráfico de línea con área sombreada, media móvil y media global.

    El gráfico representa la evolución temporal de eventos, mostrando la serie original,
    un área sombreada bajo la curva, una línea de media móvil y una línea horizontal
    con la media global. Además, resalta los tres valores más altos de la serie.

    Args:
        df_count (pandas.DataFrame): DataFrame con al menos dos columnas:
            - 'fecha' (datetime): eje X, intervalo temporal.
            - 'n' (float o int): eje Y, número de eventos en cada fecha.
        titulo (str, optional): Título del gráfico. Por defecto "Serie temporal de eventos".
        rol_window (int, optional): Tamaño de la ventana para la media móvil.
            Por defecto 4.

    Returns:
        None: Muestra el gráfico directamente con `matplotlib.pyplot.show()`.

    Notas:
        - El área bajo la serie se muestra con un tono azul claro.
        - La línea de media móvil se dibuja con estilo punteado.
        - La media global se representa como línea horizontal roja.
        - Se resaltan con puntos y etiquetas los tres valores más altos de la serie.
        - Se usan `matplotlib.dates` para el formateo conciso de fechas.
    """
    x = df_count['fecha']
    y = df_count['n'].astype(float)

    media = y.mean()
    rol = y.rolling(rol_window, min_periods=max(1, rol_window//2)).mean()

    line_color = "#1f77b4"
    rol_color = "#ff7f0e"
    area_color = cm.get_cmap("Blues")(0.3)

    plt.figure(figsize=(12, 4.8))
    plt.fill_between(x, y, color=area_color, alpha=0.3)
    plt.plot(x, y, color=line_color, linewidth=2, marker='o', markersize=4, label="Conteo")
    plt.plot(x, rol, color=rol_color, linewidth=2, linestyle='--', label=f"Media móvil ({rol_window})")
    plt.axhline(media, linestyle=':', linewidth=2, color="red", alpha=0.7,label=f"Media = {media:.0f}")

    ax = plt.gca()
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    plt.title(titulo, fontsize=16, fontweight='bold', pad=10)
    plt.xlabel("Tiempo")
    plt.ylabel("Conteo")
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.legend(frameon=False)
    top_idx = np.argsort(y.values)[-3:]
    for i in top_idx:
        plt.scatter(x.iloc[i], y.iloc[i], color=line_color, s=60, zorder=3)
        plt.text(x.iloc[i], y.iloc[i] * 1.02,
                 f"{int(y.iloc[i]):,}".replace(',', '.'),
                 va='bottom', ha='center', fontsize=9, fontweight='bold', color=line_color)

    plt.tight_layout()
    plt.show()


def lemmas_filtrados_docs(texts ,nlp, batch_size=512, n_process=4 ):
    """
    Extrae y filtra los lemas de una lista de textos usando spaCy.

    La función procesa los textos con el pipeline de spaCy, eliminando tokens
    que sean stopwords o signos de puntuación, y conservando únicamente tokens
    alfabéticos. Devuelve una lista con los lemas en minúscula.

    Args:
        texts (list of str): Lista de documentos en formato texto.
        batch_size (int, optional): Tamaño de lote para el procesamiento por spaCy.
            Por defecto 512.
        n_process (int, optional): Número de procesos en paralelo a usar en spaCy.
            Por defecto 4.

    Returns:
        list of str: Lista de lemas filtrados en minúscula.

    Notas:
        - Se deshabilitan los componentes `ner` y `parser` para mejorar la eficiencia.
        - Se usan los atributos de spaCy `IS_STOP`, `IS_PUNCT`, `IS_ALPHA` y `LEMMA`.
        - Si no hay tokens válidos, se devuelve una lista vacía.
    """
    vocab_strings = nlp.vocab.strings
    lemas_ids_all = []

    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process,disable=["ner", "parser"]):

        arr = doc.to_array([IS_STOP, IS_PUNCT, IS_ALPHA, LEMMA])
        mask = (~arr[:,0].astype(bool)) & (~arr[:,1].astype(bool)) & (arr[:,2].astype(bool))
        lemas_ids_all.append(arr[mask, 3])

    lemas_ids_all = np.concatenate(lemas_ids_all) if lemas_ids_all else np.array([], dtype=np.int64)
    return [vocab_strings[lem_id].lower() for lem_id in lemas_ids_all]


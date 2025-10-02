import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import seaborn as sns

def plot_sentiment_bars(df, col="sentimiento"):
    """
    Genera un gráfico de barras horizontales que muestra la distribución de
    categorías de sentimiento en un DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame que contiene la columna de categorías
            de sentimiento.
        col (str, optional): Nombre de la columna con las categorías de sentimiento.
            Por defecto "sentimiento".

    Returns:
        None: Muestra la figura directamente con `matplotlib.pyplot.show()`.

    Notas:
        - Ordena las categorías según un orden predefinido:
          ["VERY NEGATIVE", "NEGATIVE", "NEUTRAL", "POSITIVE",
           "VERY POSITIVE", "UNDEFINED"]. Si existen otras categorías,
          se agregan al final en el orden en que aparecen.
        - Se calcula tanto el número absoluto como el porcentaje de documentos
          para cada categoría.
        - Colorea cada barra según un mapa de colores específico de sentimiento.
        - Añade etiquetas de conteo y porcentaje al final de cada barra.
        - El gráfico incluye el total de documentos en una anotación inferior.
    """
    counts = df[col].value_counts(dropna=False)
    total = int(counts.sum())
    preferred_order = ["VERY NEGATIVE", "NEGATIVE", "NEUTRAL", "POSITIVE", "VERY POSITIVE", "UNDEFINED"]
    ordered_index = [c for c in preferred_order if c in counts.index] + \
                    [c for c in counts.index if c not in preferred_order]
    counts = counts.reindex(ordered_index)

    pct = (counts / total * 100).round(1)

    color_map = {
        "VERY NEGATIVE": "#8B0000",
        "NEGATIVE": "#FF4C4C",
        "NEUTRAL": "#808080",
        "POSITIVE": "#1E90FF",
        "VERY POSITIVE": "#228B22",
        "UNDEFINED": "#C0C0C0"}
    colors = [color_map.get(cat, "#4682B4") for cat in counts.index]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.barh(counts.index, counts.values, color=colors)

    for bar, c, p in zip(bars, counts.values, pct.values):
        w = bar.get_width()
        ax.text(
            w + max(counts.values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{c:,.0f} ({p:.1f}%)",
            va="center", ha="left", fontsize=11)

    ax.set_title("Distribución de Sentimientos", fontsize=15, fontweight="bold")
    ax.set_xlabel("Número de documentos", fontsize=12)
    ax.set_ylabel("Categoría de sentimiento", fontsize=12)

    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    ax.set_xlim(0, max(counts.values) * 1.15)
    ax.margins(y=0.08)
    ax.text(
        1.0, -0.08,
        f"Total documentos: {total:,}",
        transform=ax.transAxes, fontsize=10, ha="right", color="dimgray")
    plt.tight_layout()
    plt.show()

color_map = {
    "VERY NEGATIVE": "#8B0000",
    "NEGATIVE": "#FF4C4C",
    "NEUTRAL": "#808080",
    "POSITIVE": "#1E90FF",
    "VERY POSITIVE": "#228B22",
    "UNDEFINED": "#C0C0C0"}

def top_ngrams(texts, n=3, ngram_range=(3,3)):
    """
    Obtiene los n-gramas más frecuentes en una lista de textos.

    Args:
        texts (list of str): Lista de documentos en formato texto.
        n (int, optional): Número de n-gramas más frecuentes a devolver.
            Por defecto 3.
        ngram_range (tuple of int, optional): Rango de tamaños de n-gramas a considerar,
            en el formato (min_n, max_n). Por defecto (3, 3), que corresponde a trigramas.

    Returns:
        list of tuple: Lista con los n-gramas más frecuentes y su frecuencia absoluta,
        ordenados de mayor a menor frecuencia. Cada elemento es una tupla:
        (ngram, frecuencia).

    Notas:
        - Utiliza `sklearn.feature_extraction.text.CountVectorizer` para tokenizar y
          contar los n-gramas.
        - El preprocesamiento y la tokenización son los que aplica CountVectorizer
          por defecto (minúsculas, tokenización básica).
        - Puede ser sensible a stopwords y puntuación si no se aplicó
          preprocesamiento previo.
    """

    vec = CountVectorizer(ngram_range=ngram_range)
    X = vec.fit_transform(texts)
    counts = X.sum(axis=0).A1
    vocab = vec.get_feature_names_out()
    freq = sorted(zip(vocab, counts), key=lambda x: x[1], reverse=True)[:n]
    return freq



def plot_sentiment_stack(
corpus: pd.DataFrame,
    fecha_col: str = "fecha",
    sent_col: str = "sentimiento",
    window: int = 3,
    min_docs: int = 20,
    order = None,
    color_map = None,
    tz = "UTC",
    ax= None,
    title = None,
    return_data: bool = False):

    """
    Traza la evolución temporal del **porcentaje** de documentos por clase de sentimiento
    usando una **media móvil** y apilado (stackplot).

    Parámetros
    ----------
    corpus : pd.DataFrame
        DataFrame con al menos las columnas `fecha_col` y `sent_col`.
    fecha_col : str, por defecto "fecha"
        Nombre de la columna de fechas.
    sent_col : str, por defecto "sentimiento"
        Nombre de la columna con las etiquetas de sentimiento (str).
    window : int, por defecto 3
        Ventana (en días) para la media móvil.
    min_docs : int, por defecto 20
        Mínimo de documentos por día para incluir en el cálculo de porcentajes
        (días por debajo quedan como NaN antes del suavizado).
    order : list[str], opcional
        Orden de las clases a mostrar (y leyenda). Si None, se infiere del DataFrame.
    color_map : dict[str, str], opcional
        Mapeo de color por clase. Si None, se usa un esquema por defecto.
    tz : str o None, por defecto "UTC"
        Zona horaria a la que convertir las fechas. Si None, no fuerza TZ.
    ax : matplotlib.axes.Axes, opcional
        Ejes sobre los que dibujar. Si None, crea figura y ejes nuevos.
    title : str, opcional
        Título del gráfico. Si None, se genera uno automáticamente.
    return_data : bool, por defecto False
        Si True, también devuelve los DataFrames intermedios:
        {'daily_counts','daily_tot','daily_pct','daily_pct_filtered','daily_pct_smooth'}.

    Devuelve
    --------
    (fig, ax) o (fig, ax, data_dict)
        La figura y ejes (y opcionalmente los DataFrames intermedios).

    Notas
    -----
    - Usa `floor('D')` para agrupar por día.
    - Rellena días faltantes entre min y max con 0 (antes del %).
    - Calcula porcentajes por día y aplica media móvil con `rolling(window, min_periods=1)`.
    - Días con total < `min_docs` se ponen NaN antes del suavizado (para no sesgar).
    """

    df = corpus.copy()
    df[fecha_col] = pd.to_datetime(df[fecha_col], utc=True, errors="coerce")
    if tz is None:
        pass
    else:
        df[fecha_col] = df[fecha_col].dt.tz_convert(tz) if df[fecha_col].dt.tz is not None else df[fecha_col].dt.tz_localize(tz)

    df = df.dropna(subset=[fecha_col])
    if df.empty:
        raise ValueError("No hay filas con fecha válida después de convertir/parsing.")
    df["dia"] = df[fecha_col].dt.floor("D")

    daily_counts = (
        df.groupby(["dia", sent_col])
          .size()
          .unstack(fill_value=0)
          .sort_index())
    
    all_days = pd.date_range(daily_counts.index.min(), daily_counts.index.max(), freq="D", tz=daily_counts.index.tz)
    daily_counts = daily_counts.reindex(all_days, fill_value=0)
    daily_counts.index.name = "dia"

    daily_tot = daily_counts.sum(axis=1)
    daily_pct = daily_counts.div(daily_tot.replace(0, np.nan), axis=0) * 100.0

    daily_pct_filtered = daily_pct.where(daily_tot >= min_docs)
    daily_pct_smooth = daily_pct_filtered.rolling(window=window, min_periods=1).mean()

    if order is None:
        default_order = ["VERY NEGATIVE","NEGATIVE","NEUTRAL","POSITIVE","VERY POSITIVE","UNDEFINED"]
        order = [c for c in default_order if c in daily_pct_smooth.columns] + \
                [c for c in daily_pct_smooth.columns if c not in default_order]
    else:
        order = [c for c in order if c in daily_pct_smooth.columns]  

    if not order:
        raise ValueError("No hay clases de sentimiento presentes para graficar.")

    default_color_map = {
        "VERY NEGATIVE": "#8B0000",
        "NEGATIVE":      "#FF6B6B",
        "NEUTRAL":       "#8A8A8A",
        "POSITIVE":      "#3A86FF",
        "VERY POSITIVE": "#228B22",
        "UNDEFINED":     "#C0C0C0"}
    
    color_map = color_map or default_color_map
    colors = [color_map.get(c, None) for c in order]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        created_fig = True
    else:
        fig = ax.figure

    x = daily_pct_smooth.index.to_pydatetime()
    y_stacks = [daily_pct_smooth[c].values for c in order]
    ax.stackplot(x, *y_stacks, labels=order, colors=colors, alpha=0.95)

    # Etiquetas y estilo
    title = title or f"Evolución temporal del sentimiento (%) — media móvil {window} días"
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_ylabel("Porcentaje de documentos", fontsize=12)
    ax.set_xlabel("Fecha", fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Sentimiento", bbox_to_anchor=(1.02, 1), loc="upper left")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    if return_data:
        data_dict = {
            "daily_counts": daily_counts,
            "daily_tot": daily_tot,
            "daily_pct": daily_pct,
            "daily_pct_filtered": daily_pct_filtered,
            "daily_pct_smooth": daily_pct_smooth}
        return fig, ax, data_dict

    return fig, ax




def plot_sentiment_violin(
    df: pd.DataFrame,
    sent_col: str = "sentimiento",
    score_col: str = "score",
    order = None,
    palette= None,
    figsize: tuple = (9, 5.5),
    title: str = "Distribución de Scores por Clase de Sentimiento"):
    """
    Genera un gráfico tipo violin plot (con box interno) que muestra la distribución
    de los scores de confianza por clase de sentimiento.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con al menos dos columnas: sent_col y score_col.
    sent_col : str, por defecto "sentimiento"
        Nombre de la columna con las etiquetas de sentimiento.
    score_col : str, por defecto "score"
        Nombre de la columna con los valores de confianza del modelo.
    order : list[str], opcional
        Orden de las categorías en el eje X. Si None, se toma el orden del DataFrame.
    palette : dict[str,str], opcional
        Mapa de colores por categoría. Si None, usa uno por defecto.
    figsize : tuple, por defecto (9, 5.5)
        Tamaño de la figura en pulgadas.
    title : str, por defecto "Distribución de Scores por Clase de Sentimiento"
        Título del gráfico.

    Returns
    -------
    fig, ax : matplotlib Figure y Axes
    """
    if order is None:
        order = ["VERY NEGATIVE", "NEGATIVE", "NEUTRAL",
                 "POSITIVE", "VERY POSITIVE", "UNDEFINED"]

    if palette is None:
        palette = {
            "VERY NEGATIVE": "#8B0000",
            "NEGATIVE": "#FF4C4C",
            "NEUTRAL": "#808080",
            "POSITIVE": "#1E90FF",
            "VERY POSITIVE": "#228B22",
            "UNDEFINED": "#C0C0C0"}

    if sent_col not in df.columns or score_col not in df.columns:
        raise ValueError(f"El DataFrame no contiene las columnas '{sent_col}' y '{score_col}'.")

    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(
        data=df,
        x=sent_col,
        y=score_col,
        order=[c for c in order if c in df[sent_col].unique()],
        palette=palette,
        inner="box",
        cut=0,
        ax=ax)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Categoría de Sentimiento", fontsize=12)
    ax.set_ylabel("Score (confianza del modelo)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    return fig, ax


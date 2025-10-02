import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_transform_embeddings(df_train, col="texto_tfidf",
                             word_ngram_range=(1, 2),
                             char_ngram_range=(3, 5),
                             min_df=3, max_df=0.9,
                             sublinear_tf=True, norm="l2"):
    """
    Ajusta y transforma embeddings combinando TF-IDF de palabras (unigramas + bigramas)
    y de caracteres (trigramas a pentagramas).

    Parámetros
    ----------
    df_train : pandas.DataFrame
        DataFrame de entrenamiento que contiene la columna de texto.
    col : str, default="texto_tfidf"
        Nombre de la columna que contiene el texto a vectorizar.
    word_ngram_range : tuple, default=(1, 2)
        Rango de n-gramas para el análisis de palabras.
    char_ngram_range : tuple, default=(3, 5)
        Rango de n-gramas para el análisis de caracteres.
    min_df : int, default=3
        Frecuencia mínima para que un término se incluya.
    max_df : float, default=0.9
        Proporción máxima de documentos que pueden contener un término.
    sublinear_tf : bool, default=True
        Si True, aplica sublinear tf scaling (1 + log(tf)).
    norm : str, default="l2"
        Norma de normalización a aplicar a cada vector.

    Retorna
    -------
    X : scipy.sparse.csr_matrix
        Matriz TF-IDF combinada (palabras + caracteres).
    v_word : sklearn.feature_extraction.text.TfidfVectorizer
        Vectorizador ajustado para palabras.
    v_char : sklearn.feature_extraction.text.TfidfVectorizer
        Vectorizador ajustado para caracteres.

    Ejemplo
    -------
    >>> X_nlp, v_word, v_char = fit_transform_embeddings(noticias1, col="texto_tfidf")
    """
    
    v_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        norm=norm)

    v_char = TfidfVectorizer(
        analyzer="char",
        ngram_range=char_ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        norm=norm)

    # Ajuste y transformación
    Xw = v_word.fit_transform(df_train[col])
    Xc = v_char.fit_transform(df_train[col])

    X = sp.hstack([Xw, Xc], format="csr")

    return X, v_word, v_char



def transform_embeddings(df_test, v_word, v_char, col="texto_tfidf"):
    """
    Transforma un conjunto de prueba con los vectorizadores ya ajustados.

    Parámetros
    ----------
    df_test : pandas.DataFrame
        DataFrame con la columna de texto a vectorizar.
    v_word : sklearn.feature_extraction.text.TfidfVectorizer
        Vectorizador ya ajustado para palabras.
    v_char : sklearn.feature_extraction.text.TfidfVectorizer
        Vectorizador ya ajustado para caracteres.
    col : str, default="texto_tfidf"
        Nombre de la columna que contiene el texto a transformar.

    Retorna
    -------
    X : scipy.sparse.csr_matrix
        Matriz TF-IDF combinada (palabras + caracteres).
    """
    Xw = v_word.transform(df_test[col])
    Xc = v_char.transform(df_test[col])
    X  = sp.hstack([Xw, Xc], format="csr")
    return X
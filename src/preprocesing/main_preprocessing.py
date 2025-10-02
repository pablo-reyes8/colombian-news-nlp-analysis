import re 
import unicodedata
import unidecode 
import spacy 

def preprocess_text_column(df,input_col,output_col = "texto_tfidf",
    spacy_model = "es_core_news_sm",
    remove_accents = True,remove_numbers = True,
    remove_punct = True,remove_stopwords = True,
    min_token_len = 2,batch_size = 256,n_process = 1, return_tokens = False):
    """
    Preprocesa una columna de texto en español con limpieza regex, normalización opcional
    de acentos y lematización con spaCy, filtrando tokens según reglas configurables.

    El flujo incluye: (1) limpieza básica (URLs, emails, @menciones, #hashtags, símbolos
    de moneda, caracteres no alfanuméricos), (2) lowercasing, (3) normalización opcional
    de acentos, (4) lematización y filtrado (stopwords, signos, números y longitud mínima).
    Devuelve una nueva columna con tokens listos para TF-IDF (string espacio-separado) o,
    si se especifica, una lista de tokens.

    Args:
        df (pandas.DataFrame): DataFrame de entrada que contiene la columna de texto crudo.
        input_col (str): Nombre de la columna con el texto crudo a procesar.
        output_col (str, optional): Nombre de la nueva columna con el texto procesado.
            Por defecto "texto_tfidf".
        spacy_model (str, optional): Nombre del modelo spaCy a cargar (p. ej.,
            "es_core_news_sm", "es_core_news_md"). Por defecto "es_core_news_sm".
        remove_accents (bool, optional): Si True, normaliza acentos/tildes con `unidecode`.
            Por defecto True.
        remove_numbers (bool, optional): Si True, elimina tokens numéricos. Por defecto True.
        remove_punct (bool, optional): Si True, elimina signos de puntuación/espacios/citas.
            Por defecto True.
        remove_stopwords (bool, optional): Si True, filtra stopwords de spaCy. Por defecto True.
        min_token_len (int, optional): Longitud mínima de lema para conservar el token.
            Por defecto 2.
        batch_size (int, optional): Tamaño de lote para `nlp.pipe`. Por defecto 256.
        n_process (int, optional): Número de procesos en paralelo para spaCy. Por defecto 1.
        return_tokens (bool, optional): Si True, `output_col` contendrá listas de tokens; si
            False, contendrá una cadena con tokens separados por espacio. Por defecto False.

    Returns:
        pandas.DataFrame: Copia de `df` con la columna `output_col` añadida, conteniendo
        texto procesado (string o lista de tokens según `return_tokens`).

    Notas:
        - La limpieza regex elimina URLs, emails, menciones, hashtags, símbolos de moneda y
          caracteres no alfanuméricos básicos (manteniendo letras y dígitos).
        - `remove_accents=True` aplica `unidecode`, lo que puede perder distinciones como "ñ".
        - Se usa `lemma_` de spaCy en minúscula; la calidad depende del modelo elegido.
        - `remove_stopwords` usa `nlp.Defaults.stop_words`; ajuste según necesidades.
        - Para datasets grandes, ajuste `batch_size` y `n_process` según su hardware.
        - La función no modifica `df` in place; retorna una copia con la columna añadida.
    """
    # Regex para limpieza previa
    re_url = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
    re_email = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    re_mention = re.compile(r"(?<!\w)@\w+")
    re_hashtag = re.compile(r"(?<!\w)#\w+")
    re_currency= re.compile(r"[€$£¥₿]")
    re_space = re.compile(r"\s+")
    re_nonbasic = re.compile(r"[^0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ\s]", flags=re.UNICODE)

    def _clean_text_basic(txt: str):

        if not isinstance(txt, str):
            return ""

        t = txt.strip()
        t = re_url.sub(" ", t)
        t = re_email.sub(" ", t)
        t = re_mention.sub(" ", t)
        t = re_hashtag.sub(" ", t)
        t = re_currency.sub(" ", t)
        t = t.lower()
        t = re_nonbasic.sub(" ", t)
        t = re_space.sub(" ", t).strip()
        return t

    # Normalización de acentos
    def _strip_accents(txt: str) -> str:
        return unidecode(txt) if txt else txt

    # spaCy para lematizar y filtrar
    nlp = spacy.load(spacy_model, disable=["ner"])

    if nlp.max_length < 2_000_000:
        nlp.max_length = 2_000_000

    #stopwords
    stopwords_spacy = nlp.Defaults.stop_words if remove_stopwords else set()

    def _token_ok(tok):
        if remove_punct and (tok.is_punct or tok.is_space or tok.is_quote):
            return False
        if remove_numbers and (tok.like_num or tok.shape_.isdigit()):
            return False
        if remove_stopwords and tok.lemma_ in stopwords_spacy:
            return False
        if len(tok.lemma_) < min_token_len:
            return False
        return True

    # Pipeline
    cleaned = df[input_col].fillna("").map(_clean_text_basic)
    if remove_accents:
        cleaned = cleaned.map(_strip_accents)

    results = []
    for doc in nlp.pipe(cleaned.tolist(), batch_size=batch_size, n_process=n_process):
        toks = [tok.lemma_.lower() for tok in doc if _token_ok(tok)]
        if return_tokens:
            results.append(toks)
        else:
            results.append(" ".join(toks))

    df = df.copy()
    df[output_col] = results
    return df
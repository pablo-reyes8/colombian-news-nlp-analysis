from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
import joblib


@dataclass
class BowConfig:
    """
    Configuración del CountVectorizer para BOW.
    """
    ngram_range: tuple[int, int] = (1, 1)        # (1,1)=unigramas; (1,2)=uni+bi; etc.
    max_features: Optional[int] = None        # recorta al top-k por frecuencia
    min_df: Union[int, float] = 1             # ignora términos raros (int=cuentas / float=proporción)
    max_df: Union[int, float] = 1.0           # ignora términos demasiado comunes
    binary: bool = False             # True = presencia/ausencia; False = conteos
    lowercase: bool = True
    strip_accents: Optional[str] = None        # {'ascii','unicode',None}
    stop_words: Optional[Union[str, List[str]]] = None  # 'spanish' o lista personalizada
    analyzer: str = "word"                       # 'word' o 'char'
    token_pattern: str = r"(?u)\b\w\w+\b"      # tokens de 2+ chars (ajústalo si ya limpiaste)
    vocabulary: Optional[dict] = None          # fija vocabulario si quieres reproducibilidad


class BowFeaturizerDF:
    """
    Crea features Bag-of-Words y:
      (a) devuelve matriz X (sparse CSR) de forma [N, d]
      (b) agrega una columna al DataFrame con cada vector (lista o sparse row)

    API principal:
      - fit(df, text_col)
      - transform(df, text_col, out_col, tolist)
      - fit_transform_add_column(...)
    """
    def __init__(self, config: BowConfig = BowConfig()):
        self.config = config
        self.vectorizer: Optional[TransformerMixin] = None
        self.feature_names_: Optional[List[str]] = None

    def _build_vectorizer(self):
        return CountVectorizer(
            ngram_range=self.config.ngram_range,
            max_features=self.config.max_features,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            binary=self.config.binary,
            lowercase=self.config.lowercase,
            strip_accents=self.config.strip_accents,
            stop_words=self.config.stop_words,
            analyzer=self.config.analyzer,
            token_pattern=self.config.token_pattern,
            vocabulary=self.config.vocabulary)

    def fit(self, df: pd.DataFrame, text_col: str) -> "BowFeaturizerDF":
        assert text_col in df.columns, f"No existe la columna '{text_col}'."
        texts = df[text_col].astype(str).tolist()
        self.vectorizer = self._build_vectorizer()
        X = self.vectorizer.fit_transform(texts)
        self.feature_names_ = list(self.vectorizer.get_feature_names_out())
        # Devolvemos self por encadenamiento
        return self

    def transform(
        self,
        df: pd.DataFrame,
        text_col: str,
        out_col: str = "bow",
        tolist: bool = False) :
        """
        Transforma y agrega columna:
          - tolist=True -> guarda cada fila como lista (denso). Útil solo para datasets pequeños.
          - tolist=False -> guarda cada fila como csr_row (sparse), mucho más eficiente.
        """
        assert self.vectorizer is not None, "Debes llamar a fit() primero."
        assert text_col in df.columns, f"No existe la columna '{text_col}'."
        texts = df[text_col].astype(str).tolist()
        X = self.vectorizer.transform(texts)  # CSR [N, d]
        df_out = df.copy()

        if tolist:
            dense = X.toarray()
            df_out[out_col] = [row.tolist() for row in dense]
        else:
            df_out[out_col] = [X.getrow(i) for i in range(X.shape[0])]

        return df_out, X

    def fit_transform_add_column(
        self,
        df: pd.DataFrame,
        text_col: str,
        out_col: str = "bow",
        tolist: bool = False):
        self.fit(df, text_col)
        return self.transform(df, text_col, out_col=out_col, tolist=tolist)

    # Utilidades de persistencia
    def save_vectorizer(self, path: str) -> None:
        assert self.vectorizer is not None, "Vectorizer vacío; ejecuta fit() primero."
        joblib.dump(self.vectorizer, path)

    def load_vectorizer(self, path: str) -> None:
        self.vectorizer = joblib.load(path)
        self.feature_names_ = list(self.vectorizer.get_feature_names_out())


if __name__ == "__main__":
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "texto_clean": [
            "banrep decide tasa interes explica postura",
            "gustavo petro reune lideres regionales discutir reformas",
            "regulacion inteligencia artificial europa avanza nuevo borrador"]})

    cfg = BowConfig(ngram_range=(1, 2),     
        max_features=5000,      
        min_df=2,              
        max_df=0.95,            
        binary=False,          
        stop_words=None)

    bow = BowFeaturizerDF(cfg)

    bow.fit(df, text_col="texto_clean")
    df_bow, X = bow.transform(df, text_col="texto_clean", out_col="bow", tolist=False)
    print(df_bow.head())
    print("Matriz BOW:", X.shape, "sparse?", sp.issparse(X))

    df_bow2, X2 = bow.fit_transform_add_column(
        df, text_col="texto_clean", out_col="bow_list", tolist=True)
    
    print(df_bow2.head())
    print("Matriz BOW:", X2.shape)

    bow.save_vectorizer("bow_vectorizer.joblib")
    bow2 = BowFeaturizerDF()
    bow2.load_vectorizer("bow_vectorizer.joblib")
    df_bow_loaded, X_loaded = bow2.transform(df, text_col="texto_clean", out_col="bow_loaded", tolist=False)
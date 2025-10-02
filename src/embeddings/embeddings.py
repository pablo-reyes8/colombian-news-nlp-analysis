import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Union, Literal
import pandas as pd

class BertEmbedderDF:
    """
    Crea embeddings con un modelo tipo BERT y los agrega a un DataFrame.
    - Soporta batch, mean/CLS pooling y normalización L2.
    - Devuelve:
        (a) el DataFrame con una nueva columna (lista/np.array por fila), y
        (b) opcionalmente una matriz np.ndarray [N, d].
    """
    def __init__(
        self,device = None, 
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        pooling: Literal["mean", "cls"] = "mean",
        normalize: bool = True,
        max_length: int = 256,)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        self.normalize = normalize
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            emb = last_hidden_state[:, 0] 
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1) 
            summed = (last_hidden_state * mask).sum(dim=1)  
            counts = mask.sum(dim=1).clamp(min=1)
            emb = summed / counts
        else:
            raise ValueError("pooling debe ser 'mean' o 'cls'")

        if self.normalize:
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb

    @torch.no_grad()
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state
        emb = self._pool(last_hidden, inputs["attention_mask"])  
        return emb.cpu().numpy()

    def add_embeddings_column(
        self,
        df: pd.DataFrame,
        text_col: str,
        out_col: str = "embedding",
        batch_size: int = 64,
        return_matrix: bool = True,
        tolist: bool = True,):
        """
        Params
        ------
        df: DataFrame con la columna de texto preprocesado (str).
        text_col: nombre de la columna con texto limpio.
        out_col: nombre de la columna a crear con embeddings.
        batch_size: tamaño de lote para acelerar.
        return_matrix: si True, también devuelve la matriz [N, d].
        tolist: si True, guarda cada embedding como list (JSON/parquet-friendly).
                Si False, guarda como np.ndarray por fila.

        Returns
        -------
        df_out  (y opcionalmente) X_matrix
        """
        assert text_col in df.columns, f"No existe la columna '{text_col}' en el DataFrame."

        texts = df[text_col].astype(str).tolist()
        embs_list = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embs = self.encode_batch(batch)  
            embs_list.append(embs)

        X = np.vstack(embs_list) 
        df_out = df.copy()

        if tolist:
            df_out[out_col] = [row.tolist() for row in X]
        else:
            df_out[out_col] = list(X)

        return (df_out, X) if return_matrix else df_out


if __name__ == "__main__":
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "texto_clean": [
            "banrep anuncia decisión de tasas de interés y explica su postura",
            "gustavo petro se reúne con líderes regionales para discutir reformas",
            "regulación de inteligencia artificial en europa avanza en nuevo borrador"]})

    embedder = BertEmbedderDF(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        pooling="mean",
        normalize=True,
        max_length=256)

    df_emb, X = embedder.add_embeddings_column(df,
        text_col="texto_clean",
        out_col="embedding",
        batch_size=32,
        return_matrix=True,   
        tolist=True )

    print(df_emb.head())
    print("Matriz:", X.shape)  

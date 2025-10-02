from __future__ import annotations
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

from scipy.stats import loguniform, randint, uniform

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix)

from sklearn.utils.class_weight import compute_class_weight

from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse as sp


def plot_confusion_matrix(
    y_true,
    y_pred,
    y_labels=None,
    titulo="Matriz de Confusión",
    normalize: Optional[str] = None,  
    figsize: Tuple[int, int] = (7, 6),
    cmap: str = "Blues",
    fmt: str = ".2f"):

    """
    Dibuja la matriz de confusión con opción de normalización.

    Args:
        y_true, y_pred: etiquetas verdaderas y predichas
        y_labels: lista de nombres de clases (opcional; si None, usa np.unique(y_true))
        normalize: {"true","pred","all"} o None (como en sklearn.confusion_matrix)
        figsize: tamaño de figura
        cmap: colormap de seaborn.heatmap
        fmt: formato numérico; si normalize=None, se fuerza "d"

    Notas:
        - Normalizaciones útiles:
            "true": por fila (recall por clase)
            "pred": por columna (precision por clase)
            "all": por total
    """
    if y_labels is None:
        y_labels = np.unique(y_true)

    cm = confusion_matrix(y_true, y_pred, labels=y_labels, normalize=normalize)
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d" if normalize is None else fmt,
        cmap=cmap,
        xticklabels=y_labels,
        yticklabels=y_labels
    )
    plt.xlabel("Predicción")
    plt.ylabel("Verdadero")
    norm_txt = "" if normalize is None else f" — normalizada ({normalize})"
    plt.title(f"{titulo}{norm_txt}")
    plt.tight_layout()
    plt.show()

def _evaluate_and_report(y_test, y_pred, etiquetas=None, titulo=""):
    print(f"\n🔎 {titulo}")
    print("Accuracy:", f"{accuracy_score(y_test, y_pred):.4f}")
    print("Balanced Accuracy:", f"{balanced_accuracy_score(y_test, y_pred):.4f}")
    print("\nReporte de clasificación:\n", classification_report(y_test, y_pred, target_names=etiquetas))
    # Dos vistas de la matriz: cruda y normalizada por fila
    plot_confusion_matrix(y_test, y_pred, etiquetas, titulo=f"{titulo} — Matriz (cruda)", normalize=None)
    plot_confusion_matrix(y_test, y_pred, etiquetas, titulo=f"{titulo} — Matriz (normalizada por fila)", normalize="true")


# -----------------------------
def _split_stratified(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)


def entrenar_mlp_randomsearch(
    X,y,
    test_size: float = 0.2,
    random_state: int = 42,
    n_iter: int = 30,
    cv_splits: int = 5,
    max_iter: int = 200,
    n_jobs: int = -1,
    scoring: str = "f1_macro",
    verbose: int = 1):

    """
    Tuning de MLPClassifier con RandomizedSearchCV.
    Nota: si X es sparse, se convierte a denso para el MLP (cuidado con memoria).

    Retorna:
        best_model, cv_results_df_ordenado
    """
    X_train, X_test, y_train, y_test = _split_stratified(X, y, test_size, random_state)

    if sp.issparse(X_train):
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    param_distributions = {
        "hidden_layer_sizes": [(h1, h2) for h1 in [64, 128, 256, 384] for h2 in [32, 64, 128]],
        "alpha": loguniform(1e-6, 1e-2),
        "learning_rate_init": loguniform(1e-4, 5e-2),
        "activation": ["relu", "tanh"],
        "batch_size": [64, 128, 256],
        "solver": ["adam", "lbfgs"], }

    base = MLPClassifier(
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.1,
        verbose=False)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    rs = RandomizedSearchCV(
        base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        n_jobs=n_jobs,
        cv=cv,
        refit=True,
        verbose=verbose,
        random_state=random_state)

    rs.fit(X_train, y_train)

    best_model: MLPClassifier = rs.best_estimator_
    y_pred = best_model.predict(X_test)
    _evaluate_and_report(y_test, y_pred, etiquetas=np.unique(y), titulo="MLP — Mejor modelo")

    # Resultados CV ordenados
    results = pd.DataFrame(rs.cv_results_).sort_values("rank_test_score")
    return best_model, results


def entrenar_rf_randomsearch(
    X,y,
    test_size: float = 0.2,
    random_state: int = 42,
    n_iter: int = 40,
    cv_splits: int = 5,
    n_jobs: int = -1,
    scoring: str = "f1_macro",
    verbose: int = 1,
    considerar_class_weight: bool = True):
    """
    Tuning de RandomForestClassifier.
    Retorna:
        best_model, cv_results_df_ordenado
    """
    X_train, X_test, y_train, y_test = _split_stratified(X, y, test_size, random_state)

    class_weights_grid = [None, "balanced"] if considerar_class_weight else [None]

    param_distributions = {
        "n_estimators": randint(300, 1200),
        "max_depth": randint(5, 60),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
        "class_weight": class_weights_grid}

    base = RandomForestClassifier(
        random_state=random_state,
        n_jobs=n_jobs)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    rs = RandomizedSearchCV(
        base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        n_jobs=n_jobs,
        cv=cv,
        refit=True,
        verbose=verbose,
        random_state=random_state)

    rs.fit(X_train, y_train)

    best_model: RandomForestClassifier = rs.best_estimator_
    y_pred = best_model.predict(X_test)
    _evaluate_and_report(y_test, y_pred, etiquetas=np.unique(y), titulo="RandomForest — Mejor modelo")

    results = pd.DataFrame(rs.cv_results_).sort_values("rank_test_score")
    return best_model, results


def entrenar_svm_randomsearch(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42,
    n_iter: int = 30,
    cv_splits: int = 5,
    n_jobs: int = -1,
    scoring: str = "f1_macro",
    verbose: int = 1,
    usar_class_weight_balanced: bool = True):

    """
    Tuning de LinearSVC (muy eficiente para TF-IDF/BOW).
    Retorna:
        best_model, cv_results_df_ordenado
    """
    X_train, X_test, y_train, y_test = _split_stratified(X, y, test_size, random_state)

    class_weight = "balanced" if usar_class_weight_balanced else None

    param_distributions = {
        "C": loguniform(1e-3, 1e2),
        "loss": ["hinge", "squared_hinge"],
        "tol": loguniform(1e-5, 1e-2)}

    base = LinearSVC(
        random_state=random_state,
        class_weight=class_weight,
        max_iter=10_000)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    rs = RandomizedSearchCV(
        base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        n_jobs=n_jobs,
        cv=cv,
        refit=True,
        verbose=verbose,
        random_state=random_state)

    rs.fit(X_train, y_train)

    best_model: LinearSVC = rs.best_estimator_
    y_pred = best_model.predict(X_test)
    _evaluate_and_report(y_test, y_pred, etiquetas=np.unique(y), titulo="LinearSVC — Mejor modelo")

    results = pd.DataFrame(rs.cv_results_).sort_values("rank_test_score")
    return best_model, results


if __name__ == "__main__":
     from sklearn.datasets import fetch_20newsgroups
     from sklearn.feature_extraction.text import TfidfVectorizer

     data = fetch_20newsgroups(subset="train",
                               categories=["sci.space", "talk.politics.mideast", "rec.sport.hockey"])
     vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
     X = vec.fit_transform(data.data)
     y = data.target

     # SVM lineal
     best_svm, svm_cv = entrenar_svm_randomsearch(X, y, n_iter=20)

     # Random Forest
     best_rf, rf_cv = entrenar_rf_randomsearch(X, y, n_iter=20)

     # MLP (convierte X a denso)
     best_mlp, mlp_cv = entrenar_mlp_randomsearch(X, y, n_iter=15)
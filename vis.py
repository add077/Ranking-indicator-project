from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


DEFAULT_FIGSIZE: Tuple[float, float] = (6.0, 5.0)
DEFAULT_CMAP = "Blues"


def _resolve_target_and_prediction_columns(
    df: pd.DataFrame,
    *,
    target_col: Optional[str] = None,
    pred_col: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Resout les colonnes cible / prediction avec des conventions souples.
    """
    target_candidates = [target_col, "y_true", "target"]
    pred_candidates = [pred_col, "y_pred", "pred_class"]

    resolved_target = next(
        (column for column in target_candidates if column is not None and column in df.columns),
        None,
    )
    resolved_pred = next(
        (column for column in pred_candidates if column is not None and column in df.columns),
        None,
    )

    if resolved_target is None:
        raise ValueError(
            "Impossible de trouver la colonne cible. Colonnes testees : "
            + ", ".join(str(c) for c in target_candidates if c is not None)
        )
    if resolved_pred is None:
        raise ValueError(
            "Impossible de trouver la colonne de prediction. Colonnes testees : "
            + ", ".join(str(c) for c in pred_candidates if c is not None)
        )

    return resolved_target, resolved_pred


def _coerce_binary_series(series: pd.Series, *, name: str) -> pd.Series:
    """
    Convertit proprement une serie en binaire numerique.
    """
    converted = pd.to_numeric(series, errors="coerce")
    valid_mask = converted.isin([0, 1]) | converted.isna()
    if not valid_mask.all():
        invalid_values = sorted(pd.Series(series[~valid_mask]).dropna().astype(str).unique().tolist())
        raise ValueError(
            f"La colonne `{name}` contient des valeurs non binaires : {', '.join(invalid_values[:10])}"
        )
    return converted.astype("Float64")


def extract_confusion_inputs(
    df: pd.DataFrame,
    *,
    target_col: Optional[str] = None,
    pred_col: Optional[str] = None,
    dropna: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    """
    Extrait `y_true` et `y_pred` depuis un DataFrame de scoring.
    """
    resolved_target, resolved_pred = _resolve_target_and_prediction_columns(
        df,
        target_col=target_col,
        pred_col=pred_col,
    )

    y_true = _coerce_binary_series(df[resolved_target], name=resolved_target)
    y_pred = _coerce_binary_series(df[resolved_pred], name=resolved_pred)

    if dropna:
        valid_mask = y_true.notna() & y_pred.notna()
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

    return y_true.astype(int), y_pred.astype(int)


def make_confusion_matrix_table(
    y_true: Sequence[int] | pd.Series,
    y_pred: Sequence[int] | pd.Series,
    *,
    normalize: Optional[str] = None,
    labels: Sequence[int] = (0, 1),
    row_names: Sequence[str] = ("True 0", "True 1"),
    col_names: Sequence[str] = ("Pred 0", "Pred 1"),
) -> pd.DataFrame:
    """
    Construit une matrice de confusion sous forme de DataFrame.

    `normalize` suit la convention sklearn : `None`, `"true"`, `"pred"`, `"all"`.
    """
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(labels),
        normalize=normalize,
    )
    return pd.DataFrame(matrix, index=list(row_names), columns=list(col_names))


def summarize_confusion_matrix(
    y_true: Sequence[int] | pd.Series,
    y_pred: Sequence[int] | pd.Series,
) -> pd.Series:
    """
    Retourne un petit resume des metriques derivees de la matrice de confusion.
    """
    y_true_series = pd.Series(y_true).astype(int)
    y_pred_series = pd.Series(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_series, y_pred_series, labels=[0, 1]).ravel()

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    return pd.Series(
        {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "precision": float(precision_score(y_true_series, y_pred_series, zero_division=0)),
            "recall": float(recall_score(y_true_series, y_pred_series, zero_division=0)),
            "specificity": specificity,
            "f1": float(f1_score(y_true_series, y_pred_series, zero_division=0)),
        }
    )


def plot_confusion_matrix_heatmap(
    y_true: Sequence[int] | pd.Series,
    y_pred: Sequence[int] | pd.Series,
    *,
    normalize: Optional[str] = None,
    title: Optional[str] = None,
    cmap: str = DEFAULT_CMAP,
    fmt: Optional[str] = None,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    annot: bool = True,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Affiche une heatmap de la matrice de confusion.
    """
    table = make_confusion_matrix_table(y_true, y_pred, normalize=normalize)
    if fmt is None:
        fmt = ".2f" if normalize is not None else "g"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sns.heatmap(
        table,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Observation")
    if title is None:
        title = "Matrice de confusion"
        if normalize is not None:
            title += f" ({normalize})"
    ax.set_title(title)
    return fig, ax


def plot_confusion_matrix_from_df(
    df: pd.DataFrame,
    *,
    target_col: Optional[str] = None,
    pred_col: Optional[str] = None,
    normalize: Optional[str] = None,
    title: Optional[str] = None,
    cmap: str = DEFAULT_CMAP,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Version pratique quand on part d'un DataFrame de scoring.
    """
    y_true, y_pred = extract_confusion_inputs(
        df,
        target_col=target_col,
        pred_col=pred_col,
        dropna=True,
    )
    return plot_confusion_matrix_heatmap(
        y_true,
        y_pred,
        normalize=normalize,
        title=title,
        cmap=cmap,
        figsize=figsize,
        ax=ax,
    )


def plot_confusion_matrix_pair(
    df: pd.DataFrame,
    *,
    target_col: Optional[str] = None,
    pred_col: Optional[str] = None,
    titles: Sequence[str] = ("Matrice brute", "Matrice normalisee par ligne"),
    cmap: str = DEFAULT_CMAP,
    figsize: Tuple[float, float] = (12.0, 5.0),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Affiche cote a cote la matrice brute et la matrice normalisee par ligne.
    """
    y_true, y_pred = extract_confusion_inputs(
        df,
        target_col=target_col,
        pred_col=pred_col,
        dropna=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_confusion_matrix_heatmap(
        y_true,
        y_pred,
        normalize=None,
        title=titles[0],
        cmap=cmap,
        ax=axes[0],
    )
    plot_confusion_matrix_heatmap(
        y_true,
        y_pred,
        normalize="true",
        title=titles[1],
        cmap=cmap,
        ax=axes[1],
    )
    fig.tight_layout()
    return fig, axes


def plot_confusion_matrices_by_threshold(
    df: pd.DataFrame,
    *,
    prob_col: str = "pred_proba_success",
    target_col: Optional[str] = None,
    thresholds: Iterable[float] = (0.3, 0.5, 0.7),
    normalize: Optional[str] = "true",
    cmap: str = DEFAULT_CMAP,
    figsize_per_plot: Tuple[float, float] = (5.0, 4.0),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Compare plusieurs matrices de confusion selon differents seuils de proba.
    """
    resolved_target, _ = _resolve_target_and_prediction_columns(
        df,
        target_col=target_col,
        pred_col="pred_class" if "pred_class" in df.columns else "y_pred",
    )
    if prob_col not in df.columns:
        raise ValueError(f"Colonne de probabilite introuvable : `{prob_col}`.")

    y_true = _coerce_binary_series(df[resolved_target], name=resolved_target)
    y_prob = pd.to_numeric(df[prob_col], errors="coerce")
    valid_mask = y_true.notna() & y_prob.notna()
    y_true = y_true[valid_mask].astype(int)
    y_prob = y_prob[valid_mask].astype(float)

    threshold_list = list(thresholds)
    if not threshold_list:
        raise ValueError("`thresholds` ne peut pas etre vide.")

    n_cols = len(threshold_list)
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1]),
        squeeze=False,
    )

    for idx, threshold in enumerate(threshold_list):
        y_pred = (y_prob >= float(threshold)).astype(int)
        plot_confusion_matrix_heatmap(
            y_true,
            y_pred,
            normalize=normalize,
            title=f"Seuil = {threshold:.2f}",
            cmap=cmap,
            figsize=figsize_per_plot,
            ax=axes[0, idx],
        )

    fig.tight_layout()
    return fig, axes


def plot_confusion_matrix_for_backtest(
    results: dict,
    *,
    normalize: Optional[str] = None,
    title: str = "Matrice de confusion OOS",
    cmap: str = DEFAULT_CMAP,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Raccourci pour partir directement d'un dictionnaire `results` du backtest.
    """
    if "oos_predictions_df" not in results:
        raise ValueError("`results` doit contenir `oos_predictions_df`.")

    return plot_confusion_matrix_from_df(
        results["oos_predictions_df"],
        target_col="y_true",
        pred_col="y_pred",
        normalize=normalize,
        title=title,
        cmap=cmap,
        figsize=figsize,
    )


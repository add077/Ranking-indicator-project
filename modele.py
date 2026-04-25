from __future__ import annotations

from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass
import logging
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline

from preprocessing import build_logit_preprocessor, prepare_logit_inputs


LOGGER = logging.getLogger(__name__)


DEFAULT_WALK_FORWARD_WINDOWS: List[Tuple[str, str]] = [
    ("2023-01-01", "2023-06-30"),
    ("2023-07-01", "2023-12-31"),
    ("2024-01-01", "2024-06-30"),
    ("2024-07-01", "2024-12-31"),
]


def _ensure_required_columns(df: pd.DataFrame, required_columns: Sequence[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            "Colonnes manquantes dans le DataFrame : " + ", ".join(missing)
        )


def _normalize_datetime_series(series: pd.Series) -> pd.Series:
    """
    Convertit une serie en datetime naive.
    """
    converted = pd.to_datetime(series, errors="coerce")
    if getattr(converted.dt, "tz", None) is not None:
        converted = converted.dt.tz_convert(None)
    return converted


def _to_timestamp(value: Any, label: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"Impossible de convertir `{label}` en Timestamp : {value}")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp


def _prepare_temporal_frame(
    df: pd.DataFrame,
    *,
    datetime_col: str,
    horizon_col: str,
    target_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Prepare une copie triee du dataset pour les splits temporels.
    """
    required_columns = [datetime_col, horizon_col]
    if target_col is not None:
        required_columns.append(target_col)
    _ensure_required_columns(df, required_columns)

    work = df.copy()
    work[datetime_col] = _normalize_datetime_series(work[datetime_col])
    if work[datetime_col].isna().any():
        raise ValueError(
            f"La colonne `{datetime_col}` contient des valeurs non convertibles en datetime."
        )

    work["__row_order"] = np.arange(len(work), dtype=int)
    sort_columns = [datetime_col]
    for optional_column in ["asset", "indicator_name", horizon_col]:
        if optional_column in work.columns and optional_column not in sort_columns:
            sort_columns.append(optional_column)
    sort_columns.append("__row_order")

    work = work.sort_values(sort_columns).reset_index(drop=True)
    return work


def _resolve_horizon_steps_map(
    df: pd.DataFrame,
    *,
    horizon_col: str,
    horizon_steps_map: Optional[Mapping[str, int]],
) -> Dict[str, int]:
    """
    Resout le mapping `horizon -> nombre de pas futurs`.
    """
    resolved = horizon_steps_map
    if resolved is None:
        attrs_map = df.attrs.get("normalized_horizons")
        if isinstance(attrs_map, ABCMapping):
            resolved = attrs_map

    if resolved is None:
        raise ValueError(
            "`horizon_steps_map` est obligatoire pour appliquer une purge par horizon. "
            "Aucun mapping n'a ete fourni et aucun mapping `normalized_horizons` "
            "n'a ete trouve dans `df.attrs`."
        )

    normalized: Dict[str, int] = {}
    for key, value in resolved.items():
        step_int = int(value)
        if step_int <= 0:
            raise ValueError(f"Horizon invalide `{key}` -> {value}.")
        normalized[str(key)] = step_int

    dataset_horizons = pd.Series(df[horizon_col]).dropna().astype(str).unique().tolist()
    missing_horizons = [horizon for horizon in dataset_horizons if horizon not in normalized]
    if missing_horizons:
        raise ValueError(
            "Certains horizons du dataset sont absents de `horizon_steps_map` : "
            + ", ".join(sorted(missing_horizons))
        )

    return normalized


def _validate_fixed_split_boundaries(
    *,
    train_start: Any,
    train_end: Any,
    val_start: Any,
    val_end: Any,
    test_start: Any,
    test_end: Any,
) -> Dict[str, pd.Timestamp]:
    """
    Verifie que les bornes temporelles du split fixe ne se chevauchent pas.
    """
    timestamps = {
        "train_start": _to_timestamp(train_start, "train_start"),
        "train_end": _to_timestamp(train_end, "train_end"),
        "val_start": _to_timestamp(val_start, "val_start"),
        "val_end": _to_timestamp(val_end, "val_end"),
        "test_start": _to_timestamp(test_start, "test_start"),
        "test_end": _to_timestamp(test_end, "test_end"),
    }

    if timestamps["train_start"] > timestamps["train_end"]:
        raise ValueError("`train_start` doit etre inferieur ou egal a `train_end`.")
    if timestamps["val_start"] > timestamps["val_end"]:
        raise ValueError("`val_start` doit etre inferieur ou egal a `val_end`.")
    if timestamps["test_start"] > timestamps["test_end"]:
        raise ValueError("`test_start` doit etre inferieur ou egal a `test_end`.")

    if not (
        timestamps["train_end"] < timestamps["val_start"]
        and timestamps["val_end"] < timestamps["test_start"]
    ):
        raise ValueError(
            "Les periodes train / validation / test ne doivent pas se chevaucher "
            "et doivent etre strictement ordonnees."
        )

    return timestamps


def _slice_date_block(
    df: pd.DataFrame,
    *,
    datetime_col: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    mask = df[datetime_col].between(start, end, inclusive="both")
    return df.loc[mask].copy()


def _drop_missing_target(df: pd.DataFrame, *, target_col: str) -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"Colonne target introuvable : `{target_col}`.")
    return df.loc[pd.to_numeric(df[target_col], errors="coerce").notna()].copy()


def _build_block_calendar(
    df: pd.DataFrame,
    *,
    datetime_col: str,
    block_end: pd.Timestamp,
    block_start: Optional[pd.Timestamp] = None,
) -> pd.DatetimeIndex:
    calendar = pd.DatetimeIndex(pd.Series(df[datetime_col]).dropna().unique()).sort_values()
    if block_start is not None:
        calendar = calendar[calendar >= block_start]
    calendar = calendar[calendar <= block_end]
    return calendar


def _apply_horizon_purge_with_calendar(
    df: pd.DataFrame,
    *,
    datetime_col: str,
    horizon_col: str,
    horizon_steps_map: Mapping[str, int],
    block_end: pd.Timestamp,
    calendar_dates: Sequence[pd.Timestamp],
) -> pd.DataFrame:
    """
    Applique la purge en utilisant un calendrier temporel explicite.

    Convention :
    - l'horizon represente un nombre de pas futurs sur la grille temporelle
      du panel deja resample ;
    - une ligne a la date t et a l'horizon h est conservee uniquement si la
      date atteinte apres h pas reste dans le bloc courant.
    """
    if df.empty:
        return df.copy()

    work = df.copy()
    calendar = pd.DatetimeIndex(calendar_dates).sort_values().unique()
    if len(calendar) == 0:
        return work.iloc[0:0].copy()

    date_to_position = pd.Series(np.arange(len(calendar), dtype=int), index=calendar)
    position_to_date = pd.Series(calendar, index=np.arange(len(calendar), dtype=int))

    row_positions = pd.to_datetime(work[datetime_col], errors="coerce").map(date_to_position)
    horizon_steps = pd.Series(work[horizon_col], index=work.index).astype(str).map(horizon_steps_map)
    if horizon_steps.isna().any():
        missing = sorted(pd.Series(work.loc[horizon_steps.isna(), horizon_col]).dropna().astype(str).unique())
        raise ValueError(
            "Horizons absents du mapping de purge : " + ", ".join(missing)
        )

    future_positions = row_positions + horizon_steps
    future_end_dates = future_positions.map(position_to_date)
    keep_mask = future_end_dates.notna() & (future_end_dates <= block_end)

    filtered = work.loc[keep_mask].copy()
    filtered.attrs = deepcopy(getattr(df, "attrs", {}))
    filtered.attrs["purge_block_end"] = block_end
    filtered.attrs["purged_row_count"] = int((~keep_mask).sum())
    return filtered


def apply_horizon_purge(
    df: pd.DataFrame,
    *,
    datetime_col: str,
    horizon_col: str,
    horizon_steps_map: Mapping[str, int],
    block_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Retire les lignes dont l'horizon futur deborde hors du bloc courant.

    Cette version publique construit son calendrier de purge a partir des dates
    disponibles dans `df`. Pour les splits fixes et walk-forward, les helpers
    internes utilisent un calendrier explicite issu du dataset complet afin
    d'etre plus robustes en presence de dates absentes dans un sous-bloc.
    """
    _ensure_required_columns(df, [datetime_col, horizon_col])
    prepared = _prepare_temporal_frame(
        df,
        datetime_col=datetime_col,
        horizon_col=horizon_col,
        target_col=None,
    )
    block_end_ts = _to_timestamp(block_end, "block_end")
    calendar = _build_block_calendar(
        prepared,
        datetime_col=datetime_col,
        block_end=block_end_ts,
    )
    purged = _apply_horizon_purge_with_calendar(
        prepared,
        datetime_col=datetime_col,
        horizon_col=horizon_col,
        horizon_steps_map=horizon_steps_map,
        block_end=block_end_ts,
        calendar_dates=calendar,
    )
    return purged.drop(columns="__row_order", errors="ignore").reset_index(drop=True)


def temporal_train_val_test_split(
    df: pd.DataFrame,
    *,
    datetime_col: str = "datetime",
    horizon_col: str = "horizon",
    horizon_steps_map: Optional[Mapping[str, int]] = None,
    train_start: str = "2019-01-01",
    train_end: str = "2023-12-31",
    val_start: str = "2024-01-01",
    val_end: str = "2024-12-31",
    test_start: str = "2025-01-01",
    test_end: str = "2025-12-31",
    purge: bool = True,
    drop_na_target: bool = True,
    target_col: str = "target",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Construit un split strictement temporel train / validation / test.

    Split principal par defaut :
    - train : 2019-01-01 -> 2023-12-31
    - validation : 2024-01-01 -> 2024-12-31
    - test : 2025-01-01 -> 2025-12-31

    Purge :
    - fin du train purgee pour que l'horizon cible reste dans le train ;
    - fin de la validation purgee pour que l'horizon cible reste dans la validation ;
    - le test n'est pas purge vers l'avant pour proteger un bloc suivant.
    """
    boundaries = _validate_fixed_split_boundaries(
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
    )

    prepared = _prepare_temporal_frame(
        df,
        datetime_col=datetime_col,
        horizon_col=horizon_col,
        target_col=target_col,
    )
    resolved_horizon_steps = _resolve_horizon_steps_map(
        prepared,
        horizon_col=horizon_col,
        horizon_steps_map=horizon_steps_map,
    )
    full_calendar = _build_block_calendar(
        prepared,
        datetime_col=datetime_col,
        block_end=boundaries["test_end"],
        block_start=boundaries["train_start"],
    )

    train_df = _slice_date_block(
        prepared,
        datetime_col=datetime_col,
        start=boundaries["train_start"],
        end=boundaries["train_end"],
    )
    val_df = _slice_date_block(
        prepared,
        datetime_col=datetime_col,
        start=boundaries["val_start"],
        end=boundaries["val_end"],
    )
    test_df = _slice_date_block(
        prepared,
        datetime_col=datetime_col,
        start=boundaries["test_start"],
        end=boundaries["test_end"],
    )

    if purge:
        train_calendar = full_calendar[
            (full_calendar >= boundaries["train_start"])
            & (full_calendar <= boundaries["train_end"])
        ]
        val_calendar = full_calendar[
            (full_calendar >= boundaries["val_start"])
            & (full_calendar <= boundaries["val_end"])
        ]

        train_df = _apply_horizon_purge_with_calendar(
            train_df,
            datetime_col=datetime_col,
            horizon_col=horizon_col,
            horizon_steps_map=resolved_horizon_steps,
            block_end=boundaries["train_end"],
            calendar_dates=train_calendar,
        )
        val_df = _apply_horizon_purge_with_calendar(
            val_df,
            datetime_col=datetime_col,
            horizon_col=horizon_col,
            horizon_steps_map=resolved_horizon_steps,
            block_end=boundaries["val_end"],
            calendar_dates=val_calendar,
        )

    if drop_na_target:
        train_df = _drop_missing_target(train_df, target_col=target_col)
        val_df = _drop_missing_target(val_df, target_col=target_col)
        test_df = _drop_missing_target(test_df, target_col=target_col)

    outputs = []
    for split_name, split_df in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:
        cleaned = split_df.drop(columns="__row_order", errors="ignore").reset_index(drop=True)
        cleaned.attrs = deepcopy(getattr(df, "attrs", {}))
        cleaned.attrs["split_name"] = split_name
        cleaned.attrs["purge_enabled"] = purge
        cleaned.attrs["horizon_steps_map"] = dict(resolved_horizon_steps)
        outputs.append(cleaned)

    return outputs[0], outputs[1], outputs[2]


def summarize_temporal_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    datetime_col: str = "datetime",
    asset_col: str = "asset",
    horizon_col: str = "horizon",
    target_col: str = "target",
) -> pd.DataFrame:
    """
    Retourne un resume compact des trois blocs temporels.
    """
    rows: List[Dict[str, Any]] = []

    for split_name, split_df in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:
        _ensure_required_columns(split_df, [datetime_col, horizon_col, target_col])
        dates = _normalize_datetime_series(split_df[datetime_col])
        target = pd.to_numeric(split_df[target_col], errors="coerce")
        target_valid = target.notna()

        row: Dict[str, Any] = {
            "split": split_name,
            "num_rows": int(len(split_df)),
            "date_min": dates.min() if len(split_df) else pd.NaT,
            "date_max": dates.max() if len(split_df) else pd.NaT,
            "num_assets": (
                int(split_df[asset_col].nunique(dropna=True))
                if asset_col in split_df.columns
                else np.nan
            ),
            "num_indicators": (
                int(split_df["indicator_name"].nunique(dropna=True))
                if "indicator_name" in split_df.columns
                else np.nan
            ),
            "horizon_distribution": split_df[horizon_col].value_counts(dropna=False).to_dict(),
            "target_non_missing_rate": float(target_valid.mean()) if len(split_df) else np.nan,
            "target_positive_rate": (
                float((target[target_valid] > 0).mean()) if target_valid.any() else np.nan
            ),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def _validate_walk_forward_windows(
    validation_windows: Sequence[Tuple[str, str]],
    *,
    first_train_end: pd.Timestamp,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Normalise et verifie les fenetres de validation expanding.
    """
    normalized_windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    previous_end = first_train_end

    for idx, (start, end) in enumerate(validation_windows):
        start_ts = _to_timestamp(start, f"validation_windows[{idx}][0]")
        end_ts = _to_timestamp(end, f"validation_windows[{idx}][1]")
        if start_ts > end_ts:
            raise ValueError(
                f"Fenetre de validation invalide a l'indice {idx} : start > end."
            )
        if previous_end >= start_ts:
            raise ValueError(
                "Les fenetres walk-forward doivent etre strictement posterieures "
                "a la fin du train precedent."
            )
        normalized_windows.append((start_ts, end_ts))
        previous_end = end_ts

    return normalized_windows


def generate_expanding_walk_forward_splits(
    df: pd.DataFrame,
    *,
    datetime_col: str = "datetime",
    horizon_col: str = "horizon",
    horizon_steps_map: Mapping[str, int],
    train_start: str = "2019-01-01",
    first_train_end: str = "2022-12-31",
    validation_windows: Optional[Sequence[Tuple[str, str]]] = None,
    purge: bool = True,
    drop_na_target: bool = True,
    target_col: str = "target",
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Genere des folds walk-forward expanding strictement temporels.

    Par defaut :
    - Fold 1: train 2019-01-01 -> 2022-12-31 | val 2023-01-01 -> 2023-06-30
    - Fold 2: train 2019-01-01 -> 2023-06-30 | val 2023-07-01 -> 2023-12-31
    - Fold 3: train 2019-01-01 -> 2023-12-31 | val 2024-01-01 -> 2024-06-30
    - Fold 4: train 2019-01-01 -> 2024-06-30 | val 2024-07-01 -> 2024-12-31
    """
    if validation_windows is None:
        validation_windows = DEFAULT_WALK_FORWARD_WINDOWS

    prepared = _prepare_temporal_frame(
        df,
        datetime_col=datetime_col,
        horizon_col=horizon_col,
        target_col=target_col,
    )
    resolved_horizon_steps = _resolve_horizon_steps_map(
        prepared,
        horizon_col=horizon_col,
        horizon_steps_map=horizon_steps_map,
    )

    train_start_ts = _to_timestamp(train_start, "train_start")
    first_train_end_ts = _to_timestamp(first_train_end, "first_train_end")
    normalized_windows = _validate_walk_forward_windows(
        validation_windows,
        first_train_end=first_train_end_ts,
    )
    full_calendar = _build_block_calendar(
        prepared,
        datetime_col=datetime_col,
        block_start=train_start_ts,
        block_end=normalized_windows[-1][1],
    )

    folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    current_train_end = first_train_end_ts

    for fold_idx, (val_start_ts, val_end_ts) in enumerate(normalized_windows, start=1):
        train_fold = _slice_date_block(
            prepared,
            datetime_col=datetime_col,
            start=train_start_ts,
            end=current_train_end,
        )
        val_fold = _slice_date_block(
            prepared,
            datetime_col=datetime_col,
            start=val_start_ts,
            end=val_end_ts,
        )

        if purge:
            train_calendar = full_calendar[
                (full_calendar >= train_start_ts) & (full_calendar <= current_train_end)
            ]
            val_calendar = full_calendar[
                (full_calendar >= val_start_ts) & (full_calendar <= val_end_ts)
            ]

            train_fold = _apply_horizon_purge_with_calendar(
                train_fold,
                datetime_col=datetime_col,
                horizon_col=horizon_col,
                horizon_steps_map=resolved_horizon_steps,
                block_end=current_train_end,
                calendar_dates=train_calendar,
            )
            val_fold = _apply_horizon_purge_with_calendar(
                val_fold,
                datetime_col=datetime_col,
                horizon_col=horizon_col,
                horizon_steps_map=resolved_horizon_steps,
                block_end=val_end_ts,
                calendar_dates=val_calendar,
            )

        if drop_na_target:
            train_fold = _drop_missing_target(train_fold, target_col=target_col)
            val_fold = _drop_missing_target(val_fold, target_col=target_col)

        train_output = train_fold.drop(columns="__row_order", errors="ignore").reset_index(drop=True)
        val_output = val_fold.drop(columns="__row_order", errors="ignore").reset_index(drop=True)

        train_output.attrs = deepcopy(getattr(df, "attrs", {}))
        train_output.attrs["split_name"] = f"train_fold_{fold_idx}"
        train_output.attrs["fold"] = fold_idx
        train_output.attrs["horizon_steps_map"] = dict(resolved_horizon_steps)

        val_output.attrs = deepcopy(getattr(df, "attrs", {}))
        val_output.attrs["split_name"] = f"validation_fold_{fold_idx}"
        val_output.attrs["fold"] = fold_idx
        val_output.attrs["horizon_steps_map"] = dict(resolved_horizon_steps)

        folds.append((train_output, val_output))
        current_train_end = val_end_ts

    return folds


def _extract_datetime_series_for_cv(
    X: pd.DataFrame | pd.Series | np.ndarray,
    *,
    datetime_col: str,
) -> pd.Series:
    """
    Extrait la serie temporelle utilisee par le cross-validator.

    Cas supportes :
    - DataFrame avec colonne `datetime_col`
    - DataFrame/Series avec `DatetimeIndex`
    - MultiIndex avec un niveau nomme `datetime_col`
    """
    if isinstance(X, pd.DataFrame):
        if datetime_col in X.columns:
            return _normalize_datetime_series(X[datetime_col]).dt.normalize()
        if isinstance(X.index, pd.MultiIndex) and datetime_col in X.index.names:
            values = X.index.get_level_values(datetime_col)
            return _normalize_datetime_series(pd.Series(values, index=X.index)).dt.normalize()
        if isinstance(X.index, pd.DatetimeIndex):
            return _normalize_datetime_series(pd.Series(X.index, index=X.index)).dt.normalize()

    if isinstance(X, pd.Series):
        if isinstance(X.index, pd.MultiIndex) and datetime_col in X.index.names:
            values = X.index.get_level_values(datetime_col)
            return _normalize_datetime_series(pd.Series(values, index=X.index)).dt.normalize()
        if isinstance(X.index, pd.DatetimeIndex):
            return _normalize_datetime_series(pd.Series(X.index, index=X.index)).dt.normalize()

    raise ValueError(
        "Impossible d'extraire les dates pour le cross-validator. "
        f"Le DataFrame doit fournir la colonne `{datetime_col}` ou un index temporel compatible."
    )


def _extract_horizon_series_for_cv(
    X: pd.DataFrame | pd.Series | np.ndarray,
    *,
    horizon_col: str,
) -> pd.Series:
    """
    Extrait la serie d'horizons pour la purge horizon-aware.
    """
    if isinstance(X, pd.DataFrame):
        if horizon_col in X.columns:
            return pd.Series(X[horizon_col], index=X.index).astype(str)
        if isinstance(X.index, pd.MultiIndex) and horizon_col in X.index.names:
            values = X.index.get_level_values(horizon_col)
            return pd.Series(values, index=X.index).astype(str)

    raise ValueError(
        "Impossible d'extraire les horizons pour le cross-validator. "
        f"Le DataFrame doit fournir la colonne `{horizon_col}`."
    )


class WalkForwardRolling(BaseCrossValidator):
    """
    Cross-validator sklearn-compatible en rolling window fixe avec embargo.

    Philosophie :
    - train de taille fixe `period_train`
    - embargo de longueur `period_embargo`
    - test suivant de longueur `period_test`
    - pas d'expanding window
    - pas de validation separee

    Purge horizon-aware :
    - une ligne du train a la date t et a l'horizon h est retiree si son futur
      deborde au-dela du debut de l'embargo, ou du test s'il n'y a pas d'embargo.
    - le test n'est pas purge vers l'avant ; les lignes sans target pourront etre
      supprimees ensuite lors du scoring / reporting.

    Convention sur les periodes :
    - les tailles `period_train`, `period_test`, `period_embargo` sont exprimees
      en nombre de dates uniques normalisees (typiquement des jours).
    """

    def __init__(
        self,
        period_train: int,
        period_test: int,
        period_embargo: int = 0,
        *,
        datetime_col: str = "date",
        horizon_col: str = "horizon",
        horizon_steps_map: Optional[Mapping[str, int]] = None,
        drop_unknown_horizons: bool = True,
        verbose: int = 0,
    ) -> None:
        if period_train <= 0:
            raise ValueError("`period_train` doit etre strictement positif.")
        if period_test <= 0:
            raise ValueError("`period_test` doit etre strictement positif.")
        if period_embargo < 0:
            raise ValueError("`period_embargo` doit etre positif ou nul.")

        self.period_train = int(period_train)
        self.period_test = int(period_test)
        self.period_embargo = int(period_embargo)
        self.datetime_col = datetime_col
        self.horizon_col = horizon_col
        self.horizon_steps_map = dict(horizon_steps_map) if horizon_steps_map is not None else None
        self.drop_unknown_horizons = bool(drop_unknown_horizons)
        self.verbose = int(verbose)
        self.fold_summaries_: List[Dict[str, Any]] = []

    def _prepare_split_inputs(
        self,
        X: pd.DataFrame | pd.Series | np.ndarray,
    ) -> Tuple[pd.Series, pd.Series, pd.DatetimeIndex, Dict[str, int], pd.Series]:
        datetimes = _extract_datetime_series_for_cv(X, datetime_col=self.datetime_col)
        if datetimes.isna().any():
            raise ValueError("Certaines dates sont invalides dans le cross-validator.")

        horizons = _extract_horizon_series_for_cv(X, horizon_col=self.horizon_col)
        if len(horizons) != len(datetimes):
            raise ValueError("Nombre d'horizons incoherent avec le nombre de lignes.")

        if self.horizon_steps_map is None:
            raise ValueError(
                "`horizon_steps_map` est obligatoire dans WalkForwardRolling pour appliquer la purge."
            )
        horizon_steps_map = {str(key): int(value) for key, value in self.horizon_steps_map.items()}
        invalid_horizons = [key for key, value in horizon_steps_map.items() if value <= 0]
        if invalid_horizons:
            raise ValueError(
                "Horizons invalides dans `horizon_steps_map` : " + ", ".join(invalid_horizons)
            )

        supported_mask = horizons.astype(str).isin(horizon_steps_map.keys())
        dataset_horizons = pd.Series(horizons).dropna().astype(str).unique().tolist()
        missing_horizons = [horizon for horizon in dataset_horizons if horizon not in horizon_steps_map]
        if missing_horizons:
            if not self.drop_unknown_horizons:
                raise ValueError(
                    "Certains horizons du dataset sont absents de `horizon_steps_map` : "
                    + ", ".join(sorted(missing_horizons))
                )
            LOGGER.warning(
                "Horizons absents de `horizon_steps_map` exclus du backtest : %s",
                ", ".join(sorted(missing_horizons)),
            )

        datetimes_supported = datetimes.loc[supported_mask]
        if datetimes_supported.empty:
            raise ValueError(
                "Aucune ligne du dataset ne correspond aux horizons fournis dans `horizon_steps_map`."
            )

        unique_dates = pd.DatetimeIndex(pd.Series(datetimes_supported).dropna().unique()).sort_values()
        if len(unique_dates) < self.period_train:
            raise ValueError(
                f"Pas assez de dates uniques pour construire le rolling window. "
                f"Minimum requis: {self.period_train}, observe: {len(unique_dates)}."
            )

        return datetimes, horizons, unique_dates, horizon_steps_map, supported_mask

    def get_n_splits(self, X=None, y=None, groups=None):
        if X is None:
            return None

        _, _, unique_dates, _, _ = self._prepare_split_inputs(X)
        n_splits = 0
        train_end_pos = self.period_train

        while True:
            embargo_end_pos = min(train_end_pos + self.period_embargo, len(unique_dates))
            test_start_pos = embargo_end_pos
            if test_start_pos >= len(unique_dates):
                break
            test_end_pos = min(test_start_pos + self.period_test, len(unique_dates))
            if test_start_pos >= test_end_pos:
                break
            n_splits += 1
            train_end_pos += self.period_test
            if train_end_pos > len(unique_dates):
                break

        return n_splits

    def split(self, X, y=None, groups=None):
        datetimes, horizons, unique_dates, horizon_steps_map, supported_mask = self._prepare_split_inputs(X)
        date_to_position = pd.Series(np.arange(len(unique_dates), dtype=int), index=unique_dates)
        row_positions = datetimes.map(date_to_position)
        horizon_steps = horizons.map(horizon_steps_map)

        self.fold_summaries_ = []
        train_end_pos = self.period_train
        fold_id = 0

        while True:
            train_start_pos = max(0, train_end_pos - self.period_train)
            embargo_start_pos = train_end_pos
            embargo_end_pos = min(embargo_start_pos + self.period_embargo, len(unique_dates))
            test_start_pos = embargo_end_pos
            test_end_pos = min(test_start_pos + self.period_test, len(unique_dates))

            if test_start_pos >= len(unique_dates):
                break
            if test_start_pos >= test_end_pos:
                break

            train_start_date = unique_dates[train_start_pos]
            train_end_date = unique_dates[train_end_pos - 1]
            test_start_date = unique_dates[test_start_pos]
            test_end_date = unique_dates[test_end_pos - 1]

            train_date_mask = supported_mask & datetimes.between(
                train_start_date,
                train_end_date,
                inclusive="both",
            )
            test_date_mask = supported_mask & datetimes.between(
                test_start_date,
                test_end_date,
                inclusive="both",
            )

            # Purge horizon-aware :
            # on retire du train les lignes dont le futur deborde au-dela du debut
            # de l'embargo (ou du test s'il n'y a pas d'embargo). En pratique,
            # ce cutoff correspond a `embargo_start_pos`.
            future_end_positions = row_positions + horizon_steps
            train_purge_mask = supported_mask & (future_end_positions < embargo_start_pos)
            train_mask = train_date_mask & train_purge_mask

            train_indices = np.flatnonzero(train_mask.to_numpy())
            test_indices = np.flatnonzero(test_date_mask.to_numpy())

            if len(test_indices) == 0:
                break

            fold_id += 1
            fold_summary = {
                "fold_id": fold_id,
                "train_start": train_start_date,
                "train_end": train_end_date,
                "test_start": test_start_date,
                "test_end": test_end_date,
                "period_train": self.period_train,
                "period_test": self.period_test,
                "period_embargo": self.period_embargo,
                "n_train_rows_raw": int(train_date_mask.sum()),
                "n_train_rows_after_purge": int(len(train_indices)),
                "n_train_rows_purged": int(train_date_mask.sum() - len(train_indices)),
                "n_test_rows": int(len(test_indices)),
            }
            self.fold_summaries_.append(fold_summary)

            if self.verbose >= 1:
                LOGGER.info(
                    "WalkForwardRolling fold %d | train [%s -> %s] | embargo=%d | test [%s -> %s] | train rows %d -> %d | test rows %d",
                    fold_id,
                    train_start_date.date(),
                    train_end_date.date(),
                    self.period_embargo,
                    test_start_date.date(),
                    test_end_date.date(),
                    fold_summary["n_train_rows_raw"],
                    fold_summary["n_train_rows_after_purge"],
                    fold_summary["n_test_rows"],
                )

            yield train_indices, test_indices

            train_end_pos += self.period_test
            if train_end_pos > len(unique_dates):
                break


def _add_prediction_rank_columns(
    df: pd.DataFrame,
    *,
    prob_col: str,
    rank_group_cols: Sequence[str],
) -> pd.DataFrame:
    """
    Ajoute les colonnes de ranking a partir d'une probabilite de succes.
    """
    ranked = df.copy()
    missing_group_cols = [column for column in rank_group_cols if column not in ranked.columns]
    if missing_group_cols:
        raise ValueError(
            "Colonnes de groupe de ranking absentes : "
            f"{', '.join(missing_group_cols)}"
        )

    grouped = ranked.groupby(list(rank_group_cols), sort=False)[prob_col]
    ranked["rank_pred_proba"] = grouped.rank(method="first", ascending=False)
    ranked["rank_pct_pred_proba"] = grouped.rank(method="average", ascending=True, pct=True)
    return ranked


def _resolve_penalized_logit_params(
    *,
    penalty_type: str,
    C: float,
    l1_ratio: float,
    max_iter: int,
    random_state: int,
    class_weight: Optional[str | Mapping[Any, float]],
) -> Dict[str, Any]:
    """
    Traduit `penalty_type` en parametres sklearn compatibles.

    Choix de solveur :
    - `l2` : `lbfgs`, en general plus stable et plus rapide que `saga`
      pour une logistique dense/one-hot classique ;
    - `l1` : `saga` ;
    - `elasticnet` : `saga`.
    """
    penalty = str(penalty_type).lower()
    if penalty not in {"l1", "l2", "elasticnet"}:
        raise ValueError("penalty_type doit valoir `l1`, `l2` ou `elasticnet`.")
    if C <= 0:
        raise ValueError("`C` doit etre strictement positif.")
    if penalty == "elasticnet" and not 0 <= l1_ratio <= 1:
        raise ValueError("`l1_ratio` doit etre dans [0, 1] pour Elastic Net.")

    solver = "lbfgs" if penalty == "l2" else "saga"
    params: Dict[str, Any] = {
        "penalty": penalty,
        "C": float(C),
        "solver": solver,
        "max_iter": int(max_iter),
        "random_state": int(random_state),
        "class_weight": class_weight,
    }
    if solver == "saga":
        params["n_jobs"] = None
    if penalty == "elasticnet":
        params["l1_ratio"] = float(l1_ratio)

    return params


def build_penalized_logit_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    *,
    penalty_type: str = "elasticnet",
    C: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 5000,
    random_state: int = 42,
    class_weight: Optional[str | Mapping[Any, float]] = None,
    scaling_mode: str = "classical",
    datetime_col: str = "datetime",
    asset_col: str = "asset",
    horizon_col: str = "horizon",
) -> Pipeline:
    """
    Construit un pipeline sklearn complet pour logistic regression penalisee.

    Le preprocessing est delegue a `preprocessing.build_logit_preprocessor`.
    Avec `scaling_mode="classical"`, le `StandardScaler` est fitte uniquement
    sur le train via sklearn. Avec `cross_sectional`, les numeriques sont
    standardises par groupe a date t, sans moyenne globale temporelle.
    """
    preprocessor = build_logit_preprocessor(
        numeric_features=list(numeric_features),
        categorical_features=list(categorical_features),
        scaling_mode=scaling_mode,
        datetime_col=datetime_col,
        asset_col=asset_col,
        horizon_col=horizon_col,
    )
    model_params = _resolve_penalized_logit_params(
        penalty_type=penalty_type,
        C=C,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        random_state=random_state,
        class_weight=class_weight,
    )
    classifier = LogisticRegression(**model_params)
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )


def _ensure_binary_target(y: pd.Series) -> pd.Series:
    target = pd.to_numeric(y, errors="coerce")
    if target.isna().any():
        raise ValueError("La target contient des NaN apres preparation.")
    unique_values = sorted(target.unique().tolist())
    if not set(unique_values).issubset({0, 1, 0.0, 1.0}):
        raise ValueError(
            "La logistic regression attend une target binaire 0/1. "
            f"Valeurs observees : {unique_values[:10]}"
        )
    if target.nunique() < 2:
        raise ValueError(
            "La target du train contient une seule classe : impossible de fitter une logistique."
        )
    return target.astype(int)


def _align_model_input_frame(df_model: pd.DataFrame, required_columns: Sequence[str]) -> pd.DataFrame:
    """
    Aligne un DataFrame de scoring sur les colonnes vues au train.
    """
    aligned = df_model.copy()
    missing_columns = [column for column in required_columns if column not in aligned.columns]
    for column in missing_columns:
        LOGGER.warning(
            "Colonne absente au scoring et remplie avec NaN : %s",
            column,
        )
        aligned[column] = np.nan
    return aligned[list(required_columns)].copy()


def fit_penalized_logit(
    train_df: pd.DataFrame,
    *,
    target_col: str = "target",
    datetime_col: str = "datetime",
    asset_col: str = "asset",
    indicator_col: str = "indicator_name",
    family_col: str = "signal_family",
    horizon_col: str = "horizon",
    penalty_type: str = "elasticnet",
    C: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 5000,
    class_weight: Optional[str | Mapping[Any, float]] = None,
    scaling_mode: str = "classical",
    include_indicator_name: bool = True,
    include_signal_family: bool = True,
    include_horizon: bool = True,
    include_asset: bool = False,
    filter_active_only: bool = True,
) -> Dict[str, Any]:
    """
    Fit une logistic regression penalisee sur le bloc train uniquement.
    """
    inputs = prepare_logit_inputs(
        train_df,
        target_col=target_col,
        datetime_col=datetime_col,
        asset_col=asset_col,
        indicator_col=indicator_col,
        family_col=family_col,
        horizon_col=horizon_col,
        include_indicator_name=include_indicator_name,
        include_signal_family=include_signal_family,
        include_horizon=include_horizon,
        include_asset=include_asset,
        include_signal_active=False,
        filter_active_only=filter_active_only,
        drop_na_target=True,
    )
    y_train = _ensure_binary_target(inputs["y"])

    pipeline = build_penalized_logit_pipeline(
        inputs["numeric_features"],
        inputs["categorical_features"],
        penalty_type=penalty_type,
        C=C,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        class_weight=class_weight,
        scaling_mode=scaling_mode,
        datetime_col=datetime_col,
        asset_col=asset_col,
        horizon_col=horizon_col,
    )
    pipeline.fit(inputs["X_model"], y_train)

    return {
        "pipeline": pipeline,
        "penalty_type": str(penalty_type).lower(),
        "C": float(C),
        "l1_ratio": float(l1_ratio),
        "max_iter": int(max_iter),
        "class_weight": class_weight,
        "scaling_mode": scaling_mode,
        "numeric_features": inputs["numeric_features"],
        "categorical_features": inputs["categorical_features"],
        "feature_columns": inputs["feature_columns"],
        "model_input_columns": inputs["model_input_columns"],
        "context_feature_columns": inputs["context_feature_columns"],
        "df_train": inputs["df_model"],
        "X_train": inputs["X"],
        "y_train": y_train,
        "target_col": target_col,
        "datetime_col": datetime_col,
        "asset_col": asset_col,
        "indicator_col": indicator_col,
        "family_col": family_col,
        "horizon_col": horizon_col,
        "include_indicator_name": include_indicator_name,
        "include_signal_family": include_signal_family,
        "include_horizon": include_horizon,
        "include_asset": include_asset,
        "filter_active_only": filter_active_only,
        "n_train_obs": int(len(y_train)),
        "target_rate_train": float(y_train.mean()),
    }


def _positive_class_proba(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    classifier = pipeline.named_steps["model"]
    proba = pipeline.predict_proba(X)
    classes = list(classifier.classes_)
    if 1 not in classes:
        raise ValueError("La classe positive `1` est absente du modele fitte.")
    positive_idx = classes.index(1)
    return proba[:, positive_idx]


def score_penalized_logit(
    model_bundle: Mapping[str, Any],
    df: pd.DataFrame,
    *,
    target_col: str = "target",
    datetime_col: str = "datetime",
    asset_col: str = "asset",
    indicator_col: str = "indicator_name",
    family_col: str = "signal_family",
    horizon_col: str = "horizon",
    filter_active_only: bool = False,
    add_rank: bool = True,
    rank_group_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Score un dataset avec un modele logistique penalise deja fitte.
    """
    pipeline = model_bundle["pipeline"]
    inputs = prepare_logit_inputs(
        df,
        target_col=target_col,
        datetime_col=datetime_col,
        asset_col=asset_col,
        indicator_col=indicator_col,
        family_col=family_col,
        horizon_col=horizon_col,
        include_indicator_name=bool(model_bundle.get("include_indicator_name", True)),
        include_signal_family=bool(model_bundle.get("include_signal_family", True)),
        include_horizon=bool(model_bundle.get("include_horizon", True)),
        include_asset=bool(model_bundle.get("include_asset", False)),
        include_signal_active=False,
        filter_active_only=filter_active_only,
        drop_na_target=False,
    )
    X_score = _align_model_input_frame(
        inputs["df_model"],
        model_bundle["model_input_columns"],
    )
    scored = inputs["df_model"].copy()
    proba_success = _positive_class_proba(pipeline, X_score)
    scored["pred_proba_success"] = proba_success
    clipped_proba = np.clip(proba_success, 1e-12, 1.0 - 1e-12)
    scored["pred_log_odds"] = np.log(clipped_proba / (1.0 - clipped_proba))
    scored["pred_class"] = (proba_success >= 0.5).astype(int)

    if add_rank:
        if rank_group_cols is None:
            rank_group_cols = [datetime_col, asset_col, horizon_col]
        scored = _add_prediction_rank_columns(
            scored,
            prob_col="pred_proba_success",
            rank_group_cols=rank_group_cols,
        )

    return scored


def _safe_validation_metrics(y_true: pd.Series, y_prob: pd.Series) -> Dict[str, float]:
    valid_mask = y_true.notna() & y_prob.notna()
    y = pd.to_numeric(y_true[valid_mask], errors="coerce").astype(float)
    p = pd.to_numeric(y_prob[valid_mask], errors="coerce").astype(float)
    metrics: Dict[str, float] = {
        "n_obs": float(len(y)),
        "target_rate": float(y.mean()) if len(y) else np.nan,
        "log_loss": np.nan,
        "brier_score": np.nan,
        "roc_auc": np.nan,
    }
    if len(y) == 0:
        return metrics

    clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
    try:
        metrics["log_loss"] = float(log_loss(y, clipped, labels=[0, 1]))
    except Exception as exc:
        LOGGER.warning("log_loss impossible : %s", exc)
    try:
        metrics["brier_score"] = float(brier_score_loss(y, clipped))
    except Exception as exc:
        LOGGER.warning("brier_score impossible : %s", exc)
    if y.nunique() >= 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y, clipped))
        except Exception as exc:
            LOGGER.warning("roc_auc impossible : %s", exc)
    else:
        LOGGER.warning("roc_auc impossible : une seule classe dans la validation.")

    return metrics


def _compute_binary_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prob: pd.Series,
) -> Dict[str, float]:
    """
    Calcule un paquet standard de metriques binaire sur un fold OOS.
    """
    valid_mask = y_true.notna() & y_pred.notna() & y_prob.notna()
    y = pd.to_numeric(y_true[valid_mask], errors="coerce").astype(float)
    pred = pd.to_numeric(y_pred[valid_mask], errors="coerce").astype(float)
    prob = pd.to_numeric(y_prob[valid_mask], errors="coerce").astype(float)

    metrics = {
        "n_obs": float(len(y)),
        "target_rate": float(y.mean()) if len(y) else np.nan,
        "accuracy": np.nan,
        "f1": np.nan,
        "roc_auc": np.nan,
        "log_loss": np.nan,
        "brier_score": np.nan,
    }
    if len(y) == 0:
        return metrics

    metrics["accuracy"] = float(accuracy_score(y, pred))
    try:
        metrics["f1"] = float(f1_score(y, pred))
    except Exception as exc:
        LOGGER.warning("F1 impossible : %s", exc)

    clipped_prob = np.clip(prob, 1e-12, 1 - 1e-12)
    try:
        metrics["log_loss"] = float(log_loss(y, clipped_prob, labels=[0, 1]))
    except Exception as exc:
        LOGGER.warning("log_loss impossible : %s", exc)
    try:
        metrics["brier_score"] = float(brier_score_loss(y, clipped_prob))
    except Exception as exc:
        LOGGER.warning("brier_score impossible : %s", exc)
    if y.nunique() >= 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y, clipped_prob))
        except Exception as exc:
            LOGGER.warning("roc_auc impossible : %s", exc)
    else:
        LOGGER.warning("roc_auc impossible : une seule classe dans le fold test.")

    return metrics


@dataclass
class PenalizedLogisticTrainer:
    """
    Trainer fold-par-fold inspire de `LogisticTrainer(...).train_and_evaluate(...)`.

    Il reutilise :
    - le preprocessing du pipeline existant ;
    - la construction du pipeline sklearn ;
    - la logique de scoring probabiliste et de ranking.
    """

    cv: BaseCrossValidator
    datetime_col: str = "date"
    asset_col: str = "asset"
    indicator_col: str = "indicator_name"
    family_col: str = "signal_family"
    horizon_col: str = "horizon"
    target_col: str = "target"
    penalty_type: str = "elasticnet"
    C: float = 0.1
    l1_ratio: float = 0.5
    max_iter: int = 1000
    class_weight: Optional[str | Mapping[Any, float]] = None
    scaling_mode: str = "classical"
    include_indicator_name: bool = True
    include_signal_family: bool = True
    include_horizon: bool = True
    include_asset: bool = False
    filter_active_only: bool = True
    add_rank: bool = True
    verbose: int = 0

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Entraine la logistique penalisee fold par fold et reconstruit
        les predictions out-of-sample concatenees.
        """
        _ensure_required_columns(df, [self.datetime_col, self.horizon_col, self.target_col])
        prepared = _prepare_temporal_frame(
            df,
            datetime_col=self.datetime_col,
            horizon_col=self.horizon_col,
            target_col=self.target_col,
        )

        fold_rows: List[Dict[str, Any]] = []
        fold_metrics_rows: List[Dict[str, Any]] = []
        oos_frames: List[pd.DataFrame] = []
        fitted_bundles: List[Dict[str, Any]] = []

        for fold_id, (train_idx, test_idx) in enumerate(self.cv.split(prepared), start=1):
            train_fold = prepared.iloc[train_idx].drop(columns="__row_order", errors="ignore").copy()
            test_fold = prepared.iloc[test_idx].drop(columns="__row_order", errors="ignore").copy()

            if train_fold.empty or test_fold.empty:
                LOGGER.warning("Fold %d ignore car train ou test est vide.", fold_id)
                continue

            try:
                bundle = fit_penalized_logit(
                    train_fold,
                    target_col=self.target_col,
                    datetime_col=self.datetime_col,
                    asset_col=self.asset_col,
                    indicator_col=self.indicator_col,
                    family_col=self.family_col,
                    horizon_col=self.horizon_col,
                    penalty_type=self.penalty_type,
                    C=self.C,
                    l1_ratio=self.l1_ratio,
                    max_iter=self.max_iter,
                    class_weight=self.class_weight,
                    scaling_mode=self.scaling_mode,
                    include_indicator_name=self.include_indicator_name,
                    include_signal_family=self.include_signal_family,
                    include_horizon=self.include_horizon,
                    include_asset=self.include_asset,
                    filter_active_only=self.filter_active_only,
                )
                fitted_bundles.append(bundle)

                scored_test = score_penalized_logit(
                    bundle,
                    test_fold,
                    target_col=self.target_col,
                    datetime_col=self.datetime_col,
                    asset_col=self.asset_col,
                    indicator_col=self.indicator_col,
                    family_col=self.family_col,
                    horizon_col=self.horizon_col,
                    filter_active_only=self.filter_active_only,
                    add_rank=False,
                )
            except Exception as exc:
                LOGGER.warning("Fold %d ignore suite a une erreur de fit/score : %s", fold_id, exc)
                continue

            scored_test = scored_test.loc[
                pd.to_numeric(scored_test[self.target_col], errors="coerce").notna()
            ].copy()
            if scored_test.empty:
                LOGGER.warning("Fold %d ignore apres drop des targets manquantes.", fold_id)
                continue

            scored_test["fold_id"] = fold_id
            scored_test["y_true"] = pd.to_numeric(scored_test[self.target_col], errors="coerce")
            scored_test["y_pred"] = pd.to_numeric(scored_test["pred_class"], errors="coerce")
            scored_test["y_proba_pred"] = pd.to_numeric(scored_test["pred_proba_success"], errors="coerce")

            fold_summary = {}
            if hasattr(self.cv, "fold_summaries_") and len(getattr(self.cv, "fold_summaries_", [])) >= fold_id:
                fold_summary = deepcopy(self.cv.fold_summaries_[fold_id - 1])
                for key, value in fold_summary.items():
                    scored_test[key] = value
            else:
                fold_summary = {
                    "fold_id": fold_id,
                    "train_start": pd.to_datetime(train_fold[self.datetime_col]).min(),
                    "train_end": pd.to_datetime(train_fold[self.datetime_col]).max(),
                    "test_start": pd.to_datetime(test_fold[self.datetime_col]).min(),
                    "test_end": pd.to_datetime(test_fold[self.datetime_col]).max(),
                }

            metrics = _compute_binary_classification_metrics(
                scored_test["y_true"],
                scored_test["y_pred"],
                scored_test["y_proba_pred"],
            )
            metrics_row = {
                **fold_summary,
                **metrics,
                "n_train_used_for_fit": int(bundle["n_train_obs"]),
                "n_test_scored": int(len(scored_test)),
            }
            fold_metrics_rows.append(metrics_row)
            fold_rows.append(fold_summary)
            oos_frames.append(scored_test)

            if self.verbose >= 1:
                LOGGER.info(
                    "Fold %d | accuracy=%.4f | f1=%.4f | auc=%s | log_loss=%s | n_test=%d",
                    fold_id,
                    metrics["accuracy"] if pd.notna(metrics["accuracy"]) else np.nan,
                    metrics["f1"] if pd.notna(metrics["f1"]) else np.nan,
                    f"{metrics['roc_auc']:.4f}" if pd.notna(metrics["roc_auc"]) else "nan",
                    f"{metrics['log_loss']:.4f}" if pd.notna(metrics["log_loss"]) else "nan",
                    int(metrics["n_obs"]) if pd.notna(metrics["n_obs"]) else 0,
                )

        if not oos_frames:
            raise ValueError(
                "Aucune prediction out-of-sample n'a ete produite. "
                "Verifie les periodes du rolling window, l'embargo et le filtrage."
            )

        oos_predictions_df = pd.concat(oos_frames, axis=0, ignore_index=True)
        if self.add_rank:
            rank_group_cols = [self.datetime_col, self.asset_col, self.horizon_col]
            rank_group_cols = [col for col in rank_group_cols if col in oos_predictions_df.columns]
            if rank_group_cols:
                oos_predictions_df = _add_prediction_rank_columns(
                    oos_predictions_df,
                    prob_col="pred_proba_success",
                    rank_group_cols=rank_group_cols,
                )

        fold_metrics_df = pd.DataFrame(fold_metrics_rows).sort_values("fold_id").reset_index(drop=True)
        global_metrics = _compute_binary_classification_metrics(
            oos_predictions_df["y_true"],
            oos_predictions_df["y_pred"],
            oos_predictions_df["y_proba_pred"],
        )

        results = {
            "cv": self.cv,
            "trainer": self,
            "fitted_bundles": fitted_bundles,
            "fold_summaries": pd.DataFrame(fold_rows).sort_values("fold_id").reset_index(drop=True),
            "fold_metrics_df": fold_metrics_df,
            "accuracy_scores": pd.Series(fold_metrics_df["accuracy"].values, index=fold_metrics_df["fold_id"]),
            "f1_scores": pd.Series(fold_metrics_df["f1"].values, index=fold_metrics_df["fold_id"]),
            "roc_auc_scores": pd.Series(fold_metrics_df["roc_auc"].values, index=fold_metrics_df["fold_id"]),
            "log_loss_scores": pd.Series(fold_metrics_df["log_loss"].values, index=fold_metrics_df["fold_id"]),
            "brier_scores": pd.Series(fold_metrics_df["brier_score"].values, index=fold_metrics_df["fold_id"]),
            "y_true": oos_predictions_df["y_true"].reset_index(drop=True),
            "y_pred": oos_predictions_df["y_pred"].reset_index(drop=True),
            "y_proba_pred": oos_predictions_df["y_proba_pred"].reset_index(drop=True),
            "global_metrics": global_metrics,
            "oos_predictions_df": oos_predictions_df,
        }
        return results


def run_penalized_logistic_rolling_backtest(
    df: pd.DataFrame,
    *,
    datetime_col: str = "date",
    asset_col: str = "asset",
    indicator_col: str = "indicator_name",
    family_col: str = "signal_family",
    horizon_col: str = "horizon",
    target_col: str = "target",
    horizon_steps_map: Mapping[str, int],
    period_train: int,
    period_test: int,
    period_embargo: int,
    penalty_type: str = "elasticnet",
    C: float = 0.1,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    class_weight: Optional[str | Mapping[Any, float]] = None,
    scaling_mode: str = "classical",
    include_indicator_name: bool = True,
    include_signal_family: bool = True,
    include_horizon: bool = True,
    include_asset: bool = False,
    filter_active_only: bool = True,
    verbose: int = 0,
) -> Dict[str, Any]:
    """
    Backtest rolling window complet de la logistique penalisee.

    Workflow :
    1. creation du cross-validator `WalkForwardRolling`
    2. entrainement fold par fold
    3. predictions OOS concatenees
    4. calcul des metriques fold par fold et globales
    """
    cv = WalkForwardRolling(
        period_train=period_train,
        period_test=period_test,
        period_embargo=period_embargo,
        datetime_col=datetime_col,
        horizon_col=horizon_col,
        horizon_steps_map=horizon_steps_map,
        verbose=verbose,
    )
    trainer = PenalizedLogisticTrainer(
        cv=cv,
        datetime_col=datetime_col,
        asset_col=asset_col,
        indicator_col=indicator_col,
        family_col=family_col,
        horizon_col=horizon_col,
        target_col=target_col,
        penalty_type=penalty_type,
        C=C,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        class_weight=class_weight,
        scaling_mode=scaling_mode,
        include_indicator_name=include_indicator_name,
        include_signal_family=include_signal_family,
        include_horizon=include_horizon,
        include_asset=include_asset,
        filter_active_only=filter_active_only,
        add_rank=True,
        verbose=verbose,
    )
    return trainer.train_and_evaluate(df)


def train_validate_penalized_logit_grid(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    penalty_grid: Sequence[str] = ("l2", "l1", "elasticnet"),
    C_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    l1_ratio_grid: Sequence[float] = (0.2, 0.5, 0.8),
    scaling_mode_grid: Sequence[str] = ("classical",),
    target_col: str = "target",
    datetime_col: str = "datetime",
    asset_col: str = "asset",
    indicator_col: str = "indicator_name",
    family_col: str = "signal_family",
    horizon_col: str = "horizon",
    filter_active_only: bool = True,
) -> pd.DataFrame:
    """
    Entraine plusieurs configurations et evalue leur performance validation.
    """
    rows: List[Dict[str, Any]] = []

    for scaling_mode in scaling_mode_grid:
        for penalty_type in penalty_grid:
            ratios = l1_ratio_grid if str(penalty_type).lower() == "elasticnet" else [np.nan]
            for C in C_grid:
                for l1_ratio in ratios:
                    row: Dict[str, Any] = {
                        "penalty_type": str(penalty_type).lower(),
                        "C": float(C),
                        "l1_ratio": float(l1_ratio) if not pd.isna(l1_ratio) else np.nan,
                        "scaling_mode": scaling_mode,
                        "fit_ok": False,
                        "error": None,
                    }
                    try:
                        bundle = fit_penalized_logit(
                            train_df,
                            target_col=target_col,
                            datetime_col=datetime_col,
                            asset_col=asset_col,
                            indicator_col=indicator_col,
                            family_col=family_col,
                            horizon_col=horizon_col,
                            penalty_type=str(penalty_type),
                            C=float(C),
                            l1_ratio=float(l1_ratio) if not pd.isna(l1_ratio) else 0.5,
                            scaling_mode=scaling_mode,
                            filter_active_only=filter_active_only,
                        )
                        scored_val = score_penalized_logit(
                            bundle,
                            val_df,
                            target_col=target_col,
                            datetime_col=datetime_col,
                            asset_col=asset_col,
                            indicator_col=indicator_col,
                            family_col=family_col,
                            horizon_col=horizon_col,
                            filter_active_only=filter_active_only,
                            add_rank=False,
                        )
                        metrics = _safe_validation_metrics(
                            scored_val[target_col],
                            scored_val["pred_proba_success"],
                        )
                        row.update(metrics)
                        row["fit_ok"] = True
                        row["n_train_obs"] = bundle["n_train_obs"]
                        row["target_rate_train"] = bundle["target_rate_train"]
                    except Exception as exc:
                        LOGGER.warning(
                            "Configuration logit ignoree (%s, C=%s, l1_ratio=%s, scaling=%s) : %s",
                            penalty_type,
                            C,
                            l1_ratio,
                            scaling_mode,
                            exc,
                        )
                        row["error"] = str(exc)
                        row.update(
                            {
                                "n_obs": np.nan,
                                "target_rate": np.nan,
                                "log_loss": np.nan,
                                "brier_score": np.nan,
                                "roc_auc": np.nan,
                                "n_train_obs": np.nan,
                                "target_rate_train": np.nan,
                            }
                        )

                    rows.append(row)

    results = pd.DataFrame(rows)
    return results.sort_values(
        ["fit_ok", "log_loss", "brier_score"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def _get_preprocessed_feature_names(pipeline: Pipeline) -> List[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            return list(preprocessor.get_feature_names_out())
        except Exception:
            pass
    if isinstance(preprocessor, Pipeline) and "columns" in preprocessor.named_steps:
        return list(preprocessor.named_steps["columns"].get_feature_names_out())
    raise ValueError("Impossible d'extraire les noms des features preprocessees.")


def _clean_preprocessed_feature_name(
    feature_name: str,
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> Tuple[str, str, Optional[str]]:
    if feature_name.startswith("numeric__"):
        clean = feature_name.replace("numeric__", "", 1)
        return clean, "numeric", clean

    if feature_name.startswith("categorical__"):
        clean = feature_name.replace("categorical__", "", 1)
        for source in sorted(categorical_features, key=len, reverse=True):
            prefix = f"{source}_"
            if clean.startswith(prefix):
                return clean, "categorical", source
        return clean, "categorical", None

    return feature_name, "unknown", None


def extract_penalized_logit_coefficients(
    model_bundle: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Extrait les coefficients de la logistic regression apres preprocessing.
    """
    pipeline = model_bundle["pipeline"]
    classifier = pipeline.named_steps["model"]
    feature_names = _get_preprocessed_feature_names(pipeline)
    coefficients = np.asarray(classifier.coef_).ravel()
    if len(feature_names) != len(coefficients):
        raise ValueError(
            "Nombre de coefficients different du nombre de features preprocessees."
        )

    rows = []
    for feature_name, coef in zip(feature_names, coefficients):
        clean_name, feature_type, source_feature = _clean_preprocessed_feature_name(
            feature_name,
            numeric_features=model_bundle.get("numeric_features", []),
            categorical_features=model_bundle.get("categorical_features", []),
        )
        rows.append(
            {
                "feature_name": clean_name,
                "raw_feature_name": feature_name,
                "source_feature": source_feature,
                "feature_type": feature_type,
                "coefficient": float(coef),
                "abs_coefficient": float(abs(coef)),
            }
        )

    coef_df = pd.DataFrame(rows).sort_values(
        "abs_coefficient",
        ascending=False,
    ).reset_index(drop=True)
    coef_df.attrs["sparsity_summary"] = summarize_penalized_logit_sparsity(model_bundle)
    return coef_df


def summarize_penalized_logit_sparsity(
    model_bundle: Mapping[str, Any],
    *,
    zero_tol: float = 1e-12,
) -> Dict[str, Any]:
    """
    Resume la sparsité des coefficients, utile pour L1 / Elastic Net.
    """
    classifier = model_bundle["pipeline"].named_steps["model"]
    coefficients = np.asarray(classifier.coef_).ravel()
    zero_mask = np.abs(coefficients) <= zero_tol
    num_coefficients = int(len(coefficients))
    num_zero = int(zero_mask.sum())
    num_non_zero = int(num_coefficients - num_zero)
    return {
        "num_coefficients": num_coefficients,
        "num_zero_coefficients": num_zero,
        "num_non_zero_coefficients": num_non_zero,
        "zero_coefficient_rate": float(num_zero / num_coefficients) if num_coefficients else np.nan,
        "zero_tol": zero_tol,
    }


def _default_ranking_group_cols(columns: Sequence[str]) -> List[str]:
    date_col = "date" if "date" in columns else "datetime"
    group_cols = [date_col, "asset", "horizon"]
    return [column for column in group_cols if column in columns]


def _build_top_k_summary(
    group_detail_df: pd.DataFrame,
    *,
    k: int,
) -> Dict[str, Any]:
    """
    Construit un resume agrege des metriques top-k a partir du detail par groupe.

    Definitions retenues :
    - `mean_top_k_hit_rate` : moyenne simple des `top_k_hit_rate_group`
    - `mean_base_hit_rate` : moyenne simple des `base_hit_rate_group`
    - `mean_lift` : moyenne simple des `lift_group`
    - `aggregate_lift` : ratio des moyennes
      `mean_top_k_hit_rate / mean_base_hit_rate`

    Important :
    `mean_lift` et `aggregate_lift` peuvent differer de facon legitime car
    la moyenne des ratios n'est generalement pas egale au ratio des moyennes.
    """
    if group_detail_df.empty:
        return {
            "k": int(k),
            "num_groups": 0,
            "num_valid_lift_groups": 0,
            "mean_top_k_hit_rate": np.nan,
            "mean_base_hit_rate": np.nan,
            "mean_lift": np.nan,
            "aggregate_lift": np.nan,
            "lift_gap_mean_minus_aggregate": np.nan,
            "groups_with_base_rate_zero": 0,
            "groups_with_missing_target": 0,
            "groups_smaller_than_k": 0,
        }

    mean_top_k_hit_rate = float(group_detail_df["top_k_hit_rate_group"].mean())
    mean_base_hit_rate = float(group_detail_df["base_hit_rate_group"].mean())
    mean_lift = float(group_detail_df["lift_group"].mean())
    aggregate_lift = (
        float(mean_top_k_hit_rate / mean_base_hit_rate)
        if pd.notna(mean_base_hit_rate) and mean_base_hit_rate > 0
        else np.nan
    )

    summary = {
        "k": int(k),
        "num_groups": int(len(group_detail_df)),
        "num_valid_lift_groups": int(group_detail_df["lift_group"].notna().sum()),
        "mean_top_k_hit_rate": mean_top_k_hit_rate,
        "mean_base_hit_rate": mean_base_hit_rate,
        "mean_lift": mean_lift,
        "aggregate_lift": aggregate_lift,
        "lift_gap_mean_minus_aggregate": (
            float(mean_lift - aggregate_lift)
            if pd.notna(mean_lift) and pd.notna(aggregate_lift)
            else np.nan
        ),
        "groups_with_base_rate_zero": int((group_detail_df["base_hit_rate_group"] == 0).sum()),
        "groups_with_missing_target": int((group_detail_df["valid_target_count"] < group_detail_df["group_size"]).sum()),
        "groups_smaller_than_k": int((group_detail_df["group_size"] < k).sum()),
    }

    if summary["num_valid_lift_groups"] == 0 and summary["num_groups"] > 0:
        LOGGER.warning(
            "Aucun groupe n'a un `base_hit_rate_group` strictement positif : `mean_lift` est NaN."
        )

    if pd.notna(summary["mean_lift"]) and not np.isfinite(summary["mean_lift"]):
        LOGGER.warning("`mean_lift` n'est pas fini. Verifie les groupes a base_rate nulle.")

    if (
        pd.notna(summary["mean_lift"])
        and pd.notna(summary["aggregate_lift"])
        and abs(summary["lift_gap_mean_minus_aggregate"]) > 0.10
    ):
        LOGGER.info(
            "`mean_lift` differe sensiblement de `aggregate_lift` (gap=%.4f). "
            "C'est normal si la distribution des base rates varie entre groupes.",
            summary["lift_gap_mean_minus_aggregate"],
        )

    return summary


def evaluate_top_k_hit_rate(
    scored_df: pd.DataFrame,
    *,
    target_col: str = "target",
    prob_col: str = "pred_proba_success",
    group_cols: Optional[Sequence[str]] = None,
    k: int = 3,
    return_group_details: bool = True,
) -> pd.DataFrame | Dict[str, Any]:
    """
    Evalue le taux de succes des top-k lignes par groupe decisionnel.

    Definitions par groupe :
    1. on trie chaque groupe par `prob_col` decroissant ;
    2. on prend les `k` premieres lignes, ou moins si le groupe est plus petit ;
    3. on calcule :
       - `top_k_hit_rate_group`
       - `base_hit_rate_group`
       - `lift_group = top_k_hit_rate_group / base_hit_rate_group`
         si `base_hit_rate_group > 0`, sinon `NaN`.

    Definitions agregees :
    - `mean_top_k_hit_rate` = moyenne simple des `top_k_hit_rate_group`
    - `mean_base_hit_rate` = moyenne simple des `base_hit_rate_group`
    - `mean_lift` = moyenne simple des `lift_group`
    - `aggregate_lift` = `mean_top_k_hit_rate / mean_base_hit_rate`

    Important :
    `mean_lift` n'est pas forcement egal a `aggregate_lift` car
    moyenne des ratios != ratio des moyennes.

    Pour compatibilite avec le pipeline existant, la fonction retourne par
    defaut le detail par groupe (DataFrame) avec le resume dans `attrs["summary"]`.
    Si `return_group_details=False`, elle retourne directement le resume.
    """
    if k <= 0:
        raise ValueError("`k` doit etre strictement positif.")
    if group_cols is None:
        group_cols = _default_ranking_group_cols(scored_df.columns)
    if not group_cols:
        raise ValueError("Impossible de determiner les colonnes de groupe pour le ranking.")

    _ensure_required_columns(scored_df, [*group_cols, target_col, prob_col])
    work = scored_df.copy()
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work[prob_col] = pd.to_numeric(work[prob_col], errors="coerce")
    work = work.loc[work[prob_col].notna()].copy()
    if work.empty:
        empty = pd.DataFrame(
            columns=[
                *group_cols,
                "group_size",
                "valid_target_count",
                "top_k_count",
                "top_k_valid_target_count",
                "base_hit_rate_group",
                "top_k_hit_rate_group",
                "lift_group",
                "mean_pred_proba_group",
                "top_k_mean_pred_proba_group",
            ]
        )
        empty.attrs["summary"] = _build_top_k_summary(empty, k=k)
        if return_group_details:
            return empty
        return empty.attrs["summary"]

    dropped_missing_target = int(work[target_col].isna().sum())
    if dropped_missing_target > 0:
        LOGGER.info(
            "%d lignes avec target manquante sont exclues de l'evaluation top-k.",
            dropped_missing_target,
        )
    work = work.loc[work[target_col].notna()].copy()
    if work.empty:
        empty = pd.DataFrame(
            columns=[
                *group_cols,
                "group_size",
                "valid_target_count",
                "top_k_count",
                "top_k_valid_target_count",
                "base_hit_rate_group",
                "top_k_hit_rate_group",
                "lift_group",
                "mean_pred_proba_group",
                "top_k_mean_pred_proba_group",
            ]
        )
        empty.attrs["summary"] = _build_top_k_summary(empty, k=k)
        if return_group_details:
            return empty
        return empty.attrs["summary"]

    work["__rank_prob"] = work.groupby(list(group_cols), sort=False)[prob_col].rank(
        method="first",
        ascending=False,
    )
    top_k = work.loc[work["__rank_prob"] <= k].copy()

    base = work.groupby(list(group_cols), dropna=False).agg(
        group_size=(prob_col, "size"),
        valid_target_count=(target_col, "count"),
        base_hit_rate_group=(target_col, "mean"),
        mean_pred_proba_group=(prob_col, "mean"),
    )
    top = top_k.groupby(list(group_cols), dropna=False).agg(
        top_k_count=(prob_col, "size"),
        top_k_valid_target_count=(target_col, "count"),
        top_k_hit_rate_group=(target_col, "mean"),
        top_k_mean_pred_proba_group=(prob_col, "mean"),
    )
    result = base.join(top, how="left").reset_index()
    result["top_k_count"] = result["top_k_count"].fillna(0).astype(int)
    result["top_k_valid_target_count"] = result["top_k_valid_target_count"].fillna(0).astype(int)

    base_positive_mask = result["base_hit_rate_group"] > 0
    result["lift_group"] = np.where(
        base_positive_mask,
        result["top_k_hit_rate_group"] / result["base_hit_rate_group"],
        np.nan,
    )

    # Alias conserves pour retro-compatibilite avec les anciens notebooks.
    result["base_hit_rate"] = result["base_hit_rate_group"]
    result["top_k_hit_rate"] = result["top_k_hit_rate_group"]
    result["top_k_lift_vs_base"] = result["lift_group"]

    summary = _build_top_k_summary(result, k=k)
    summary["dropped_missing_target_rows"] = dropped_missing_target
    result.attrs["summary"] = summary
    result.attrs["summary_explanation"] = {
        "mean_lift": "moyenne simple des `lift_group` par groupe",
        "aggregate_lift": "ratio `mean_top_k_hit_rate / mean_base_hit_rate`",
        "difference_note": "Ces deux quantites peuvent differer car moyenne des ratios != ratio des moyennes.",
    }

    if return_group_details:
        return result
    return summary


__all__ = [
    "DEFAULT_WALK_FORWARD_WINDOWS",
    "WalkForwardRolling",
    "apply_horizon_purge",
    "build_penalized_logit_pipeline",
    "evaluate_top_k_hit_rate",
    "extract_penalized_logit_coefficients",
    "fit_penalized_logit",
    "generate_expanding_walk_forward_splits",
    "PenalizedLogisticTrainer",
    "run_penalized_logistic_rolling_backtest",
    "score_penalized_logit",
    "summarize_penalized_logit_sparsity",
    "summarize_temporal_split",
    "train_validate_penalized_logit_grid",
    "temporal_train_val_test_split",
]

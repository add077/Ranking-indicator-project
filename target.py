from __future__ import annotations

from collections.abc import Mapping as ABCMapping
import logging
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning


LOGGER = logging.getLogger(__name__)


DEFAULT_TARGET_CONFIG: Dict[str, Any] = {
    "general": {
        "copy": True,
        "print_summary": False,
        "duplicate_policy": "warn",  # "warn", "error", "ignore"
        "min_vol": 1e-12,
        "small_vol_policy": "nan",  # "nan" ou "epsilon"
        "vol_ddof": 1,
    },
    "columns": {
        "asset_candidates": ["asset", "symbol", "ticker", "crypto", "instrument"],
        "datetime_candidates": ["datetime", "date", "timestamp", "time"],
        "context_columns": None,
    },
    "returns": {
        "add_log_return": True,
        "vol_return_kind": "simple",  # "simple" ou "log"
        "vol_method": "rolling_std",
        "vol_window": 20,
        "vol_min_periods": None,
        "store_volatility_column": False,
    },
    "cross_sectional": {
        "min_cs_assets": 3,
        "min_cs_std": 1e-12,
        "small_std_policy": "nan",  # "nan" ou "epsilon"
        "cs_ddof": 1,
    },
    "target": {
        "tau": 0.1,
        "inactive_policy": "nan",  # "nan", "zero", "drop_later"
        "use_scaled_return": True,
        "target_scaling": "cross_sectional_future_return",
    },
    "validation": {
        "atol": 1e-10,
        "rtol": 1e-8,
    },
}


@dataclass
class TargetPanelContext:
    """
    Contexte de travail pour un panel cross-sectionnel d'actifs.
    """

    work: pd.DataFrame
    asset_col: str
    datetime_col: str
    context_cols: List[str]
    cs_group_cols: List[str]
    ts_group_cols: List[str]
    row_id_col: str = "__row_id"


def get_default_target_config() -> Dict[str, Any]:
    """
    Retourne une copie profonde de la configuration des targets.
    """
    return deepcopy(DEFAULT_TARGET_CONFIG)


def _deep_update(base: Dict[str, Any], updates: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if updates is None:
        return base

    for key, value in updates.items():
        if isinstance(value, ABCMapping) and isinstance(base.get(key), dict):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def _resolve_config(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return _deep_update(get_default_target_config(), config)


def _normalize_name(value: Any) -> str:
    return str(value).strip().lower()


def _find_existing_column(columns: Sequence[str], preferred: str, candidates: Sequence[str]) -> Optional[str]:
    normalized = {_normalize_name(col): col for col in columns}
    if _normalize_name(preferred) in normalized:
        return normalized[_normalize_name(preferred)]
    for candidate in candidates:
        if _normalize_name(candidate) in normalized:
            return normalized[_normalize_name(candidate)]
    return None


def _resolve_panel_columns(
    frame: pd.DataFrame,
    *,
    asset_col: str,
    datetime_col: str,
    config: Dict[str, Any],
) -> Tuple[str, str, List[str]]:
    asset_resolved = _find_existing_column(
        frame.columns,
        asset_col,
        config["columns"]["asset_candidates"],
    )
    datetime_resolved = _find_existing_column(
        frame.columns,
        datetime_col,
        config["columns"]["datetime_candidates"],
    )

    if asset_resolved is None:
        raise ValueError(
            "Impossible de trouver la colonne d'actif dans le panel. "
            f"Valeur demandee: `{asset_col}`."
        )
    if datetime_resolved is None:
        raise ValueError(
            "Impossible de trouver la colonne datetime dans le panel. "
            f"Valeur demandee: `{datetime_col}`."
        )

    context_cols = config["columns"].get("context_columns")
    if context_cols is None:
        potential_context = []
        for col in frame.columns:
            if col in {asset_resolved, datetime_resolved}:
                continue
            if col.startswith("__"):
                continue
            if pd.api.types.is_numeric_dtype(frame[col]) or pd.api.types.is_datetime64_any_dtype(frame[col]):
                continue
            potential_context.append(col)
        context_cols = potential_context

    return asset_resolved, datetime_resolved, list(context_cols)


def _prepare_target_panel(
    df: pd.DataFrame,
    *,
    asset_col: str,
    datetime_col: str,
    config: Dict[str, Any],
) -> TargetPanelContext:
    work = df.copy().reset_index()
    work["__row_id"] = np.arange(len(work), dtype=int)

    asset_resolved, datetime_resolved, context_cols = _resolve_panel_columns(
        work,
        asset_col=asset_col,
        datetime_col=datetime_col,
        config=config,
    )

    work[datetime_resolved] = pd.to_datetime(work[datetime_resolved], errors="coerce")
    if work[datetime_resolved].isna().any():
        LOGGER.warning("Certaines dates n'ont pas pu etre converties en datetime dans `%s`.", datetime_resolved)

    duplicate_mask = work.duplicated(
        subset=context_cols + [asset_resolved, datetime_resolved],
        keep=False,
    )
    duplicate_count = int(duplicate_mask.sum())
    duplicate_policy = str(config["general"]["duplicate_policy"])
    if duplicate_count > 0:
        message = (
            "Des doublons ont ete detectes sur (context, asset, datetime) : "
            f"{duplicate_count} lignes."
        )
        if duplicate_policy == "error":
            raise ValueError(message)
        if duplicate_policy == "warn":
            LOGGER.warning(message)

    sort_cols = context_cols + [asset_resolved, datetime_resolved, "__row_id"]
    work = work.sort_values(sort_cols).reset_index(drop=True)
    cs_group_cols = context_cols + [datetime_resolved]
    ts_group_cols = context_cols + [asset_resolved]

    return TargetPanelContext(
        work=work,
        asset_col=asset_resolved,
        datetime_col=datetime_resolved,
        context_cols=context_cols,
        cs_group_cols=cs_group_cols,
        ts_group_cols=ts_group_cols,
    )


def _restore_original_index(
    original_df: pd.DataFrame,
    work: pd.DataFrame,
    new_columns: Sequence[str],
) -> pd.DataFrame:
    restored = original_df.copy()
    aligned = work.sort_values("__row_id")
    if new_columns:
        overlapping = [column for column in new_columns if column in restored.columns]
        if overlapping:
            restored = restored.drop(columns=overlapping)
        added = aligned[list(new_columns)].reset_index(drop=True)
        added.index = restored.index
        restored = pd.concat([restored, added], axis=1)
    return restored


def normalize_horizons(
    horizons: int | Sequence[int] | Mapping[str, int],
) -> Dict[str, int]:
    """
    Normalise les horizons en mapping `{suffixe_colonne: nb_pas_futurs}`.

    Convention :
    - si `horizons` est une liste d'entiers `[1, 4, 24]`,
      les suffixes deviennent `h1`, `h4`, `h24`;
    - si `horizons` est un dictionnaire `{"4h": 4, "1d": 24}`,
      les suffixes deviennent `4h`, `1d`.
    """
    if isinstance(horizons, int):
        if horizons <= 0:
            raise ValueError("Un horizon doit etre strictement positif.")
        return {f"h{horizons}": int(horizons)}

    if isinstance(horizons, ABCMapping):
        normalized: Dict[str, int] = {}
        for label, step in horizons.items():
            step_int = int(step)
            if step_int <= 0:
                raise ValueError(f"Horizon invalide `{label}`: {step}.")
            suffix = str(label).strip().replace(" ", "_")
            if not suffix:
                raise ValueError("Le libelle d'horizon ne peut pas etre vide.")
            normalized[suffix] = step_int
        return normalized

    normalized = {}
    for step in horizons:
        step_int = int(step)
        if step_int <= 0:
            raise ValueError(f"Horizon invalide: {step}.")
        normalized[f"h{step_int}"] = step_int
    return normalized


def _groupby_series(series: pd.Series, frame: pd.DataFrame, group_cols: Sequence[str]):
    return series.groupby([frame[col] for col in group_cols], sort=False)


def _compute_future_shift(
    frame: pd.DataFrame,
    column: str,
    group_cols: Sequence[str],
    periods: int,
) -> pd.Series:
    grouped = frame.groupby(list(group_cols), sort=False)[column]
    return grouped.shift(-periods)


def _compute_past_returns(
    frame: pd.DataFrame,
    price_col: str,
    group_cols: Sequence[str],
    *,
    return_kind: str = "simple",
) -> pd.Series:
    prices = pd.to_numeric(frame[price_col], errors="coerce")
    prev_price = frame.groupby(list(group_cols), sort=False)[price_col].shift(1)
    prev_price = pd.to_numeric(prev_price, errors="coerce")

    if return_kind == "log":
        valid = (prices > 0) & (prev_price > 0)
        ratio = prices / prev_price
        return np.log(ratio.where(valid))

    return prices / prev_price - 1.0


def _compute_grouped_rolling_std(
    series: pd.Series,
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    *,
    window: int,
    min_periods: int,
    ddof: int,
) -> pd.Series:
    rolling = (
        _groupby_series(series, frame, group_cols)
        .rolling(window=window, min_periods=min_periods)
        .std(ddof=ddof)
        .reset_index(level=list(range(len(group_cols))), drop=True)
    )
    rolling.index = frame.index
    return rolling


def _compute_cross_sectional_group_metrics(
    frame: pd.DataFrame,
    column: str,
    group_cols: Sequence[str],
    *,
    ddof: int,
) -> pd.DataFrame:
    values = pd.to_numeric(frame[column], errors="coerce")
    temp = frame[list(group_cols)].copy()
    temp["__value"] = values
    metrics = (
        temp.groupby(list(group_cols), dropna=False, sort=False)["__value"]
        .agg(
            cs_valid_assets="count",
            cs_std_raw=lambda series: series.std(ddof=ddof),
        )
        .reset_index()
    )
    return metrics


def _broadcast_group_metrics(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    keyed = frame[list(group_cols) + ["__row_id"]].merge(
        metrics,
        on=list(group_cols),
        how="left",
        sort=False,
    )
    keyed = keyed.sort_values("__row_id").set_index(frame.index)
    return keyed[["cs_valid_assets", "cs_std_raw"]]


def _apply_small_value_policy(
    series: pd.Series,
    *,
    threshold: float,
    policy: str,
) -> pd.Series:
    if policy == "epsilon":
        return series.where(series.isna(), series.clip(lower=threshold))
    return series.where(series >= threshold)


def _future_return_column_name(suffix: str) -> str:
    return f"future_return_{suffix}"


def _future_log_return_column_name(suffix: str) -> str:
    return f"future_log_return_{suffix}"


def _future_scaled_return_column_name(suffix: str, *, return_kind: str = "simple") -> str:
    if return_kind == "log":
        return f"future_log_return_scaled_{suffix}"
    return f"future_return_scaled_{suffix}"


def _future_cs_std_column_name(suffix: str) -> str:
    return f"future_return_cs_std_{suffix}"


def _future_cs_scaled_return_column_name(suffix: str) -> str:
    return f"future_return_cs_scaled_{suffix}"


def _past_vol_column_name(
    *,
    return_kind: str,
    window: int,
) -> str:
    return f"past_vol_{return_kind}_{window}"


def compute_future_returns(
    df: pd.DataFrame,
    horizons: int | Sequence[int] | Mapping[str, int],
    *,
    price_col: str = "close",
    asset_col: str = "asset",
    datetime_col: str = "datetime",
    add_log_return: bool = True,
    config: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """
    Ajoute les returns futurs indexes a la date du signal t.

    Alignement choisi :
    - la ligne a la date t conserve le signal observe a t ;
    - `future_return_hX` stocke la performance entre t et t+h ;
    - les dernieres lignes de chaque actif n'ayant pas assez de futur restent a NaN.
    """
    cfg = _resolve_config(config)
    normalized_horizons = normalize_horizons(horizons)
    ctx = _prepare_target_panel(
        df,
        asset_col=asset_col,
        datetime_col=datetime_col,
        config=cfg,
    )

    if price_col not in ctx.work.columns:
        raise ValueError(f"Colonne de prix introuvable : `{price_col}`.")

    sorted_frame = ctx.work
    prices = pd.to_numeric(sorted_frame[price_col], errors="coerce")
    blocks: List[pd.DataFrame] = []
    created_cols: List[str] = []

    for suffix, step in normalized_horizons.items():
        future_price = _compute_future_shift(sorted_frame, price_col, ctx.ts_group_cols, step)
        future_price = pd.to_numeric(future_price, errors="coerce")

        future_return = future_price / prices - 1.0
        block = pd.DataFrame(
            {
                _future_return_column_name(suffix): future_return,
            },
            index=sorted_frame.index,
        )
        created_cols.extend(list(block.columns))

        if add_log_return:
            valid = (prices > 0) & (future_price > 0)
            future_log_return = np.log((future_price / prices).where(valid))
            log_col = _future_log_return_column_name(suffix)
            block[log_col] = future_log_return
            created_cols.append(log_col)

        blocks.append(block)

    if blocks:
        overwrite_cols = [column for block in blocks for column in block.columns if column in ctx.work.columns]
        if overwrite_cols:
            ctx.work = ctx.work.drop(columns=list(dict.fromkeys(overwrite_cols)))
        ctx.work = pd.concat([ctx.work] + blocks, axis=1)

    result = _restore_original_index(df, ctx.work, created_cols)
    result.attrs = deepcopy(getattr(df, "attrs", {}))
    result.attrs["added_future_return_columns"] = created_cols
    result.attrs["normalized_horizons"] = normalized_horizons
    return result


def compute_vol_scaled_future_returns(
    df: pd.DataFrame,
    horizons: int | Sequence[int] | Mapping[str, int],
    *,
    price_col: str = "close",
    asset_col: str = "asset",
    datetime_col: str = "datetime",
    vol_window: int = 20,
    min_periods: Optional[int] = None,
    return_kind: str = "simple",
    store_volatility_column: bool = False,
    config: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """
    Standardise les returns futurs par une volatilite passee backward-looking.

    La volatilite `sigma_t` est calculee uniquement a partir des retours connus
    jusqu'a t, par actif. Elle ne consomme jamais d'information future.
    """
    cfg = _resolve_config(config)
    cfg["returns"]["vol_window"] = vol_window
    cfg["returns"]["vol_min_periods"] = min_periods
    cfg["returns"]["vol_return_kind"] = return_kind
    cfg["returns"]["store_volatility_column"] = store_volatility_column

    normalized_horizons = normalize_horizons(horizons)
    add_log_return = return_kind == "log" or bool(cfg["returns"]["add_log_return"])
    with_returns = compute_future_returns(
        df,
        normalized_horizons,
        price_col=price_col,
        asset_col=asset_col,
        datetime_col=datetime_col,
        add_log_return=add_log_return,
        config=cfg,
    )
    ctx = _prepare_target_panel(
        with_returns,
        asset_col=asset_col,
        datetime_col=datetime_col,
        config=cfg,
    )

    if price_col not in ctx.work.columns:
        raise ValueError(f"Colonne de prix introuvable : `{price_col}`.")

    effective_min_periods = min_periods if min_periods is not None else vol_window
    if effective_min_periods <= 0:
        raise ValueError("`min_periods` doit etre strictement positif.")

    past_returns = _compute_past_returns(
        ctx.work,
        price_col=price_col,
        group_cols=ctx.ts_group_cols,
        return_kind=return_kind,
    )
    if cfg["returns"]["vol_method"] != "rolling_std":
        raise ValueError("Seule la methode `rolling_std` est supportee pour la volatilite.")

    past_vol = _compute_grouped_rolling_std(
        past_returns,
        ctx.work,
        ctx.ts_group_cols,
        window=vol_window,
        min_periods=effective_min_periods,
        ddof=int(cfg["general"]["vol_ddof"]),
    )
    past_vol = _apply_small_value_policy(
        past_vol,
        threshold=float(cfg["general"]["min_vol"]),
        policy=str(cfg["general"]["small_vol_policy"]),
    )

    blocks: List[pd.DataFrame] = []
    created_cols: List[str] = []
    if store_volatility_column:
        vol_col = _past_vol_column_name(return_kind=return_kind, window=vol_window)
        vol_block = pd.DataFrame({vol_col: past_vol}, index=ctx.work.index)
        blocks.append(vol_block)
        created_cols.append(vol_col)

    base_prefix = _future_log_return_column_name if return_kind == "log" else _future_return_column_name
    for suffix in normalized_horizons:
        base_return_col = base_prefix(suffix)
        if base_return_col not in ctx.work.columns:
            raise ValueError(
                f"Colonne de return futur introuvable pour l'horizon `{suffix}` : `{base_return_col}`."
            )
        scaled_col = _future_scaled_return_column_name(suffix, return_kind=return_kind)
        scaled_return = pd.to_numeric(ctx.work[base_return_col], errors="coerce") / past_vol
        block = pd.DataFrame({scaled_col: scaled_return}, index=ctx.work.index)
        blocks.append(block)
        created_cols.append(scaled_col)

    if blocks:
        overwrite_cols = [column for block in blocks for column in block.columns if column in ctx.work.columns]
        if overwrite_cols:
            ctx.work = ctx.work.drop(columns=list(dict.fromkeys(overwrite_cols)))
        ctx.work = pd.concat([ctx.work] + blocks, axis=1)

    result = _restore_original_index(with_returns, ctx.work, created_cols)
    result.attrs = deepcopy(getattr(with_returns, "attrs", {}))
    result.attrs["added_scaled_return_columns"] = created_cols
    return result


def compute_cross_sectionally_scaled_future_returns(
    df: pd.DataFrame,
    horizons: int | Sequence[int] | Mapping[str, int],
    *,
    price_col: str = "close",
    asset_col: str = "asset",
    datetime_col: str = "datetime",
    min_cs_assets: int = 3,
    min_cs_std: float = 1e-12,
    small_std_policy: str = "nan",
    config: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """
    Standardise les futurs returns par dispersion cross-sectionnelle a date t.

    Pour chaque horizon h :
    - `future_return_h` est calcule par actif entre t et t+h ;
    - a chaque date t, on calcule l'ecart-type cross-sectionnel entre actifs ;
    - `future_return_cs_scaled_h = future_return_h / sigma_cs_t`.

    La target reste ensuite portee par la ligne t du signal.
    """
    if min_cs_assets <= 0:
        raise ValueError("`min_cs_assets` doit etre strictement positif.")
    if min_cs_std <= 0:
        raise ValueError("`min_cs_std` doit etre strictement positif.")

    cfg = _resolve_config(config)
    cfg["cross_sectional"]["min_cs_assets"] = min_cs_assets
    cfg["cross_sectional"]["min_cs_std"] = min_cs_std
    cfg["cross_sectional"]["small_std_policy"] = small_std_policy

    normalized_horizons = normalize_horizons(horizons)
    with_returns = compute_future_returns(
        df,
        normalized_horizons,
        price_col=price_col,
        asset_col=asset_col,
        datetime_col=datetime_col,
        add_log_return=bool(cfg["returns"]["add_log_return"]),
        config=cfg,
    )
    ctx = _prepare_target_panel(
        with_returns,
        asset_col=asset_col,
        datetime_col=datetime_col,
        config=cfg,
    )

    blocks: List[pd.DataFrame] = []
    created_cols: List[str] = []
    cs_group_summary: Dict[str, Dict[str, int]] = {}
    cs_ddof = int(cfg["cross_sectional"]["cs_ddof"])

    for suffix in normalized_horizons:
        future_return_col = _future_return_column_name(suffix)
        if future_return_col not in ctx.work.columns:
            raise ValueError(
                f"Colonne de future return introuvable pour l'horizon `{suffix}` : `{future_return_col}`."
            )

        group_metrics = _compute_cross_sectional_group_metrics(
            ctx.work,
            future_return_col,
            ctx.cs_group_cols,
            ddof=cs_ddof,
        )
        broadcast = _broadcast_group_metrics(ctx.work, group_metrics, ctx.cs_group_cols)
        cs_std = pd.to_numeric(broadcast["cs_std_raw"], errors="coerce")
        valid_assets = pd.to_numeric(broadcast["cs_valid_assets"], errors="coerce")

        insufficient_assets_mask = valid_assets < min_cs_assets
        too_small_std_mask = cs_std < min_cs_std
        cs_std = cs_std.where(~insufficient_assets_mask)
        cs_std = _apply_small_value_policy(
            cs_std,
            threshold=min_cs_std,
            policy=small_std_policy,
        )

        future_return = pd.to_numeric(ctx.work[future_return_col], errors="coerce")
        cs_scaled = future_return / cs_std

        std_col = _future_cs_std_column_name(suffix)
        scaled_col = _future_cs_scaled_return_column_name(suffix)
        block = pd.DataFrame(
            {
                std_col: cs_std,
                scaled_col: cs_scaled,
            },
            index=ctx.work.index,
        )
        blocks.append(block)
        created_cols.extend([std_col, scaled_col])

        unique_group_metrics = group_metrics.copy()
        cs_group_summary[suffix] = {
            "num_cs_groups": int(len(unique_group_metrics)),
            "num_groups_insufficient_assets": int((unique_group_metrics["cs_valid_assets"] < min_cs_assets).sum()),
            "num_groups_small_std": int((unique_group_metrics["cs_std_raw"] < min_cs_std).fillna(False).sum()),
            "num_groups_nan_std": int(unique_group_metrics["cs_std_raw"].isna().sum()),
        }

    if blocks:
        overwrite_cols = [column for block in blocks for column in block.columns if column in ctx.work.columns]
        if overwrite_cols:
            ctx.work = ctx.work.drop(columns=list(dict.fromkeys(overwrite_cols)))
        ctx.work = pd.concat([ctx.work] + blocks, axis=1)

    result = _restore_original_index(with_returns, ctx.work, created_cols)
    result.attrs = deepcopy(getattr(with_returns, "attrs", {}))
    result.attrs["added_cs_scaled_return_columns"] = created_cols
    result.attrs["cross_sectional_group_summary"] = cs_group_summary
    return result


def _extract_horizon_suffix_from_return_col(return_col: str) -> str:
    prefixes = [
        "future_return_cs_scaled_",
        "future_return_cs_std_",
        "future_return_scaled_",
        "future_log_return_scaled_",
        "future_log_return_",
        "future_return_",
    ]
    for prefix in prefixes:
        if return_col.startswith(prefix):
            return return_col.replace(prefix, "", 1)
    raise ValueError(f"Impossible d'extraire le suffixe d'horizon depuis `{return_col}`.")


def _resolve_target_return_column_name(
    suffix: str,
    *,
    target_scaling: str,
    vol_return_kind: str,
) -> str:
    if target_scaling == "cross_sectional_future_return":
        return _future_cs_scaled_return_column_name(suffix)
    if target_scaling == "past_vol":
        return _future_scaled_return_column_name(suffix, return_kind=vol_return_kind)
    if target_scaling == "raw":
        if vol_return_kind == "log":
            return _future_log_return_column_name(suffix)
        return _future_return_column_name(suffix)
    raise ValueError(
        "target_scaling doit valoir `cross_sectional_future_return`, `past_vol` ou `raw`."
    )


def _compute_binary_target(
    signal_direction: pd.Series,
    future_return: pd.Series,
    *,
    signal_active: Optional[pd.Series],
    tau: float,
    inactive_policy: str,
) -> pd.Series:
    direction = pd.to_numeric(signal_direction, errors="coerce")
    future = pd.to_numeric(future_return, errors="coerce")
    raw_score = direction * future

    target = pd.Series(
        np.where(raw_score > tau, 1.0, 0.0),
        index=direction.index,
        dtype=float,
    )
    target = target.where(direction.notna() & future.notna())

    if signal_active is None:
        return target

    active = pd.to_numeric(signal_active, errors="coerce")
    target = target.where(active.notna())

    if inactive_policy == "nan":
        target = target.where(active != 0, np.nan)
    elif inactive_policy == "zero":
        target = target.where(active != 0, 0.0)
    elif inactive_policy == "drop_later":
        # On conserve la target calculee sur les signaux inactifs.
        # L'utilisateur peut filtrer ensuite via la colonne `signal_active`.
        pass
    else:
        raise ValueError(
            "inactive_policy doit valoir `nan`, `zero` ou `drop_later`."
        )

    return target


def add_binary_signal_target(
    df: pd.DataFrame,
    *,
    signal_direction_col: str,
    signal_active_col: Optional[str],
    return_col: str,
    tau: float = 0.1,
    inactive_policy: str = "nan",
    target_col: Optional[str] = None,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Ajoute une target binaire pour un signal et un horizon deja calcules.
    """
    if signal_direction_col not in df.columns:
        raise ValueError(f"Colonne signal_direction introuvable : `{signal_direction_col}`.")
    if return_col not in df.columns:
        raise ValueError(f"Colonne de return introuvable : `{return_col}`.")

    result = df.copy() if copy else df
    signal_active = None
    if signal_active_col is not None:
        if signal_active_col not in result.columns:
            LOGGER.warning(
                "Colonne signal_active absente pour `%s` : la target sera calculee sans filtre d'activite.",
                signal_direction_col,
            )
        else:
            signal_active = result[signal_active_col]

    if target_col is None:
        if not signal_direction_col.endswith("__signal_direction"):
            raise ValueError(
                "Si `target_col` est absent, `signal_direction_col` doit finir par `__signal_direction`."
            )
        feature = signal_direction_col.replace("__signal_direction", "")
        suffix = _extract_horizon_suffix_from_return_col(return_col)
        target_col = f"{feature}__target_{suffix}"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerformanceWarning)
        result[target_col] = _compute_binary_target(
            result[signal_direction_col],
            result[return_col],
            signal_active=signal_active,
            tau=tau,
            inactive_policy=inactive_policy,
        )
    return result


def _resolve_signal_columns(
    df: pd.DataFrame,
    signal_columns: Optional[Sequence[str] | Mapping[str, Mapping[str, str] | str]],
) -> Dict[str, Dict[str, Optional[str]]]:
    if signal_columns is None:
        resolved: Dict[str, Dict[str, Optional[str]]] = {}
        for column in df.columns:
            if not str(column).endswith("__signal_direction"):
                continue
            feature = str(column).replace("__signal_direction", "")
            active_col = f"{feature}__signal_active"
            resolved[feature] = {
                "direction": str(column),
                "active": active_col if active_col in df.columns else None,
            }
        return resolved

    if isinstance(signal_columns, ABCMapping):
        resolved = {}
        for feature, spec in signal_columns.items():
            if isinstance(spec, ABCMapping):
                direction_col = spec.get("direction")
                active_col = spec.get("active")
            else:
                direction_col = str(spec)
                active_col = None

            if direction_col is None:
                raise ValueError(f"`direction` manquant pour le signal `{feature}`.")
            if direction_col not in df.columns:
                raise ValueError(f"Colonne introuvable pour `{feature}`: `{direction_col}`.")
            if active_col is not None and active_col not in df.columns:
                LOGGER.warning(
                    "Colonne signal_active absente pour `%s` (%s). Le signal sera traite sans filtre d'activite.",
                    feature,
                    active_col,
                )
                active_col = None
            resolved[str(feature)] = {
                "direction": str(direction_col),
                "active": str(active_col) if active_col is not None else None,
            }
        return resolved

    resolved = {}
    for item in signal_columns:
        name = str(item)
        if name.endswith("__signal_direction"):
            feature = name.replace("__signal_direction", "")
            direction_col = name
        else:
            feature = name
            direction_col = f"{feature}__signal_direction"

        if direction_col not in df.columns:
            raise ValueError(f"Colonne introuvable pour `{feature}`: `{direction_col}`.")
        active_col = f"{feature}__signal_active"
        if active_col not in df.columns:
            LOGGER.warning(
                "Colonne signal_active absente pour `%s` (%s). Le signal sera traite sans filtre d'activite.",
                feature,
                active_col,
            )
            active_col = None
        resolved[feature] = {
            "direction": direction_col,
            "active": active_col,
        }
    return resolved


def add_targets_for_multiple_horizons(
    df: pd.DataFrame,
    horizons: Mapping[str, int] | Sequence[int] | int,
    signal_columns: Optional[Sequence[str] | Mapping[str, Mapping[str, str] | str]],
    *,
    price_col: str = "close",
    asset_col: str = "asset",
    datetime_col: str = "datetime",
    use_scaled_return: bool = True,
    target_scaling: str = "cross_sectional_future_return",
    vol_window: int = 20,
    tau: float = 0.1,
    inactive_policy: str = "nan",
    add_log_return: bool = True,
    vol_min_periods: Optional[int] = None,
    vol_return_kind: str = "simple",
    min_cs_assets: int = 3,
    min_cs_std: float = 1e-12,
    small_cs_std_policy: str = "nan",
    config: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """
    Ajoute les returns futurs communs par horizon et les targets binaires par signal.

    Alignement critique :
    - la target reste portee par la ligne a la date t ;
    - le signal est lu a t ;
    - le return futur mesure la performance entre t et t+h ;
    - aucune colonne n'est decalee vers t+h.
    """
    cfg = _resolve_config(config)
    cfg["target"]["tau"] = tau
    cfg["target"]["inactive_policy"] = inactive_policy
    cfg["target"]["use_scaled_return"] = use_scaled_return
    cfg["target"]["target_scaling"] = target_scaling
    cfg["returns"]["add_log_return"] = add_log_return
    cfg["returns"]["vol_window"] = vol_window
    cfg["returns"]["vol_min_periods"] = vol_min_periods
    cfg["returns"]["vol_return_kind"] = vol_return_kind
    cfg["cross_sectional"]["min_cs_assets"] = min_cs_assets
    cfg["cross_sectional"]["min_cs_std"] = min_cs_std
    cfg["cross_sectional"]["small_std_policy"] = small_cs_std_policy

    normalized_horizons = normalize_horizons(horizons)
    resolved_signal_columns = _resolve_signal_columns(df, signal_columns)

    result = compute_future_returns(
        df,
        normalized_horizons,
        price_col=price_col,
        asset_col=asset_col,
        datetime_col=datetime_col,
        add_log_return=add_log_return,
        config=cfg,
    )

    if target_scaling == "cross_sectional_future_return":
        result = compute_cross_sectionally_scaled_future_returns(
            result,
            normalized_horizons,
            price_col=price_col,
            asset_col=asset_col,
            datetime_col=datetime_col,
            min_cs_assets=min_cs_assets,
            min_cs_std=min_cs_std,
            small_std_policy=small_cs_std_policy,
            config=cfg,
        )
    elif target_scaling == "past_vol":
        result = compute_vol_scaled_future_returns(
            result,
            normalized_horizons,
            price_col=price_col,
            asset_col=asset_col,
            datetime_col=datetime_col,
            vol_window=vol_window,
            min_periods=vol_min_periods,
            return_kind=vol_return_kind,
            store_volatility_column=bool(cfg["returns"]["store_volatility_column"]),
            config=cfg,
        )
    elif target_scaling == "raw":
        pass
    else:
        raise ValueError(
            "target_scaling doit valoir `cross_sectional_future_return`, `past_vol` ou `raw`."
        )

    added_target_columns: List[str] = []
    target_column_map: Dict[str, Dict[str, str]] = {}
    target_return_column_by_horizon: Dict[str, str] = {}

    for suffix in normalized_horizons:
        return_col = _resolve_target_return_column_name(
            suffix,
            target_scaling=target_scaling,
            vol_return_kind=vol_return_kind,
        )
        target_return_column_by_horizon[suffix] = return_col

        if return_col not in result.columns:
            raise ValueError(f"Colonne de return cible absente : `{return_col}`.")

        for feature, spec in resolved_signal_columns.items():
            target_col = f"{feature}__target_{suffix}"
            result = add_binary_signal_target(
                result,
                signal_direction_col=spec["direction"],
                signal_active_col=spec["active"],
                return_col=return_col,
                tau=tau,
                inactive_policy=inactive_policy,
                target_col=target_col,
                copy=False,
            )
            added_target_columns.append(target_col)
            target_column_map.setdefault(feature, {})
            target_column_map[feature][suffix] = target_col

    result.attrs = deepcopy(getattr(result, "attrs", {}))
    result.attrs["normalized_horizons"] = normalized_horizons
    result.attrs["resolved_signal_columns"] = resolved_signal_columns
    result.attrs["target_column_map"] = target_column_map
    result.attrs["target_return_column_by_horizon"] = target_return_column_by_horizon
    result.attrs["added_target_columns"] = added_target_columns
    result.attrs["target_scaling"] = target_scaling

    if cfg["general"].get("print_summary", False):
        validate_target_alignment(
            result,
            horizons=normalized_horizons,
            signal_columns=resolved_signal_columns,
            price_col=price_col,
            asset_col=asset_col,
            datetime_col=datetime_col,
            use_scaled_return=use_scaled_return,
            target_scaling=target_scaling,
            vol_window=vol_window,
            vol_min_periods=vol_min_periods,
            vol_return_kind=vol_return_kind,
            min_cs_assets=min_cs_assets,
            min_cs_std=min_cs_std,
            small_cs_std_policy=small_cs_std_policy,
            verbose=True,
        )

    return result


def validate_target_alignment(
    df: pd.DataFrame,
    *,
    horizons: Mapping[str, int] | Sequence[int] | int,
    signal_columns: Optional[Sequence[str] | Mapping[str, Mapping[str, str] | str]],
    price_col: str = "close",
    asset_col: str = "asset",
    datetime_col: str = "datetime",
    use_scaled_return: bool = True,
    target_scaling: str = "cross_sectional_future_return",
    vol_window: int = 20,
    vol_min_periods: Optional[int] = None,
    vol_return_kind: str = "simple",
    min_cs_assets: int = 3,
    min_cs_std: float = 1e-12,
    small_cs_std_policy: str = "nan",
    verbose: bool = True,
    config: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """
    Verifie l'alignement temporel et produit un resume par horizon et signal.
    """
    cfg = _resolve_config(config)
    cfg["target"]["target_scaling"] = target_scaling
    cfg["cross_sectional"]["min_cs_assets"] = min_cs_assets
    cfg["cross_sectional"]["min_cs_std"] = min_cs_std
    cfg["cross_sectional"]["small_std_policy"] = small_cs_std_policy
    normalized_horizons = normalize_horizons(horizons)
    resolved_signal_columns = _resolve_signal_columns(df, signal_columns)

    ctx = _prepare_target_panel(
        df,
        asset_col=asset_col,
        datetime_col=datetime_col,
        config=cfg,
    )
    atol = float(cfg["validation"]["atol"])
    rtol = float(cfg["validation"]["rtol"])

    expected_future = compute_future_returns(
        df,
        normalized_horizons,
        price_col=price_col,
        asset_col=asset_col,
        datetime_col=datetime_col,
        add_log_return=True,
        config=cfg,
    )
    if target_scaling == "cross_sectional_future_return":
        expected_future = compute_cross_sectionally_scaled_future_returns(
            expected_future,
            normalized_horizons,
            price_col=price_col,
            asset_col=asset_col,
            datetime_col=datetime_col,
            min_cs_assets=min_cs_assets,
            min_cs_std=min_cs_std,
            small_std_policy=small_cs_std_policy,
            config=cfg,
        )
    elif target_scaling == "past_vol":
        expected_future = compute_vol_scaled_future_returns(
            expected_future,
            normalized_horizons,
            price_col=price_col,
            asset_col=asset_col,
            datetime_col=datetime_col,
            vol_window=vol_window,
            min_periods=vol_min_periods,
            return_kind=vol_return_kind,
            store_volatility_column=False,
            config=cfg,
        )
    elif target_scaling == "raw":
        pass
    else:
        raise ValueError(
            "target_scaling doit valoir `cross_sectional_future_return`, `past_vol` ou `raw`."
        )

    expected_ctx = _prepare_target_panel(
        expected_future,
        asset_col=asset_col,
        datetime_col=datetime_col,
        config=cfg,
    )

    rows: List[Dict[str, Any]] = []
    duplicate_count = int(
        ctx.work.duplicated(
            subset=ctx.context_cols + [ctx.asset_col, ctx.datetime_col],
            keep=False,
        ).sum()
    )

    for suffix, step in normalized_horizons.items():
        simple_col = _future_return_column_name(suffix)
        target_return_col = _resolve_target_return_column_name(
            suffix,
            target_scaling=target_scaling,
            vol_return_kind=vol_return_kind,
        )

        if target_return_col not in ctx.work.columns:
            LOGGER.warning("Colonne de return absente dans le DataFrame a valider : %s", target_return_col)
            continue

        expected_series = pd.to_numeric(expected_ctx.work[target_return_col], errors="coerce")
        actual_series = pd.to_numeric(ctx.work[target_return_col], errors="coerce")
        aligned_mask = expected_series.notna() | actual_series.notna()
        if aligned_mask.any():
            diff = (actual_series - expected_series).abs()
            max_abs_diff = float(diff[aligned_mask].max())
            alignment_ok = bool(
                np.allclose(
                    actual_series[aligned_mask],
                    expected_series[aligned_mask],
                    equal_nan=True,
                    atol=atol,
                    rtol=rtol,
                )
            )
        else:
            max_abs_diff = np.nan
            alignment_ok = True

        future_nan_count = int(actual_series.isna().sum())

        cs_std_col = _future_cs_std_column_name(suffix)
        cs_scaled_col = _future_cs_scaled_return_column_name(suffix)
        cs_std_present = cs_std_col in ctx.work.columns
        cs_scaled_present = cs_scaled_col in ctx.work.columns
        cs_std_alignment_ok = np.nan
        cs_std_alignment_max_abs_diff = np.nan
        cs_group_count = np.nan
        cs_group_nan_std_count = np.nan
        cs_group_insufficient_assets_count = np.nan
        cs_group_small_std_count = np.nan

        if target_scaling == "cross_sectional_future_return":
            group_metrics = _compute_cross_sectional_group_metrics(
                expected_ctx.work,
                simple_col,
                expected_ctx.cs_group_cols,
                ddof=int(cfg["cross_sectional"]["cs_ddof"]),
            )
            cs_group_count = int(len(group_metrics))
            cs_group_nan_std_count = int(group_metrics["cs_std_raw"].isna().sum())
            cs_group_insufficient_assets_count = int((group_metrics["cs_valid_assets"] < min_cs_assets).sum())
            cs_group_small_std_count = int((group_metrics["cs_std_raw"] < min_cs_std).fillna(False).sum())

            if cs_std_present:
                expected_cs_std = pd.to_numeric(expected_ctx.work[cs_std_col], errors="coerce")
                actual_cs_std = pd.to_numeric(ctx.work[cs_std_col], errors="coerce")
                cs_mask = expected_cs_std.notna() | actual_cs_std.notna()
                if cs_mask.any():
                    cs_diff = (actual_cs_std - expected_cs_std).abs()
                    cs_std_alignment_max_abs_diff = float(cs_diff[cs_mask].max())
                    cs_std_alignment_ok = bool(
                        np.allclose(
                            actual_cs_std[cs_mask],
                            expected_cs_std[cs_mask],
                            equal_nan=True,
                            atol=atol,
                            rtol=rtol,
                        )
                    )
                else:
                    cs_std_alignment_ok = True

        for feature, spec in resolved_signal_columns.items():
            target_col = f"{feature}__target_{suffix}"
            if target_col not in ctx.work.columns:
                continue

            target_series = pd.to_numeric(ctx.work[target_col], errors="coerce")
            active_series = (
                pd.to_numeric(ctx.work[spec["active"]], errors="coerce")
                if spec["active"] is not None and spec["active"] in ctx.work.columns
                else pd.Series(np.nan, index=ctx.work.index)
            )

            valid_mask = actual_series.notna()
            active_rate = float(active_series[valid_mask].mean()) if valid_mask.any() else np.nan
            positive_count = int((target_series == 1).sum())
            zero_count = int((target_series == 0).sum())
            nan_count = int(target_series.isna().sum())

            rows.append(
                {
                    "horizon": suffix,
                    "horizon_steps": step,
                    "feature": feature,
                    "future_return_col": target_return_col,
                    "target_col": target_col,
                    "num_rows": int(len(ctx.work)),
                    "num_duplicate_asset_datetime": duplicate_count,
                    "future_return_nan_count": future_nan_count,
                    "target_nan_count": nan_count,
                    "target_positive_count": positive_count,
                    "target_zero_count": zero_count,
                    "active_rate_on_available_returns": active_rate,
                    "target_scaling": target_scaling,
                    "future_return_cs_std_present": cs_std_present,
                    "future_return_cs_scaled_present": cs_scaled_present,
                    "future_return_cs_std_alignment_ok": cs_std_alignment_ok,
                    "future_return_cs_std_alignment_max_abs_diff": cs_std_alignment_max_abs_diff,
                    "cs_group_count": cs_group_count,
                    "cs_group_nan_std_count": cs_group_nan_std_count,
                    "cs_group_insufficient_assets_count": cs_group_insufficient_assets_count,
                    "cs_group_small_std_count": cs_group_small_std_count,
                    "future_return_alignment_ok": alignment_ok,
                    "future_return_alignment_max_abs_diff": max_abs_diff,
                }
            )

    summary = pd.DataFrame(rows)
    if verbose and not summary.empty:
        print("Validation target alignment")
        grouped = summary.groupby("horizon", dropna=False).agg(
            features=("feature", "count"),
            future_return_nan_count=("future_return_nan_count", "first"),
            target_positive_count=("target_positive_count", "sum"),
            target_zero_count=("target_zero_count", "sum"),
            target_nan_count=("target_nan_count", "sum"),
            mean_active_rate=("active_rate_on_available_returns", "mean"),
            alignment_ok=("future_return_alignment_ok", "all"),
            max_abs_diff=("future_return_alignment_max_abs_diff", "max"),
            cs_std_present=("future_return_cs_std_present", "all"),
            cs_scaled_present=("future_return_cs_scaled_present", "all"),
            cs_std_alignment_ok=("future_return_cs_std_alignment_ok", "all"),
            cs_group_count=("cs_group_count", "first"),
            cs_group_nan_std_count=("cs_group_nan_std_count", "first"),
            cs_group_insufficient_assets_count=("cs_group_insufficient_assets_count", "first"),
            cs_group_small_std_count=("cs_group_small_std_count", "first"),
        )
        print(grouped.to_string())
    return summary


__all__ = [
    "add_binary_signal_target",
    "add_targets_for_multiple_horizons",
    "compute_cross_sectionally_scaled_future_returns",
    "compute_future_returns",
    "compute_vol_scaled_future_returns",
    "get_default_target_config",
    "normalize_horizons",
    "validate_target_alignment",
]

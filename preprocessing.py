from __future__ import annotations

from collections.abc import Mapping as ABCMapping
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LOGGER = logging.getLogger(__name__)


DEFAULT_SIGNAL_CONFIG: Dict[str, Any] = {
    "general": {
        "copy": True,
        "print_summary": False,
        "keep_intermediates": True,
        "context_filter_placeholder": False,
        "unimplemented_band_policy": "skip",
        "small_std_policy": "nan",
        "min_std": 1e-12,
        "std_ddof": 1,
    },
    "columns": {
        "asset_candidates": ["asset", "symbol", "ticker", "crypto", "instrument"],
        "date_candidates": ["date", "datetime", "timestamp", "time"],
        "context_columns": None,
    },
    "thresholds": {
        "centered": 2.0,
        "trend": 2.0,
        "volume": 2.0,
        "bollinger": 2.0,
    },
    "trend_filter": {
        "lag": 1,
    },
    "volume_confirmation": {
        "lag": 1,
    },
    "extreme_zone": {
        "clip_to_bounds": True,
        "bounds_by_column": {},
        "bounds_by_prefix": {
            "rsi_": (0.0, 100.0),
            "rsx_": (0.0, 100.0),
            "stochastic_k_": (0.0, 100.0),
            "stochastic_d_": (0.0, 100.0),
            "mfi_": (0.0, 100.0),
            "cci_": (-200.0, 200.0),
            "pgo_": (-3.0, 3.0),
            "cog_": (-10.0, 10.0),
            "emotion_index_": (0.0, 100.0),
            "willingness_index_": (0.0, 100.0),
        },
    },
    "band_channel_level": {
        "reference_price_column": "close",
    },
}


PRIMARY_SIGNAL_COLUMN_RULES: Dict[str, Any] = {
    "acceleration_bands": lambda cols: [col for col in cols if col.startswith("accbands_percent_b_")],
    "bollinger_bands": lambda cols: [col for col in cols if col.startswith("bb_percent_b_")],
    "donchian_channel": lambda cols: [col for col in cols if col.startswith("donchian_width_")],
    "keltner_channel": lambda cols: [col for col in cols if col.startswith("keltner_width_")],
    "chande_forecast_oscillator": lambda cols: [
        col for col in cols if col.startswith("cfo_") and "fitted" not in col
    ],
    "ease_of_movement": lambda cols: [
        col for col in cols if col.startswith("eom_") and "sma" not in col
    ],
    "elders_force_index": lambda cols: [col for col in cols if col.startswith("efi_ema_")],
    "know_sure_thing": lambda cols: [
        col for col in cols if col.startswith("kst_") and "signal" not in col
    ],
    "macd": lambda cols: [
        col for col in cols if col.startswith("macd_line_") or col.startswith("macd_hist_")
    ],
    "percentage_price_oscillator": lambda cols: [
        col for col in cols if col.startswith("ppo_") and "signal" not in col
    ],
    "percentage_volume_oscillator": lambda cols: [
        col for col in cols if col.startswith("pvo_") and "signal" not in col
    ],
    "relative_vigor_index": lambda cols: [
        col for col in cols if col.startswith("rvi_") and "signal" not in col
    ],
    "ichimoku": lambda cols: [col for col in cols if col.startswith("ichimoku_cloud_delta_")],
    "inertia_indicator": lambda cols: [col for col in cols if col.startswith("inertia_") and "fitted" not in col],
    "linear_regression": lambda cols: [col for col in cols if col.startswith("linreg_slope_")],
    "moving_averages": lambda cols: [
        col for col in cols if col.startswith("sma_spread_") or col.startswith("ema_spread_")
    ],
    "parabolic_sar": lambda cols: [col for col in cols if col.startswith("psar_distance_")],
}


@dataclass
class PanelContext:
    """
    Structure de travail pour un panel cross-sectionnel.
    """

    work: pd.DataFrame
    asset_col: str
    date_col: str
    context_cols: List[str]
    cs_group_cols: List[str]
    ts_group_cols: List[str]
    row_id_col: str = "__row_id"


def get_default_signal_config() -> Dict[str, Any]:
    """
    Retourne une copie profonde de la configuration des signaux.
    """
    return deepcopy(DEFAULT_SIGNAL_CONFIG)


def _deep_update(base: Dict[str, Any], updates: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if updates is None:
        return base

    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def _resolve_config(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return _deep_update(get_default_signal_config(), config)


def _normalize_column_name(column: Any) -> str:
    return str(column).strip().lower()


def _find_candidate_column(
    columns: Sequence[str],
    candidates: Sequence[str],
) -> Optional[str]:
    normalized = {_normalize_column_name(col): col for col in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _resolve_panel_columns(frame: pd.DataFrame, config: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    asset_col = _find_candidate_column(frame.columns, config["columns"]["asset_candidates"])
    date_col = _find_candidate_column(frame.columns, config["columns"]["date_candidates"])

    if date_col is None:
        datetime_columns = [
            col for col in frame.columns if pd.api.types.is_datetime64_any_dtype(frame[col])
        ]
        if len(datetime_columns) == 1:
            date_col = datetime_columns[0]

    if asset_col is None:
        non_datetime_object_columns = [
            col
            for col in frame.columns
            if col != date_col and not pd.api.types.is_datetime64_any_dtype(frame[col])
            and not pd.api.types.is_numeric_dtype(frame[col])
        ]
        if len(non_datetime_object_columns) == 1:
            asset_col = non_datetime_object_columns[0]

    if date_col is None or asset_col is None:
        raise ValueError(
            "Impossible d'identifier les colonnes du panel. "
            "Le DataFrame doit fournir des colonnes/index pour `asset` et `date`."
        )

    context_cols = config["columns"].get("context_columns")
    if context_cols is None:
        potential_context = []
        for col in frame.columns:
            if col in {asset_col, date_col}:
                continue
            if col.startswith("__"):
                continue
            if pd.api.types.is_numeric_dtype(frame[col]) or pd.api.types.is_datetime64_any_dtype(frame[col]):
                continue
            potential_context.append(col)
        context_cols = potential_context

    return asset_col, date_col, list(context_cols)


def _prepare_panel_context(df: pd.DataFrame, config: Dict[str, Any]) -> PanelContext:
    work = df.copy().reset_index()
    work["__row_id"] = np.arange(len(work), dtype=int)

    asset_col, date_col, context_cols = _resolve_panel_columns(work, config)
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    if work[date_col].isna().all():
        raise ValueError("La colonne de date du panel ne peut pas etre convertie en datetime.")

    cs_group_cols = context_cols + [date_col]
    ts_group_cols = context_cols + [asset_col]

    sort_cols = context_cols + [asset_col, date_col]
    work = work.sort_values(sort_cols + ["__row_id"]).reset_index(drop=True)
    return PanelContext(
        work=work,
        asset_col=asset_col,
        date_col=date_col,
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
        added = aligned[list(new_columns)].reset_index(drop=True)
        added.index = restored.index
        restored = pd.concat([restored, added], axis=1)
    return restored


def _groupby_series(series: pd.Series, frame: pd.DataFrame, group_cols: Sequence[str]):
    return series.groupby([frame[col] for col in group_cols], sort=False)


def compute_cross_sectional_leave_one_out_zscore(
    frame: pd.DataFrame,
    column: str,
    group_cols: Sequence[str],
    *,
    ddof: int = 1,
    min_std: float = 1e-12,
    small_std_policy: str = "nan",
) -> pd.DataFrame:
    """
    Calcule moyenne, ecart-type et z-score cross-sectionnels leave-one-out.

    Les calculs sont effectues par groupe cross-sectionnel, typiquement
    `date` ou (`timeframe`, `date`), et pour une seule colonne d'indicateur.
    """
    values = pd.to_numeric(frame[column], errors="coerce")
    valid = values.notna().astype(int)
    group_sum = _groupby_series(values, frame, group_cols).transform("sum")
    group_count = _groupby_series(values, frame, group_cols).transform("count")
    values_filled = values.where(values.notna(), 0.0)

    loo_count = group_count - valid
    loo_sum = group_sum - values_filled
    loo_mean = loo_sum / loo_count.where(loo_count > 0)

    squared = values_filled ** 2
    group_sumsq = _groupby_series(squared, frame, group_cols).transform("sum")
    loo_sumsq = group_sumsq - squared

    loo_var_num = loo_sumsq - (loo_sum ** 2) / loo_count.where(loo_count > 0)
    loo_denom = loo_count - ddof
    loo_var = loo_var_num / loo_denom.where(loo_denom > 0)
    loo_var = loo_var.clip(lower=0.0)
    loo_std = np.sqrt(loo_var)

    if small_std_policy == "epsilon":
        loo_std = loo_std.where(loo_std.isna(), loo_std.clip(lower=min_std))
    else:
        loo_std = loo_std.where(loo_std >= min_std)

    loo_zscore = (values - loo_mean) / loo_std
    return pd.DataFrame(
        {
            "cs_mean_loo": loo_mean,
            "cs_std_loo": loo_std,
            "cs_zscore_loo": loo_zscore,
        },
        index=frame.index,
    )


def _compute_grouped_delta(
    frame: pd.DataFrame,
    column: str,
    group_cols: Sequence[str],
    date_col: str,
    lag: int,
) -> pd.Series:
    sorted_frame = frame.sort_values(list(group_cols) + [date_col, "__row_id"])
    sorted_delta = (
        sorted_frame.groupby(list(group_cols), sort=False)[column]
        .diff(lag)
    )
    delta = pd.Series(np.nan, index=frame.index, dtype=float)
    delta.loc[sorted_frame.index] = pd.to_numeric(sorted_delta, errors="coerce")
    return delta


def _safe_abs(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").abs()


def _sign(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.sign(values), index=values.index, dtype=float)


def _signal_columns(feature: str) -> Dict[str, str]:
    return {
        "direction": f"{feature}__signal_direction",
        "active": f"{feature}__signal_active",
        "strength": f"{feature}__signal_strength",
        "signed_value": f"{feature}__signal_signed_value",
        "cs_mean_loo": f"{feature}__cs_mean_loo",
        "cs_std_loo": f"{feature}__cs_std_loo",
        "cs_zscore_loo": f"{feature}__cs_zscore_loo",
        "delta": f"{feature}__delta",
        "trend_score": f"{feature}__trend_score",
        "mid": f"{feature}__mid",
        "signed_force": f"{feature}__signed_force",
        "distance_to_nearest_band": f"{feature}__distance_to_nearest_band",
        "band_score": f"{feature}__band_score",
    }


def _build_base_signal_block(
    index: pd.Index,
    feature: str,
    *,
    direction: pd.Series,
    active: pd.Series,
    strength: pd.Series,
    signed_value: pd.Series,
) -> pd.DataFrame:
    names = _signal_columns(feature)
    return pd.DataFrame(
        {
            names["direction"]: direction,
            names["active"]: active.astype(float),
            names["strength"]: strength,
            names["signed_value"]: signed_value,
        },
        index=index,
    )


def _build_cs_intermediate_block(
    index: pd.Index,
    feature: str,
    stats: pd.DataFrame,
    keep_intermediates: bool,
) -> pd.DataFrame:
    if not keep_intermediates:
        return pd.DataFrame(index=index)

    names = _signal_columns(feature)
    return pd.DataFrame(
        {
            names["cs_mean_loo"]: stats["cs_mean_loo"],
            names["cs_std_loo"]: stats["cs_std_loo"],
            names["cs_zscore_loo"]: stats["cs_zscore_loo"],
        },
        index=index,
    )


def _build_nan_signal_block(index: pd.Index, feature: str, keep_intermediates: bool) -> pd.DataFrame:
    names = _signal_columns(feature)
    columns = {
        names["direction"]: pd.Series(np.nan, index=index),
        names["active"]: pd.Series(np.nan, index=index),
        names["strength"]: pd.Series(np.nan, index=index),
        names["signed_value"]: pd.Series(np.nan, index=index),
    }
    if keep_intermediates:
        columns[names["cs_mean_loo"]] = pd.Series(np.nan, index=index)
        columns[names["cs_std_loo"]] = pd.Series(np.nan, index=index)
        columns[names["cs_zscore_loo"]] = pd.Series(np.nan, index=index)
    return pd.DataFrame(columns, index=index)


def _resolve_feature_columns(
    df: pd.DataFrame,
    indicator_family_map: Mapping[str, str],
    family: str,
) -> List[str]:
    return [
        column for column, mapped_family in indicator_family_map.items()
        if mapped_family == family and column in df.columns
    ]


def _select_default_columns_for_indicator(indicator_key: str, columns: Sequence[str]) -> List[str]:
    if indicator_key in PRIMARY_SIGNAL_COLUMN_RULES:
        selected = PRIMARY_SIGNAL_COLUMN_RULES[indicator_key](list(columns))
        if selected:
            return selected
    return list(columns)


def build_default_indicator_family_map(
    df: pd.DataFrame,
    *,
    primary_only: bool = True,
    include_context_filters: bool = True,
    include_unimplemented_band_columns: bool = False,
) -> Dict[str, str]:
    """
    Construit automatiquement un mapping `feature_col -> famille`.

    Si le DataFrame provient de `features.compute_all_indicators`, la fonction
    exploite directement les `attrs` stockes dans le DataFrame.
    """
    indicator_column_map = df.attrs.get("indicator_column_map")
    indicator_registry = df.attrs.get("indicator_registry")
    if not indicator_column_map or not indicator_registry:
        raise ValueError(
            "Impossible de construire automatiquement le mapping des familles. "
            "Passe `indicator_family_map` explicitement ou utilise un DataFrame "
            "issu de `compute_all_indicators`."
        )

    family_map: Dict[str, str] = {}
    for indicator_key, columns in indicator_column_map.items():
        registry_entry = indicator_registry.get(indicator_key, {})
        family = registry_entry.get("family")
        if family is None:
            continue
        if family == "context_filter" and not include_context_filters:
            continue
        if family == "band_channel_level" and not include_unimplemented_band_columns:
            if indicator_key != "bollinger_bands":
                continue

        selected_columns = list(columns)
        if primary_only:
            selected_columns = _select_default_columns_for_indicator(indicator_key, columns)

        for column in selected_columns:
            if column in df.columns:
                family_map[column] = family

    return family_map


def get_added_signal_columns(df: pd.DataFrame) -> List[str]:
    """
    Retourne la liste des colonnes de signal ajoutees.
    """
    return list(df.attrs.get("added_signal_columns", []))


def _resolve_extreme_bounds(feature: str, config: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    bounds_by_column = config["extreme_zone"].get("bounds_by_column", {})
    if feature in bounds_by_column:
        lower, upper = bounds_by_column[feature]
        return float(lower), float(upper)

    for prefix, bounds in config["extreme_zone"].get("bounds_by_prefix", {}).items():
        if feature.startswith(prefix):
            lower, upper = bounds
            return float(lower), float(upper)

    return None


def add_centered_oscillator_signals(
    ctx: PanelContext,
    feature_columns: Sequence[str],
    config: Dict[str, Any],
) -> Dict[str, List[str]]:
    created: Dict[str, List[str]] = {}
    threshold = float(config["thresholds"]["centered"])
    keep_intermediates = bool(config["general"]["keep_intermediates"])
    blocks: List[pd.DataFrame] = []

    for feature in feature_columns:
        stats = compute_cross_sectional_leave_one_out_zscore(
            ctx.work,
            feature,
            ctx.cs_group_cols,
            ddof=int(config["general"]["std_ddof"]),
            min_std=float(config["general"]["min_std"]),
            small_std_policy=str(config["general"]["small_std_policy"]),
        )
        zscore = stats["cs_zscore_loo"]
        direction = _sign(ctx.work[feature])
        active = (zscore.abs() > threshold).fillna(False).astype(float)
        strength = zscore.abs()
        signed_value = zscore

        feature_block = _build_base_signal_block(
            ctx.work.index,
            feature,
            direction=direction,
            active=active,
            strength=strength,
            signed_value=signed_value,
        )
        feature_block = pd.concat(
            [feature_block, _build_cs_intermediate_block(ctx.work.index, feature, stats, keep_intermediates)],
            axis=1,
        )
        blocks.append(feature_block)
        created[feature] = list(feature_block.columns)

    if blocks:
        ctx.work = pd.concat([ctx.work] + blocks, axis=1)

    return created


def add_extreme_zone_oscillator_signals(
    ctx: PanelContext,
    feature_columns: Sequence[str],
    config: Dict[str, Any],
) -> Dict[str, List[str]]:
    created: Dict[str, List[str]] = {}
    keep_intermediates = bool(config["general"]["keep_intermediates"])
    clip_to_bounds = bool(config["extreme_zone"]["clip_to_bounds"])
    blocks: List[pd.DataFrame] = []

    for feature in feature_columns:
        bounds = _resolve_extreme_bounds(feature, config)
        if bounds is None:
            LOGGER.warning(
                "Bornes introuvables pour `%s` dans la famille extreme_zone_oscillator : colonne ignoree.",
                feature,
            )
            continue

        lower_bound, upper_bound = bounds
        if lower_bound >= upper_bound:
            raise ValueError(
                f"Bornes invalides pour `{feature}` : lower_bound >= upper_bound."
            )

        raw = pd.to_numeric(ctx.work[feature], errors="coerce")
        clipped = raw.clip(lower=lower_bound, upper=upper_bound) if clip_to_bounds else raw
        mid = (lower_bound + upper_bound) / 2.0
        direction = pd.Series(
            np.where(clipped < mid, 1.0, -1.0),
            index=ctx.work.index,
            dtype=float,
        ).where(raw.notna())
        in_interval = raw.between(lower_bound, upper_bound, inclusive="both")
        active = (raw.notna() & in_interval).astype(float)
        strength = (clipped - mid).abs()

        # Convention retenue :
        # on transforme la distance au milieu en force signee causale,
        # positive dans la moitie basse et negative dans la moitie haute.
        signed_force = mid - clipped
        stats = compute_cross_sectional_leave_one_out_zscore(
            pd.concat([ctx.work, signed_force.rename("__signed_force_temp")], axis=1),
            "__signed_force_temp",
            ctx.cs_group_cols,
            ddof=int(config["general"]["std_ddof"]),
            min_std=float(config["general"]["min_std"]),
            small_std_policy=str(config["general"]["small_std_policy"]),
        )

        feature_block = _build_base_signal_block(
            ctx.work.index,
            feature,
            direction=direction,
            active=active,
            strength=strength,
            signed_value=stats["cs_zscore_loo"],
        )

        if keep_intermediates:
            names = _signal_columns(feature)
            extra_block = pd.DataFrame(
                {
                    names["mid"]: pd.Series(mid, index=ctx.work.index),
                    names["signed_force"]: signed_force,
                },
                index=ctx.work.index,
            )
            feature_block = pd.concat(
                [
                    feature_block,
                    extra_block,
                    _build_cs_intermediate_block(ctx.work.index, feature, stats, True),
                ],
                axis=1,
            )

        blocks.append(feature_block)
        created[feature] = list(feature_block.columns)

    if blocks:
        ctx.work = pd.concat([ctx.work] + blocks, axis=1)

    return created


def add_trend_filter_signals(
    ctx: PanelContext,
    feature_columns: Sequence[str],
    config: Dict[str, Any],
) -> Dict[str, List[str]]:
    created: Dict[str, List[str]] = {}
    threshold = float(config["thresholds"]["trend"])
    lag = int(config["trend_filter"]["lag"])
    keep_intermediates = bool(config["general"]["keep_intermediates"])
    blocks: List[pd.DataFrame] = []

    for feature in feature_columns:
        delta = _compute_grouped_delta(
            ctx.work,
            feature,
            ctx.ts_group_cols,
            ctx.date_col,
            lag=lag,
        )
        frame_with_delta = pd.concat([ctx.work, delta.rename("__delta_temp")], axis=1)
        stats = compute_cross_sectional_leave_one_out_zscore(
            frame_with_delta,
            "__delta_temp",
            ctx.cs_group_cols,
            ddof=int(config["general"]["std_ddof"]),
            min_std=float(config["general"]["min_std"]),
            small_std_policy=str(config["general"]["small_std_policy"]),
        )
        trend_score = stats["cs_zscore_loo"]

        direction = _sign(trend_score)
        active = (trend_score.abs() > threshold).fillna(False).astype(float)
        strength = trend_score.abs()
        signed_value = trend_score

        feature_block = _build_base_signal_block(
            ctx.work.index,
            feature,
            direction=direction,
            active=active,
            strength=strength,
            signed_value=signed_value,
        )

        if keep_intermediates:
            names = _signal_columns(feature)
            extra_block = pd.DataFrame(
                {
                    names["delta"]: delta,
                    names["trend_score"]: trend_score,
                },
                index=ctx.work.index,
            )
            feature_block = pd.concat(
                [
                    feature_block,
                    extra_block,
                    _build_cs_intermediate_block(ctx.work.index, feature, stats, True),
                ],
                axis=1,
            )

        blocks.append(feature_block)
        created[feature] = list(feature_block.columns)

    if blocks:
        ctx.work = pd.concat([ctx.work] + blocks, axis=1)

    return created


def add_volume_confirmation_signals(
    ctx: PanelContext,
    feature_columns: Sequence[str],
    config: Dict[str, Any],
) -> Dict[str, List[str]]:
    created: Dict[str, List[str]] = {}
    threshold = float(config["thresholds"]["volume"])
    lag = int(config["volume_confirmation"]["lag"])
    keep_intermediates = bool(config["general"]["keep_intermediates"])
    blocks: List[pd.DataFrame] = []

    for feature in feature_columns:
        delta = _compute_grouped_delta(
            ctx.work,
            feature,
            ctx.ts_group_cols,
            ctx.date_col,
            lag=lag,
        )
        frame_with_delta = pd.concat([ctx.work, delta.rename("__delta_temp")], axis=1)
        stats = compute_cross_sectional_leave_one_out_zscore(
            frame_with_delta,
            "__delta_temp",
            ctx.cs_group_cols,
            ddof=int(config["general"]["std_ddof"]),
            min_std=float(config["general"]["min_std"]),
            small_std_policy=str(config["general"]["small_std_policy"]),
        )
        variance_score = stats["cs_zscore_loo"]
        direction = _sign(delta)
        active = (variance_score.abs() > threshold).fillna(False).astype(float)
        strength = variance_score.abs()
        signed_value = variance_score

        feature_block = _build_base_signal_block(
            ctx.work.index,
            feature,
            direction=direction,
            active=active,
            strength=strength,
            signed_value=signed_value,
        )

        if keep_intermediates:
            names = _signal_columns(feature)
            extra_block = pd.DataFrame(
                {
                    names["delta"]: delta,
                },
                index=ctx.work.index,
            )
            feature_block = pd.concat(
                [
                    feature_block,
                    extra_block,
                    _build_cs_intermediate_block(ctx.work.index, feature, stats, True),
                ],
                axis=1,
            )

        blocks.append(feature_block)
        created[feature] = list(feature_block.columns)

    if blocks:
        ctx.work = pd.concat([ctx.work] + blocks, axis=1)

    return created


def add_bollinger_band_signals(
    ctx: PanelContext,
    feature_columns: Sequence[str],
    config: Dict[str, Any],
) -> Dict[str, List[str]]:
    created: Dict[str, List[str]] = {}
    keep_intermediates = bool(config["general"]["keep_intermediates"])
    threshold = float(config["thresholds"]["bollinger"])
    reference_price_col = str(config["band_channel_level"]["reference_price_column"])
    blocks: List[pd.DataFrame] = []

    if reference_price_col not in ctx.work.columns:
        raise ValueError(
            f"La colonne de reference `{reference_price_col}` est indispensable pour les signaux Bollinger."
        )

    for feature in feature_columns:
        if not feature.startswith("bb_percent_b_"):
            policy = str(config["general"]["unimplemented_band_policy"])
            if policy == "nan":
                feature_block = _build_nan_signal_block(ctx.work.index, feature, keep_intermediates)
                blocks.append(feature_block)
                created[feature] = list(feature_block.columns)
            else:
                LOGGER.warning(
                    "La logique band_channel_level n'est implemente que pour Bollinger (`bb_percent_b_*`). "
                    "Colonne ignoree : %s",
                    feature,
                )
            continue

        suffix = feature.replace("bb_percent_b_", "", 1)
        upper_col = f"bb_upper_{suffix}"
        lower_col = f"bb_lower_{suffix}"

        missing_cols = [col for col in (upper_col, lower_col) if col not in ctx.work.columns]
        if missing_cols:
            raise ValueError(
                f"Colonnes Bollinger manquantes pour `{feature}` : {', '.join(missing_cols)}"
            )

        price = pd.to_numeric(ctx.work[reference_price_col], errors="coerce")
        upper = pd.to_numeric(ctx.work[upper_col], errors="coerce")
        lower = pd.to_numeric(ctx.work[lower_col], errors="coerce")

        above_upper = price > upper
        below_lower = price < lower
        inside_band = ~(above_upper | below_lower)

        direction = pd.Series(0.0, index=ctx.work.index, dtype=float)
        direction = direction.where(~above_upper, 1.0)
        direction = direction.where(~below_lower, -1.0)
        direction = direction.where(price.notna() & upper.notna() & lower.notna())

        distance = pd.Series(0.0, index=ctx.work.index, dtype=float)
        distance = distance.where(~above_upper, price - upper)
        distance = distance.where(~below_lower, lower - price)
        distance = distance.where(price.notna() & upper.notna() & lower.notna())

        stats = compute_cross_sectional_leave_one_out_zscore(
            pd.concat([ctx.work, distance.rename("__bollinger_distance_temp")], axis=1),
            "__bollinger_distance_temp",
            ctx.cs_group_cols,
            ddof=int(config["general"]["std_ddof"]),
            min_std=float(config["general"]["min_std"]),
            small_std_policy=str(config["general"]["small_std_policy"]),
        )

        band_score = distance / stats["cs_std_loo"]
        signal_strength = band_score.abs()
        signal_signed_value = direction * band_score.abs()
        signal_active = (
            direction.ne(0) & (band_score > threshold)
        ).fillna(False).astype(float)

        feature_block = _build_base_signal_block(
            ctx.work.index,
            feature,
            direction=direction,
            active=signal_active,
            strength=signal_strength,
            signed_value=signal_signed_value,
        )

        if keep_intermediates:
            names = _signal_columns(feature)
            extra_block = pd.DataFrame(
                {
                    names["distance_to_nearest_band"]: distance,
                    names["band_score"]: band_score,
                },
                index=ctx.work.index,
            )
            feature_block = pd.concat(
                [
                    feature_block,
                    extra_block,
                    _build_cs_intermediate_block(ctx.work.index, feature, stats, True),
                ],
                axis=1,
            )

        blocks.append(feature_block)
        created[feature] = list(feature_block.columns)

    if blocks:
        ctx.work = pd.concat([ctx.work] + blocks, axis=1)

    return created


def add_context_filter_signals(
    ctx: PanelContext,
    feature_columns: Sequence[str],
    config: Dict[str, Any],
) -> Dict[str, List[str]]:
    """
    Famille volontairement exclue de la construction actuelle des signaux.
    """
    created: Dict[str, List[str]] = {}
    if not feature_columns:
        return created

    if not config["general"]["context_filter_placeholder"]:
        LOGGER.info(
            "Les context_filters sont exclus de la couche de signaux actuelle (%d colonnes ignorees).",
            len(feature_columns),
        )
        return created

    keep_intermediates = bool(config["general"]["keep_intermediates"])
    blocks: List[pd.DataFrame] = []
    for feature in feature_columns:
        feature_block = _build_nan_signal_block(ctx.work.index, feature, keep_intermediates)
        blocks.append(feature_block)
        created[feature] = list(feature_block.columns)

    if blocks:
        ctx.work = pd.concat([ctx.work] + blocks, axis=1)

    return created


def summarize_signal_features(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Retourne un resume compact des signaux ajoutes.
    """
    created_map = df.attrs.get("signal_column_map", {})
    family_summary = df.attrs.get("signal_family_summary", {})
    return {
        "num_signal_base_features": len(created_map),
        "num_added_signal_columns": len(df.attrs.get("added_signal_columns", [])),
        "family_summary": family_summary,
    }


def print_signal_summary(df: pd.DataFrame) -> None:
    """
    Affiche un resume lisible des colonnes de signal ajoutees.
    """
    summary = summarize_signal_features(df)
    print(
        f"Signaux generes : {summary['num_signal_base_features']} features brutes, "
        f"{summary['num_added_signal_columns']} colonnes ajoutees."
    )
    for family, count in summary["family_summary"].items():
        print(f"- {family}: {count} features brutes traitees")


def add_all_signal_features(
    df: pd.DataFrame,
    indicator_family_map: Optional[Mapping[str, str]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """
    Enrichit un DataFrame panel d'indicateurs avec des colonnes de signaux.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame panel contenant au minimum une dimension `asset` et `date`,
        ainsi que les colonnes d'indicateurs deja calculees.
    indicator_family_map : Optional[Mapping[str, str]]
        Mapping `{feature_col: signal_family}`. Si `None`, la fonction tente
        de le construire depuis les `attrs` produits par `compute_all_indicators`.
    config : Optional[Mapping[str, Any]]
        Surcharge optionnelle de la configuration par defaut.

    Returns
    -------
    pd.DataFrame
        Meme DataFrame enrichi avec les colonnes :
        `<feature>__signal_direction`,
        `<feature>__signal_active`,
        `<feature>__signal_strength`,
        `<feature>__signal_signed_value`,
        ainsi que certaines colonnes intermediaires si activees.
    """
    if indicator_family_map is None:
        indicator_family_map = build_default_indicator_family_map(df)

    cfg = _resolve_config(config)
    original = df.copy() if cfg["general"]["copy"] else df
    original_attrs = deepcopy(getattr(df, "attrs", {}))

    ctx = _prepare_panel_context(original, cfg)
    resolved_family_map = {
        column: family
        for column, family in indicator_family_map.items()
        if column in original.columns
    }

    missing_columns = [column for column in indicator_family_map if column not in original.columns]
    for column in missing_columns:
        LOGGER.warning("Colonne absente du DataFrame ignoree dans indicator_family_map : %s", column)

    created_map: Dict[str, List[str]] = {}
    family_summary: Dict[str, int] = {}

    centered_features = _resolve_feature_columns(original, resolved_family_map, "centered_oscillator")
    centered_created = add_centered_oscillator_signals(ctx, centered_features, cfg)
    created_map.update(centered_created)
    family_summary["centered_oscillator"] = len(centered_created)

    extreme_features = _resolve_feature_columns(original, resolved_family_map, "extreme_zone_oscillator")
    extreme_created = add_extreme_zone_oscillator_signals(ctx, extreme_features, cfg)
    created_map.update(extreme_created)
    family_summary["extreme_zone_oscillator"] = len(extreme_created)

    trend_features = _resolve_feature_columns(original, resolved_family_map, "trend_filter")
    trend_created = add_trend_filter_signals(ctx, trend_features, cfg)
    created_map.update(trend_created)
    family_summary["trend_filter"] = len(trend_created)

    volume_features = _resolve_feature_columns(original, resolved_family_map, "volume_confirmation")
    volume_created = add_volume_confirmation_signals(ctx, volume_features, cfg)
    created_map.update(volume_created)
    family_summary["volume_confirmation"] = len(volume_created)

    band_features = _resolve_feature_columns(original, resolved_family_map, "band_channel_level")
    band_created = add_bollinger_band_signals(ctx, band_features, cfg)
    created_map.update(band_created)
    family_summary["band_channel_level"] = len(band_created)

    context_features = _resolve_feature_columns(original, resolved_family_map, "context_filter")
    context_created = add_context_filter_signals(ctx, context_features, cfg)
    created_map.update(context_created)
    family_summary["context_filter"] = len(context_created)

    added_signal_columns = [
        column
        for columns in created_map.values()
        for column in columns
    ]
    added_signal_columns = list(dict.fromkeys(added_signal_columns))

    result = _restore_original_index(original, ctx.work, added_signal_columns)
    result.attrs = original_attrs
    result.attrs["signal_column_map"] = created_map
    result.attrs["signal_family_summary"] = family_summary
    result.attrs["indicator_family_map"] = dict(resolved_family_map)
    result.attrs["added_signal_columns"] = added_signal_columns
    result.attrs["signal_config"] = cfg

    if cfg["general"].get("print_summary", False):
        print_signal_summary(result)

    return result


def _prepare_panel_context_with_preferences(
    df: pd.DataFrame,
    *,
    asset_col: str,
    datetime_col: str,
    config: Dict[str, Any],
) -> PanelContext:
    """
    Variante de `_prepare_panel_context` avec colonnes preferees explicites.
    """
    preferred_config = deepcopy(config)
    preferred_config["columns"]["asset_candidates"] = [
        asset_col,
        *[
            candidate
            for candidate in config["columns"]["asset_candidates"]
            if _normalize_column_name(candidate) != _normalize_column_name(asset_col)
        ],
    ]
    preferred_config["columns"]["date_candidates"] = [
        datetime_col,
        *[
            candidate
            for candidate in config["columns"]["date_candidates"]
            if _normalize_column_name(candidate) != _normalize_column_name(datetime_col)
        ],
    ]
    return _prepare_panel_context(df, preferred_config)


def _coerce_indicator_family_map(
    df: pd.DataFrame,
    indicator_family_map: Optional[Mapping[str, str] | pd.DataFrame],
) -> Dict[str, str]:
    """
    Resout un mapping `feature -> signal_family` a partir des `attrs` ou d'un objet externe.

    Priorite :
    1. mapping passe explicitement par l'utilisateur ;
    2. mapping stocke dans `df.attrs["indicator_family_map"]` ;
    3. reconstruction via `build_default_indicator_family_map`.
    """
    resolved: Dict[str, str] = {}

    attrs_map = df.attrs.get("indicator_family_map")
    if isinstance(attrs_map, dict):
        resolved.update({str(key): str(value) for key, value in attrs_map.items()})

    if indicator_family_map is None:
        if resolved:
            return resolved
        return build_default_indicator_family_map(df)

    if isinstance(indicator_family_map, pd.DataFrame):
        mapping_df = indicator_family_map.copy()
        feature_candidates = [
            "feature_name",
            "feature",
            "feature_col",
            "column",
            "indicator_name",
        ]
        family_candidates = ["signal_family", "family"]
        feature_col = next((col for col in feature_candidates if col in mapping_df.columns), None)
        family_col = next((col for col in family_candidates if col in mapping_df.columns), None)
        if feature_col is None or family_col is None:
            raise ValueError(
                "Le DataFrame `indicator_family_map` doit contenir au moins "
                "`feature_name`/`indicator_name` et `signal_family`/`family`."
            )

        explicit_map = (
            mapping_df[[feature_col, family_col]]
            .dropna(subset=[feature_col, family_col])
            .drop_duplicates(subset=[feature_col], keep="last")
        )
        explicit_dict = {
            str(feature): str(family)
            for feature, family in explicit_map.itertuples(index=False, name=None)
        }

        matched_columns = sum(1 for feature in explicit_dict if feature in df.columns)
        if matched_columns == 0 and resolved:
            LOGGER.warning(
                "Le mapping de familles fourni ne matche aucune colonne du panel large ; "
                "les `attrs` du DataFrame sont utilises a la place."
            )
            return resolved

        resolved.update(explicit_dict)
        return resolved

    if isinstance(indicator_family_map, ABCMapping):
        resolved.update({str(key): str(value) for key, value in indicator_family_map.items()})
        return resolved

    raise TypeError(
        "`indicator_family_map` doit etre un dict, un DataFrame pandas ou `None`."
    )


def identify_signal_feature_columns(
    df: pd.DataFrame,
) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Detecte les features brutes qui possedent des colonnes de signal exploitables.

    Une feature est retenue si elle expose au minimum :
    - `__signal_direction`
    - `__signal_strength`
    - `__signal_signed_value`
    La colonne `__signal_active` est facultative ; si elle manque, elle sera
    simplement remplacee par NaN dans le dataset final.
    """
    detected: Dict[str, Dict[str, Optional[str]]] = {}

    for column in df.columns:
        name = str(column)
        if not name.endswith("__signal_direction"):
            continue

        feature = name.replace("__signal_direction", "")
        active_col = f"{feature}__signal_active"
        strength_col = f"{feature}__signal_strength"
        signed_value_col = f"{feature}__signal_signed_value"

        missing_required = [
            required
            for required in [strength_col, signed_value_col]
            if required not in df.columns
        ]
        if missing_required:
            LOGGER.warning(
                "Feature de signal ignoree `%s` : colonnes manquantes %s.",
                feature,
                ", ".join(missing_required),
            )
            continue

        detected[feature] = {
            "direction": name,
            "active": active_col if active_col in df.columns else None,
            "strength": strength_col,
            "signed_value": signed_value_col,
        }

    return detected


def identify_target_columns(
    df: pd.DataFrame,
    *,
    horizons: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Detecte les colonnes de target au format `<feature>__target_<horizon>`.
    """
    allowed_horizons = set(horizons) if horizons is not None else None
    detected: Dict[str, Dict[str, str]] = {}

    for column in df.columns:
        name = str(column)
        if "__target_" not in name:
            continue

        feature, horizon = name.rsplit("__target_", 1)
        if not feature or not horizon:
            continue
        if allowed_horizons is not None and horizon not in allowed_horizons:
            continue

        detected.setdefault(feature, {})
        detected[feature][horizon] = name

    return detected


def get_context_feature_columns(
    df: pd.DataFrame,
    indicator_family_map: Mapping[str, str],
    *,
    context_family_name: str = "context_filter",
    exclude_vix_context: bool = True,
) -> List[str]:
    """
    Selectionne les features de contexte a recopier sur chaque ligne du dataset long.
    """
    context_columns = [
        column
        for column, family in indicator_family_map.items()
        if family == context_family_name and column in df.columns
    ]

    filtered_columns: List[str] = []
    for column in context_columns:
        if exclude_vix_context and "vix" in str(column).lower():
            continue
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        filtered_columns.append(str(column))

    return list(dict.fromkeys(filtered_columns))


def _resolve_output_horizons(
    detected_target_columns: Mapping[str, Mapping[str, str]],
    requested_horizons: Optional[Sequence[str]],
) -> List[str]:
    """
    Determine l'ordre final des horizons a empiler.
    """
    detected_order: List[str] = []
    for horizon_map in detected_target_columns.values():
        for horizon in horizon_map:
            if horizon not in detected_order:
                detected_order.append(horizon)

    if requested_horizons is None:
        return detected_order

    missing = [horizon for horizon in requested_horizons if horizon not in detected_order]
    if missing:
        raise ValueError(
            "Horizons demandes introuvables dans le panel large : "
            f"{', '.join(missing)}"
        )
    return list(requested_horizons)


def _build_signal_long_frame(
    panel: pd.DataFrame,
    *,
    key_cols: Sequence[str],
    signal_feature_map: Mapping[str, Mapping[str, Optional[str]]],
    family_map: Mapping[str, str],
) -> pd.DataFrame:
    """
    Empile les colonnes de signal en format long par `indicator_name`.
    """
    blocks: List[pd.DataFrame] = []

    for feature, spec in signal_feature_map.items():
        block = panel[list(key_cols)].copy()
        block["indicator_name"] = feature
        block["signal_family"] = family_map.get(feature, np.nan)
        block["signal_direction"] = pd.to_numeric(panel[spec["direction"]], errors="coerce")
        if spec["active"] is None:
            block["signal_active"] = np.nan
        else:
            block["signal_active"] = pd.to_numeric(panel[spec["active"]], errors="coerce")
        block["signal_strength"] = pd.to_numeric(panel[spec["strength"]], errors="coerce")
        block["signal_signed_value"] = pd.to_numeric(panel[spec["signed_value"]], errors="coerce")
        blocks.append(block)

    if not blocks:
        return pd.DataFrame(
            columns=[
                *key_cols,
                "indicator_name",
                "signal_family",
                "signal_direction",
                "signal_active",
                "signal_strength",
                "signal_signed_value",
            ]
        )

    return pd.concat(blocks, ignore_index=True)


def _build_target_long_frame(
    panel: pd.DataFrame,
    *,
    key_cols: Sequence[str],
    target_columns: Mapping[str, Mapping[str, str]],
    horizons: Sequence[str],
) -> pd.DataFrame:
    """
    Empile les colonnes de target en format long par `(indicator_name, horizon)`.
    """
    blocks: List[pd.DataFrame] = []

    for feature, horizon_map in target_columns.items():
        for horizon in horizons:
            target_col = horizon_map.get(horizon)
            if target_col is None:
                continue

            block = panel[list(key_cols)].copy()
            block["indicator_name"] = feature
            block["horizon"] = horizon
            block["target"] = pd.to_numeric(panel[target_col], errors="coerce")
            blocks.append(block)

    if not blocks:
        return pd.DataFrame(columns=[*key_cols, "indicator_name", "horizon", "target"])

    return pd.concat(blocks, ignore_index=True)


def validate_regression_dataset(
    df: pd.DataFrame,
    *,
    asset_col: str = "asset",
    datetime_col: str = "datetime",
    indicator_col: str = "indicator_name",
    horizon_col: str = "horizon",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Valide le dataset final pret pour la regression.

    Le controle reste volontairement simple et se concentre sur :
    - la presence des colonnes minimales ;
    - l'absence de doublons sur `(date, asset, indicator_name, horizon)` ;
    - quelques statistiques de forme pour le debugging du pipeline.
    """
    required_columns = [
        indicator_col,
        horizon_col,
        "signal_family",
        "signal_direction",
        "signal_active",
        "signal_strength",
        "signal_signed_value",
        "target",
    ]
    missing_required = [column for column in required_columns if column not in df.columns]
    if missing_required:
        raise ValueError(
            "Le dataset de regression ne contient pas toutes les colonnes attendues : "
            f"{', '.join(missing_required)}"
        )

    cfg = _resolve_config(None)
    ctx = _prepare_panel_context_with_preferences(
        df,
        asset_col=asset_col,
        datetime_col=datetime_col,
        config=cfg,
    )

    work = ctx.work.copy()
    duplicate_mask = work.duplicated(
        subset=[ctx.date_col, ctx.asset_col, indicator_col, horizon_col],
        keep=False,
    )
    duplicate_count = int(duplicate_mask.sum())

    context_feature_columns = list(df.attrs.get("context_feature_columns", []))
    if not context_feature_columns:
        reserved = {
            ctx.date_col,
            ctx.asset_col,
            indicator_col,
            horizon_col,
            "signal_family",
            "signal_direction",
            "signal_active",
            "signal_strength",
            "signal_signed_value",
            "target",
        }
        context_feature_columns = [column for column in df.columns if column not in reserved]

    summary: Dict[str, Any] = {
        "num_rows": int(len(df)),
        "num_assets": int(pd.Series(work[ctx.asset_col]).nunique(dropna=True)),
        "num_datetimes": int(pd.Series(work[ctx.date_col]).nunique(dropna=True)),
        "num_indicators": int(pd.Series(work[indicator_col]).nunique(dropna=True)),
        "num_horizons": int(pd.Series(work[horizon_col]).nunique(dropna=True)),
        "num_context_features": int(len(context_feature_columns)),
        "duplicate_count": duplicate_count,
        "target_nan_count": int(pd.to_numeric(work["target"], errors="coerce").isna().sum()),
        "active_row_count": int((pd.to_numeric(work["signal_active"], errors="coerce") == 1).sum()),
        "inactive_row_count": int((pd.to_numeric(work["signal_active"], errors="coerce") != 1).sum()),
        "is_valid": duplicate_count == 0,
    }

    if verbose:
        print("Validation regression dataset")
        print(f"- rows: {summary['num_rows']}")
        print(f"- assets: {summary['num_assets']}")
        print(f"- datetimes: {summary['num_datetimes']}")
        print(f"- indicators: {summary['num_indicators']}")
        print(f"- horizons: {summary['num_horizons']}")
        print(f"- context features: {summary['num_context_features']}")
        print(f"- target NaN: {summary['target_nan_count']}")
        print(f"- duplicates(date, asset, indicator, horizon): {summary['duplicate_count']}")

    if duplicate_count > 0:
        raise ValueError(
            "Le dataset final contient des doublons sur "
            f"({ctx.date_col}, {ctx.asset_col}, {indicator_col}, {horizon_col})."
        )

    return summary


def build_regression_dataset(
    df: pd.DataFrame,
    indicator_family_map: Optional[Mapping[str, str] | pd.DataFrame] = None,
    *,
    asset_col: str = "asset",
    datetime_col: str = "datetime",
    horizons: Optional[Sequence[str]] = None,
    include_inactive: bool = False,
    drop_missing_target: bool = True,
    exclude_vix_context: bool = True,
    context_family_name: str = "context_filter",
) -> pd.DataFrame:
    """
    Construit le dataset final long pret pour une regression panel.

    Chaque ligne du resultat represente une combinaison :
    `date x asset x indicator_name x horizon`.

    Le DataFrame d'entree est suppose deja contenir :
    - les indicateurs bruts ;
    - les colonnes de signaux ;
    - les colonnes de target ;
    - idealement les `attrs` de mapping produits par les etapes precedentes.
    """
    cfg = _resolve_config(None)
    ctx = _prepare_panel_context_with_preferences(
        df,
        asset_col=asset_col,
        datetime_col=datetime_col,
        config=cfg,
    )

    panel = ctx.work.copy()
    panel_context_cols = [
        column
        for column in ctx.context_cols
        if column not in {"indicator_name", "signal_family", "horizon", "target"}
    ]
    key_cols = [*panel_context_cols, ctx.date_col, ctx.asset_col]

    duplicate_base_count = int(panel.duplicated(subset=key_cols, keep=False).sum())
    if duplicate_base_count > 0:
        raise ValueError(
            "Le panel large contient des doublons sur la cle de base "
            f"({', '.join(key_cols)}). Le reshape long serait ambigu."
        )

    resolved_family_map = _coerce_indicator_family_map(df, indicator_family_map)
    signal_feature_map = identify_signal_feature_columns(panel)
    if not signal_feature_map:
        raise ValueError(
            "Aucune feature de signal exploitable n'a ete detectee dans le panel large."
        )

    target_columns = identify_target_columns(panel, horizons=horizons)
    if not target_columns:
        raise ValueError(
            "Aucune colonne de target `<feature>__target_<horizon>` n'a ete detectee."
        )

    resolved_horizons = _resolve_output_horizons(target_columns, horizons)

    features_with_targets = [
        feature
        for feature in signal_feature_map
        if any(horizon in target_columns.get(feature, {}) for horizon in resolved_horizons)
    ]
    if not features_with_targets:
        raise ValueError(
            "Aucun indicateur ne dispose a la fois des colonnes de signal requises et "
            "d'au moins une target sur les horizons demandes."
        )

    dropped_signal_features = [
        feature for feature in signal_feature_map if feature not in features_with_targets
    ]
    if dropped_signal_features:
        LOGGER.warning(
            "Indicateurs de signal ignores faute de target associee : %s",
            ", ".join(dropped_signal_features),
        )

    missing_families = [
        feature for feature in features_with_targets if feature not in resolved_family_map
    ]
    if missing_families:
        raise ValueError(
            "Famille d'indicateur introuvable pour : "
            f"{', '.join(missing_families)}"
        )

    filtered_signal_feature_map = {
        feature: signal_feature_map[feature]
        for feature in features_with_targets
    }
    filtered_target_columns = {
        feature: {
            horizon: column
            for horizon, column in target_columns.get(feature, {}).items()
            if horizon in resolved_horizons
        }
        for feature in features_with_targets
    }

    signal_long = _build_signal_long_frame(
        panel,
        key_cols=key_cols,
        signal_feature_map=filtered_signal_feature_map,
        family_map=resolved_family_map,
    )
    target_long = _build_target_long_frame(
        panel,
        key_cols=key_cols,
        target_columns=filtered_target_columns,
        horizons=resolved_horizons,
    )

    regression_df = signal_long.merge(
        target_long,
        on=[*key_cols, "indicator_name"],
        how="inner",
        validate="one_to_many",
    )

    context_feature_columns = get_context_feature_columns(
        panel,
        resolved_family_map,
        context_family_name=context_family_name,
        exclude_vix_context=exclude_vix_context,
    )
    if context_feature_columns:
        context_frame = panel[key_cols + context_feature_columns].copy()
        regression_df = regression_df.merge(
            context_frame,
            on=key_cols,
            how="left",
            validate="many_to_one",
        )

    if not include_inactive:
        regression_df = regression_df[
            pd.to_numeric(regression_df["signal_active"], errors="coerce") == 1
        ].copy()

    if drop_missing_target:
        regression_df = regression_df[
            pd.to_numeric(regression_df["target"], errors="coerce").notna()
        ].copy()

    output_columns = [
        ctx.date_col,
        ctx.asset_col,
        *panel_context_cols,
        "indicator_name",
        "signal_family",
        "horizon",
        "signal_direction",
        "signal_active",
        "signal_strength",
        "signal_signed_value",
        "target",
        *context_feature_columns,
    ]
    output_columns = [column for column in output_columns if column in regression_df.columns]

    regression_df = regression_df[output_columns].sort_values(
        [ctx.date_col, ctx.asset_col, "indicator_name", "horizon"]
    ).reset_index(drop=True)

    regression_df.attrs = deepcopy(getattr(df, "attrs", {}))
    regression_df.attrs["regression_indicator_features"] = features_with_targets
    regression_df.attrs["regression_horizons"] = resolved_horizons
    regression_df.attrs["context_feature_columns"] = context_feature_columns
    regression_df.attrs["panel_context_columns"] = panel_context_cols
    regression_df.attrs["regression_signal_column_map"] = filtered_signal_feature_map
    regression_df.attrs["regression_target_column_map"] = filtered_target_columns

    summary = validate_regression_dataset(
        regression_df,
        asset_col=ctx.asset_col,
        datetime_col=ctx.date_col,
        verbose=False,
    )
    summary["num_signal_features_detected"] = len(signal_feature_map)
    summary["num_signal_features_retained"] = len(features_with_targets)
    summary["num_horizons_detected"] = len(resolved_horizons)
    regression_df.attrs["regression_dataset_summary"] = summary

    LOGGER.info(
        "Dataset de regression construit : %d lignes | %d indicateurs | %d horizons | %d context features",
        len(regression_df),
        len(features_with_targets),
        len(resolved_horizons),
        len(context_feature_columns),
    )

    return regression_df


def _unique_existing_columns(columns: Sequence[str], available_columns: Sequence[str]) -> List[str]:
    available = set(available_columns)
    out: List[str] = []
    for column in columns:
        if column in available and column not in out:
            out.append(column)
    return out


def _require_requested_columns(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]],
    *,
    label: str,
) -> List[str]:
    if columns is None:
        return []

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes {label} demandees mais absentes du DataFrame : "
            f"{', '.join(missing)}"
        )
    return list(columns)


def infer_model_context_features(
    df: pd.DataFrame,
    *,
    target_col: str = "target",
    datetime_col: str = "datetime",
    asset_col: str = "asset",
    indicator_col: str = "indicator_name",
    family_col: str = "signal_family",
    horizon_col: str = "horizon",
    exclude_vix_context: bool = True,
) -> List[str]:
    """
    Identifie les variables de contexte numeriques du dataset long final.

    Priorite :
    - utiliser `df.attrs["context_feature_columns"]` si disponible ;
    - sinon inferer les colonnes numeriques qui ne sont ni signal, ni target,
      ni metadata, ni colonnes internes.
    """
    attrs_context = [
        column
        for column in df.attrs.get("context_feature_columns", [])
        if column in df.columns
    ]
    if attrs_context:
        context_columns = attrs_context
    else:
        reserved = {
            target_col,
            datetime_col,
            asset_col,
            indicator_col,
            family_col,
            horizon_col,
            "signal_direction",
            "signal_active",
            "signal_strength",
            "signal_signed_value",
        }
        excluded_prefixes = ("pred_", "rank_", "__")
        excluded_patterns = ("__signal_", "__target_", "future_return")
        context_columns = []
        for column in df.columns:
            name = str(column)
            lowered = name.lower()
            if name in reserved:
                continue
            if lowered.startswith(excluded_prefixes):
                continue
            if any(pattern in lowered for pattern in excluded_patterns):
                continue
            if not pd.api.types.is_numeric_dtype(df[column]):
                continue
            context_columns.append(name)

    filtered_columns = []
    for column in context_columns:
        if exclude_vix_context and "vix" in str(column).lower():
            continue
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            filtered_columns.append(str(column))

    return list(dict.fromkeys(filtered_columns))


def prepare_logit_inputs(
    df: pd.DataFrame,
    *,
    target_col: str = "target",
    datetime_col: str = "datetime",
    asset_col: str = "asset",
    indicator_col: str = "indicator_name",
    family_col: str = "signal_family",
    horizon_col: str = "horizon",
    include_indicator_name: bool = True,
    include_signal_family: bool = True,
    include_horizon: bool = True,
    include_asset: bool = False,
    include_signal_active: bool = False,
    filter_active_only: bool = True,
    drop_na_target: bool = True,
    extra_numeric_features: Optional[Sequence[str]] = None,
    extra_categorical_features: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Prepare le dataset long final pour une regression logistique penalisee.

    La fonction ne fit aucun scaler : elle ne fait que selectionner les lignes,
    separer `X` / `y`, et lister les colonnes numeriques et categorielles.
    Le fit des transformations reste dans le pipeline sklearn pour eviter toute
    fuite d'information.
    """
    required_columns = [
        datetime_col,
        indicator_col,
        family_col,
        horizon_col,
        "signal_direction",
        "signal_strength",
        "signal_signed_value",
    ]
    if filter_active_only or include_signal_active:
        required_columns.append("signal_active")
    missing_required = [column for column in required_columns if column not in df.columns]
    if missing_required:
        raise ValueError(
            "Colonnes indispensables absentes pour la preparation logit : "
            f"{', '.join(missing_required)}"
        )

    df_model = df.copy()
    df_model[datetime_col] = pd.to_datetime(df_model[datetime_col], errors="coerce")
    if df_model[datetime_col].isna().any():
        raise ValueError(
            f"La colonne `{datetime_col}` contient des dates non convertibles."
        )

    if filter_active_only:
        df_model = df_model[
            pd.to_numeric(df_model["signal_active"], errors="coerce") == 1
        ].copy()

    if drop_na_target:
        if target_col not in df_model.columns:
            raise ValueError(f"Colonne target absente : `{target_col}`.")
        df_model = df_model[
            pd.to_numeric(df_model[target_col], errors="coerce").notna()
        ].copy()

    base_numeric = ["signal_direction", "signal_strength", "signal_signed_value"]
    if include_signal_active:
        base_numeric.append("signal_active")

    context_numeric = infer_model_context_features(
        df_model,
        target_col=target_col,
        datetime_col=datetime_col,
        asset_col=asset_col,
        indicator_col=indicator_col,
        family_col=family_col,
        horizon_col=horizon_col,
        exclude_vix_context=True,
    )
    requested_numeric = _require_requested_columns(
        df_model,
        extra_numeric_features,
        label="numeriques supplementaires",
    )
    numeric_features = _unique_existing_columns(
        [*base_numeric, *context_numeric, *requested_numeric],
        df_model.columns,
    )

    categorical_features: List[str] = []
    if include_indicator_name:
        categorical_features.append(indicator_col)
    if include_signal_family:
        categorical_features.append(family_col)
    if include_horizon:
        categorical_features.append(horizon_col)
    if include_asset:
        categorical_features.append(asset_col)
    requested_categorical = _require_requested_columns(
        df_model,
        extra_categorical_features,
        label="categorielles supplementaires",
    )
    categorical_features = _unique_existing_columns(
        [*categorical_features, *requested_categorical],
        df_model.columns,
    )

    for column in numeric_features:
        df_model[column] = pd.to_numeric(df_model[column], errors="coerce")

    feature_columns = [*numeric_features, *categorical_features]
    helper_columns = _unique_existing_columns(
        [datetime_col, asset_col, horizon_col],
        df_model.columns,
    )
    model_input_columns = _unique_existing_columns(
        [*feature_columns, *helper_columns],
        df_model.columns,
    )

    y = (
        pd.to_numeric(df_model[target_col], errors="coerce")
        if target_col in df_model.columns
        else None
    )

    return {
        "df_model": df_model.reset_index(drop=True),
        "X": df_model[feature_columns].copy().reset_index(drop=True),
        "X_model": df_model[model_input_columns].copy().reset_index(drop=True),
        "y": y.reset_index(drop=True) if y is not None else None,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_columns": feature_columns,
        "model_input_columns": model_input_columns,
        "context_feature_columns": context_numeric,
    }


class CrossSectionalNumericScaler(BaseEstimator, TransformerMixin):
    """
    Standardise des colonnes numeriques en coupe transversale a date t.

    Convention retenue pour le dataset long :
    - les groupes sont `[datetime]` ou `[datetime, horizon]` si la colonne
      horizon est disponible ;
    - la transformation utilise uniquement les valeurs presentes dans le
      groupe courant a la date t, sans statistique globale fittee sur le futur ;
    - `fit` ne memorise aucune moyenne globale, ce qui evite le leakage temporel.
    """

    def __init__(
        self,
        numeric_features: Sequence[str],
        *,
        datetime_col: str = "datetime",
        horizon_col: str = "horizon",
        min_group_size: int = 2,
        min_std: float = 1e-12,
        small_std_policy: str = "nan",
        ddof: int = 1,
    ) -> None:
        self.numeric_features = list(numeric_features)
        self.datetime_col = datetime_col
        self.horizon_col = horizon_col
        self.min_group_size = min_group_size
        self.min_std = min_std
        self.small_std_policy = small_std_policy
        self.ddof = ddof

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "`CrossSectionalNumericScaler` attend un DataFrame pandas en entree."
            )
        if self.datetime_col not in X.columns:
            raise ValueError(
                f"La colonne `{self.datetime_col}` est indispensable au scaling cross-sectionnel."
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "`CrossSectionalNumericScaler` attend un DataFrame pandas en entree."
            )

        out = X.copy()
        group_cols = [self.datetime_col]
        if self.horizon_col in out.columns:
            group_cols.append(self.horizon_col)

        group_keys = [out[col] for col in group_cols]
        for column in self.numeric_features:
            if column not in out.columns:
                continue

            values = pd.to_numeric(out[column], errors="coerce")
            grouped = values.groupby(group_keys, sort=False)
            group_mean = grouped.transform("mean")
            group_std = grouped.transform(lambda x: x.std(ddof=self.ddof))
            group_count = grouped.transform("count")

            valid_std = group_std.where(group_count >= self.min_group_size)
            if self.small_std_policy == "epsilon":
                valid_std = valid_std.where(valid_std.isna(), valid_std.clip(lower=self.min_std))
            elif self.small_std_policy == "nan":
                valid_std = valid_std.where(valid_std >= self.min_std)
            else:
                raise ValueError("small_std_policy doit valoir `nan` ou `epsilon`.")

            out[column] = (values - group_mean) / valid_std

        return out

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        if input_features is None:
            return np.asarray(self.numeric_features, dtype=object)
        return np.asarray(input_features, dtype=object)


def build_logit_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    *,
    scaling_mode: str = "classical",
    datetime_col: str = "datetime",
    asset_col: str = "asset",
    horizon_col: str = "horizon",
) -> ColumnTransformer | Pipeline:
    """
    Construit le preprocesseur sklearn pour la logistique penalisee.

    Modes de scaling :
    - `classical` : `StandardScaler` fit uniquement sur le train via sklearn ;
    - `cross_sectional` : z-score par groupe `[datetime, horizon]` a date t ;
    - `none` : imputation numerique sans standardisation.
    """
    scaling_mode = str(scaling_mode).lower()
    if scaling_mode not in {"classical", "cross_sectional", "none"}:
        raise ValueError("scaling_mode doit valoir `classical`, `cross_sectional` ou `none`.")

    numeric_steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]
    if scaling_mode == "classical":
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(numeric_steps)
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers: List[Tuple[str, Any, Sequence[str]]] = []
    if numeric_features:
        transformers.append(("numeric", numeric_transformer, list(numeric_features)))
    if categorical_features:
        transformers.append(("categorical", categorical_transformer, list(categorical_features)))

    column_preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    if scaling_mode != "cross_sectional":
        return column_preprocessor

    return Pipeline(
        [
            (
                "cross_sectional_scaler",
                CrossSectionalNumericScaler(
                    numeric_features=list(numeric_features),
                    datetime_col=datetime_col,
                    horizon_col=horizon_col,
                ),
            ),
            ("columns", column_preprocessor),
        ]
    )


__all__ = [
    "add_all_signal_features",
    "add_bollinger_band_signals",
    "add_centered_oscillator_signals",
    "add_context_filter_signals",
    "add_extreme_zone_oscillator_signals",
    "add_trend_filter_signals",
    "add_volume_confirmation_signals",
    "build_default_indicator_family_map",
    "build_logit_preprocessor",
    "build_regression_dataset",
    "compute_cross_sectional_leave_one_out_zscore",
    "CrossSectionalNumericScaler",
    "infer_model_context_features",
    "get_context_feature_columns",
    "get_added_signal_columns",
    "get_default_signal_config",
    "identify_signal_feature_columns",
    "identify_target_columns",
    "prepare_logit_inputs",
    "print_signal_summary",
    "summarize_signal_features",
    "validate_regression_dataset",
]

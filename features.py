from __future__ import annotations

from collections.abc import Mapping as ABCMapping
import logging
import math
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
_OPTIONAL_WARNING_CACHE: set[str] = set()

REQUIRED_OHLC_COLUMNS: Tuple[str, ...] = ("open", "high", "low", "close")
OPTIONAL_COLUMNS: Tuple[str, ...] = ("volume", "vix")

INDICATOR_REGISTRY: Dict[str, Dict[str, Any]] = {
    "acceleration_bands": {
        "display_names": ["Acceleration Bands"],
        "family": "band_channel_level",
    },
    "bollinger_bands": {
        "display_names": ["Bollinger Bands"],
        "family": "band_channel_level",
    },
    "donchian_channel": {
        "display_names": ["Donchian Channel"],
        "family": "band_channel_level",
    },
    "keltner_channel": {
        "display_names": ["Keltner Channel"],
        "family": "band_channel_level",
    },
    "absolute_price_oscillator": {
        "display_names": ["Absolute Price Oscillator"],
        "family": "centered_oscillator",
    },
    "accumulation_distribution_oscillator": {
        "display_names": ["Accumulation/Distribution Oscillator"],
        "family": "centered_oscillator",
    },
    "awesome_oscillator": {
        "display_names": ["Awesome Oscillator"],
        "family": "centered_oscillator",
    },
    "balance_of_power": {
        "display_names": ["Balance of Power"],
        "family": "centered_oscillator",
    },
    "bias_indicator": {
        "display_names": ["Bias Indicator"],
        "family": "centered_oscillator",
    },
    "chande_forecast_oscillator": {
        "display_names": ["Chande Forecast Oscillator"],
        "family": "centered_oscillator",
    },
    "chande_momentum_oscillator": {
        "display_names": ["Chande Momentum Oscillator"],
        "family": "centered_oscillator",
    },
    "coppock_curve": {
        "display_names": ["Coppock Curve"],
        "family": "centered_oscillator",
    },
    "detrended_price_oscillator": {
        "display_names": ["Detrended Price Oscillator"],
        "family": "centered_oscillator",
    },
    "ease_of_movement": {
        "display_names": ["Ease of Movement (EOM)"],
        "family": "centered_oscillator",
    },
    "elders_force_index": {
        "display_names": ["Elder's Force Index"],
        "family": "centered_oscillator",
    },
    "know_sure_thing": {
        "display_names": ["Know Sure Thing (KST)"],
        "family": "centered_oscillator",
    },
    "macd": {
        "display_names": ["MACD (Moving Average Convergence Divergence)"],
        "family": "centered_oscillator",
    },
    "momentum_indicator": {
        "display_names": ["Momentum Indicator"],
        "family": "centered_oscillator",
    },
    "percentage_price_oscillator": {
        "display_names": ["Percentage Price Oscillator (PPO)"],
        "family": "centered_oscillator",
    },
    "percentage_volume_oscillator": {
        "display_names": ["Percentage Volume Oscillator (PVO)"],
        "family": "centered_oscillator",
    },
    "qstick_indicator": {
        "display_names": ["Qstick Indicator"],
        "family": "centered_oscillator",
    },
    "rate_of_change": {
        "display_names": ["Rate of Change (ROC)"],
        "family": "centered_oscillator",
    },
    "relative_vigor_index": {
        "display_names": ["Relative Vigor Index (RVI)"],
        "family": "centered_oscillator",
    },
    "adx": {
        "display_names": ["ADX (Average Directional Index)"],
        "family": "context_filter",
    },
    "atr": {
        "display_names": ["ATR (Average True Range)", "Average True Range (ATR)"],
        "family": "context_filter",
    },
    "choppiness_index": {
        "display_names": ["Choppiness Index"],
        "family": "context_filter",
    },
    "entropy_indicator": {
        "display_names": ["Entropy Indicator"],
        "family": "context_filter",
    },
    "high_low_close_average": {
        "display_names": ["High-Low-Close Average"],
        "family": "context_filter",
    },
    "kurtosis": {
        "display_names": ["Kurtosis"],
        "family": "context_filter",
    },
    "mass_index": {
        "display_names": ["Mass Index"],
        "family": "context_filter",
    },
    "mean_absolute_deviation": {
        "display_names": ["Mean Absolute Deviation"],
        "family": "context_filter",
    },
    "median_price": {
        "display_names": ["Median Price"],
        "family": "context_filter",
    },
    "midpoint_indicator": {
        "display_names": ["Midpoint Indicator"],
        "family": "context_filter",
    },
    "midprice_indicator": {
        "display_names": ["Midprice Indicator"],
        "family": "context_filter",
    },
    "normalized_atr": {
        "display_names": ["Normalized ATR"],
        "family": "context_filter",
    },
    "open_high_low_close_average": {
        "display_names": ["Open-High-Low-Close Average"],
        "family": "context_filter",
    },
    "price_distance": {
        "display_names": ["Price Distance"],
        "family": "context_filter",
    },
    "quantile_indicator": {
        "display_names": ["Quantile Indicator"],
        "family": "context_filter",
    },
    "vix": {
        "display_names": ["VIX (Volatility Index)"],
        "family": "context_filter",
    },
    "center_of_gravity": {
        "display_names": ["Center of Gravity"],
        "family": "extreme_zone_oscillator",
    },
    "commodity_channel_index": {
        "display_names": ["Commodity Channel Index (CCI)"],
        "family": "extreme_zone_oscillator",
    },
    "emotion_index_willingness_index": {
        "display_names": ["Emotion Index & Willingness Index"],
        "family": "extreme_zone_oscillator",
    },
    "money_flow_index": {
        "display_names": ["Money Flow Index (MFI)"],
        "family": "extreme_zone_oscillator",
    },
    "pretty_good_oscillator": {
        "display_names": ["Pretty Good Oscillator"],
        "family": "extreme_zone_oscillator",
    },
    "rsi": {
        "display_names": ["RSI (Relative Strength Index)"],
        "family": "extreme_zone_oscillator",
    },
    "rsx": {
        "display_names": ["Relative Strength Xtra (RSX)"],
        "family": "extreme_zone_oscillator",
    },
    "stochastic_oscillator": {
        "display_names": ["Stochastic Oscillator"],
        "family": "extreme_zone_oscillator",
    },
    "alma": {
        "display_names": ["Arnaud Legoux Moving Average (ALMA)"],
        "family": "trend_filter",
    },
    "aroon_indicator": {
        "display_names": ["Aroon Indicator"],
        "family": "trend_filter",
    },
    "correlation_trend_indicator": {
        "display_names": ["Correlation Trend Indicator"],
        "family": "trend_filter",
    },
    "efficiency_ratio": {
        "display_names": ["Efficiency Ratio"],
        "family": "trend_filter",
    },
    "ehlers_super_smoother_filter": {
        "display_names": ["Ehlers Super Smoother Filter"],
        "family": "trend_filter",
    },
    "even_better_sinewave": {
        "display_names": ["Even Better Sinewave"],
        "family": "trend_filter",
    },
    "fibonacci_weighted_moving_average": {
        "display_names": ["Fibonacci Weighted Moving Average"],
        "family": "trend_filter",
    },
    "ichimoku": {
        "display_names": ["Ichimoku Cloud", "Ichimoku Kinko Hyo"],
        "family": "trend_filter",
    },
    "inertia_indicator": {
        "display_names": ["Inertia Indicator"],
        "family": "trend_filter",
    },
    "kaufman_adaptive_moving_average": {
        "display_names": ["Kaufman Adaptive Moving Average"],
        "family": "trend_filter",
    },
    "linear_decay": {
        "display_names": ["Linear Decay"],
        "family": "trend_filter",
    },
    "linear_regression": {
        "display_names": ["Linear Regression"],
        "family": "trend_filter",
    },
    "moving_averages": {
        "display_names": ["Moving Averages (SMA/EMA)"],
        "family": "trend_filter",
    },
    "parabolic_sar": {
        "display_names": ["Parabolic SAR"],
        "family": "trend_filter",
    },
    "accumulation_distribution_line": {
        "display_names": ["Accumulation/Distribution Line"],
        "family": "volume_confirmation",
    },
    "chaikin_money_flow": {
        "display_names": ["Chaikin Money Flow (CMF)"],
        "family": "volume_confirmation",
    },
    "elder_ray_index": {
        "display_names": ["Elder Ray Index"],
        "family": "volume_confirmation",
    },
    "negative_volume_index": {
        "display_names": ["Negative Volume Index (NVI)"],
        "family": "volume_confirmation",
    },
    "on_balance_volume": {
        "display_names": ["On-Balance Volume (OBV)"],
        "family": "volume_confirmation",
    },
    "positive_volume_index": {
        "display_names": ["Positive Volume Index (PVI)"],
        "family": "volume_confirmation",
    },
    "price_volume_rank": {
        "display_names": ["Price Volume Rank"],
        "family": "volume_confirmation",
    },
    "price_volume_trend": {
        "display_names": ["Price Volume Trend (PVT)"],
        "family": "volume_confirmation",
    },
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "general": {
        "include_all_original_columns": False,
        "include_base_features": True,
        "print_summary": False,
    },
    "bollinger": {"window": 20, "std_multiplier": 2.0},
    "donchian": {"window": 20},
    "keltner": {"ema_window": 20, "atr_window": 10, "multiplier": 2.0},
    "acceleration_bands": {"window": 20},
    "apo": {"fast": 12, "slow": 26},
    "ado": {"fast": 3, "slow": 10},
    "awesome": {"fast": 5, "slow": 34},
    "bias": {"window": 20},
    "cfo": {"window": 14},
    "cmo": {"window": 14},
    "coppock": {"long_window": 14, "short_window": 11, "wma_window": 10},
    "dpo": {"window": 20},
    "eom": {"window": 14, "smoothing_window": 14},
    "efi": {"window": 13},
    "kst": {
        "roc_windows": [10, 15, 20, 30],
        "sma_windows": [10, 10, 10, 15],
        "weights": [1, 2, 3, 4],
        "signal_window": 9,
    },
    "macd": {"fast": 12, "slow": 26, "signal_window": 9},
    "momentum": {"window": 10},
    "ppo": {"fast": 12, "slow": 26, "signal_window": 9},
    "pvo": {"fast": 12, "slow": 26, "signal_window": 9},
    "qstick": {"window": 10},
    "roc": {"window": 10},
    "rvi": {"window": 10, "signal_window": 4},
    "atr": {"window": 14},
    "adx": {"window": 14},
    "choppiness": {"window": 14},
    "entropy": {"window": 20, "mode": "sign", "bins": 10},
    "kurtosis": {"window": 20},
    "mass_index": {"ema_window": 9, "sum_window": 25},
    "mad": {"window": 20, "source": "close"},
    "midpoint": {"window": 14},
    "midprice": {"window": 14},
    "price_distance": {"window": 20, "reference": "sma"},
    "quantile": {"window": 20},
    "vix": {"roc_window": 10, "zscore_window": 20, "sma_window": 20},
    "rsi": {"window": 14},
    "stochastic": {"window": 14, "signal_window": 3},
    "cci": {"window": 20},
    "mfi": {"window": 14},
    "center_of_gravity": {"window": 10},
    "pgo": {"window": 14},
    "rsx": {"window": 14, "smoothing_window": 5},
    "emotion_willingness": {"window": 14},
    "ma": {"sma_windows": [10, 20, 50], "ema_windows": [10, 20, 50]},
    "alma": {"window": 10, "offset": 0.85, "sigma": 6.0},
    "aroon": {"window": 25},
    "cti": {"window": 20},
    "efficiency_ratio": {"window": 10},
    "super_smoother": {"period": 10},
    "even_better_sinewave": {"period": 10, "signal_window": 5},
    "fibonacci_wma": {"window": 10},
    "ichimoku": {"tenkan_window": 9, "kijun_window": 26, "span_b_window": 52},
    "inertia": {"window": 20},
    "kama": {"er_window": 10, "fast": 2, "slow": 30},
    "linear_decay": {"window": 10},
    "linear_regression": {"window": 20},
    "psar": {"step": 0.02, "max_step": 0.2},
    "cmf": {"window": 20},
    "elder_ray": {"window": 13},
    "nvi": {"start_value": 1000.0},
    "pvi": {"start_value": 1000.0},
    "price_volume_rank": {"window": 20},
}


def get_default_indicator_config() -> Dict[str, Any]:
    """
    Retourne une copie profonde de la configuration par defaut.
    """
    return deepcopy(DEFAULT_CONFIG)


def get_indicator_registry() -> Dict[str, Dict[str, Any]]:
    """
    Retourne le registre interne des indicateurs et de leurs familles.
    """
    return deepcopy(INDICATOR_REGISTRY)


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
    return _deep_update(get_default_indicator_config(), config)


def _format_token(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        text = f"{float(value):.6f}".rstrip("0").rstrip(".")
        return text.replace(".", "p").replace("-", "neg")
    return str(value).replace(".", "p").replace("-", "neg")


def _warn_optional(message: str, *, once: bool = True) -> None:
    """
    Emet un warning optionnel, avec deduplication par defaut.

    Cela evite d'inonder le notebook lorsque la meme absence de colonne
    optionnelle est rencontree pour de nombreux actifs ou sous-panels.
    """
    if once:
        if message in _OPTIONAL_WARNING_CACHE:
            return
        _OPTIONAL_WARNING_CACHE.add(message)
    LOGGER.warning(message)


def _safe_divide(
    numerator: pd.Series | pd.DataFrame,
    denominator: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def _empty_feature_frame(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(index=index)


def _nan_frame(index: pd.Index, columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(np.nan, index=index) for col in columns}, index=index)


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _wilder_ma(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def _wma(series: pd.Series, window: int, weights: Optional[np.ndarray] = None) -> pd.Series:
    if weights is None:
        weights = np.arange(1, window + 1, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return series.rolling(window=window, min_periods=window).apply(
        lambda values: float(np.dot(values, weights)),
        raw=True,
    )


def _rolling_mad(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).apply(
        lambda values: float(np.mean(np.abs(values - np.mean(values)))),
        raw=True,
    )


def _rolling_rms(series: pd.Series, window: int) -> pd.Series:
    return np.sqrt((series ** 2).rolling(window=window, min_periods=window).mean())


def _rolling_percent_rank(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).apply(
        lambda values: float(np.mean(values <= values[-1])) if not np.isnan(values[-1]) else np.nan,
        raw=True,
    )


def _rolling_entropy(series: pd.Series, window: int, mode: str = "sign", bins: int = 10) -> pd.Series:
    def _entropy(values: np.ndarray) -> float:
        clean = values[~np.isnan(values)]
        if clean.size == 0:
            return np.nan

        if mode == "bins":
            hist, _ = np.histogram(clean, bins=bins)
            probs = hist[hist > 0] / hist.sum()
            max_entropy = math.log(max(bins, 2))
        else:
            states = np.sign(clean)
            _, counts = np.unique(states, return_counts=True)
            probs = counts / counts.sum()
            max_entropy = math.log(3)

        if probs.size == 0 or max_entropy == 0:
            return np.nan

        entropy = -np.sum(probs * np.log(probs))
        return float(entropy / max_entropy)

    return series.rolling(window=window, min_periods=window).apply(_entropy, raw=True)


def _rolling_linear_regression(
    series: pd.Series,
    window: int,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denom = np.sum((x - x_mean) ** 2)

    def _slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        y_mean = values.mean()
        return float(np.dot(x - x_mean, values - y_mean) / denom)

    slope = series.rolling(window=window, min_periods=window).apply(_slope, raw=True)
    mean = series.rolling(window=window, min_periods=window).mean()
    intercept = mean - slope * x_mean
    fitted = intercept + slope * (window - 1)
    return slope, intercept, fitted


def _rolling_corr_with_time(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)

    def _corr(values: np.ndarray) -> float:
        if np.isnan(values).any() or np.std(values) == 0:
            return np.nan
        return float(np.corrcoef(x, values)[0, 1])

    return series.rolling(window=window, min_periods=window).apply(_corr, raw=True)


def _fibonacci_weights(window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("La fenetre Fibonacci doit etre strictement positive.")
    if window == 1:
        return np.array([1.0], dtype=float)

    fib = [1.0, 1.0]
    while len(fib) < window:
        fib.append(fib[-1] + fib[-2])
    return np.array(fib[:window], dtype=float)


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    return _wilder_ma(_true_range(df), window)


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = _wilder_ma(gains, window)
    avg_loss = _wilder_ma(losses, window)

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return rsi


def _adx(df: pd.DataFrame, window: int) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    high = df["high"]
    low = df["low"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    atr = _atr(df, window)
    plus_di = 100 * _safe_divide(_wilder_ma(plus_dm, window), atr)
    minus_di = 100 * _safe_divide(_wilder_ma(minus_dm, window), atr)
    dx = 100 * _safe_divide((plus_di - minus_di).abs(), plus_di + minus_di)
    adx = _wilder_ma(dx, window)
    return adx, plus_di, minus_di, dx


def _money_flow_components(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    spread = df["high"] - df["low"]
    multiplier = _safe_divide(
        ((df["close"] - df["low"]) - (df["high"] - df["close"])),
        spread,
    )
    volume = df["volume"] if "volume" in df.columns else pd.Series(np.nan, index=df.index)
    money_flow_volume = multiplier * volume
    return multiplier, money_flow_volume


def _chaikin_ad_line(df: pd.DataFrame) -> pd.Series:
    _, money_flow_volume = _money_flow_components(df)
    return money_flow_volume.cumsum()


def _center_of_gravity(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1, dtype=float)

    def _cog(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        reversed_values = values[::-1]
        denominator = reversed_values.sum()
        if denominator == 0:
            return np.nan
        return float(-np.dot(weights, reversed_values) / denominator)

    return series.rolling(window=window, min_periods=window).apply(_cog, raw=True)


def _alma(series: pd.Series, window: int, offset: float, sigma: float) -> pd.Series:
    m = offset * (window - 1)
    s = window / sigma
    weights = np.exp(-((np.arange(window) - m) ** 2) / (2 * (s ** 2)))
    weights = weights / weights.sum()
    return _wma(series, window, weights=weights)


def _super_smoother(series: pd.Series, period: int) -> pd.Series:
    values = series.to_numpy(dtype=float)
    output = np.full_like(values, np.nan, dtype=float)

    if period <= 0:
        raise ValueError("La periode du Super Smoother doit etre strictement positive.")

    a1 = math.exp(-1.414 * math.pi / period)
    b1 = 2 * a1 * math.cos(1.414 * math.pi / period)
    c2 = b1
    c3 = -(a1 ** 2)
    c1 = 1 - c2 - c3

    for i in range(len(values)):
        if np.isnan(values[i]):
            output[i] = np.nan
        elif i < 2 or np.isnan(output[i - 1]) or np.isnan(output[i - 2]) or np.isnan(values[i - 1]):
            output[i] = values[i]
        else:
            output[i] = (
                c1 * (values[i] + values[i - 1]) / 2.0
                + c2 * output[i - 1]
                + c3 * output[i - 2]
            )

    return pd.Series(output, index=series.index, name=f"super_smoother_{period}")


def _even_better_sinewave(
    series: pd.Series,
    period: int,
    signal_window: int,
) -> Tuple[pd.Series, pd.Series]:
    """
    Approximation causale inspiree d'Ehlers :
    on retire une version Super Smoother du prix puis on normalise
    l'oscillation residuelle par sa RMS roulante.
    """
    smoothed = _super_smoother(series, period)
    cycle = series - smoothed
    rms = _rolling_rms(cycle, period)
    oscillator = _safe_divide(cycle, rms)
    signal = _ema(oscillator, signal_window)
    return oscillator, signal


def _kama(series: pd.Series, er_window: int, fast: int, slow: int) -> pd.Series:
    change = (series - series.shift(er_window)).abs()
    volatility = series.diff().abs().rolling(er_window, min_periods=er_window).sum()
    er = _safe_divide(change, volatility).fillna(0.0)

    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    smoothing_constant = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama = pd.Series(np.nan, index=series.index)
    if len(series) < er_window:
        return kama

    start = er_window - 1
    kama.iloc[start] = series.iloc[:er_window].mean()

    for i in range(start + 1, len(series)):
        if np.isnan(series.iloc[i]):
            kama.iloc[i] = kama.iloc[i - 1]
            continue
        prev = kama.iloc[i - 1]
        if np.isnan(prev):
            prev = series.iloc[i - 1]
        kama.iloc[i] = prev + smoothing_constant.iloc[i] * (series.iloc[i] - prev)

    return kama


def _psar(df: pd.DataFrame, step: float, max_step: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    index = df.index

    psar = np.full(len(df), np.nan, dtype=float)
    trend = np.full(len(df), np.nan, dtype=float)
    af_series = np.full(len(df), np.nan, dtype=float)

    if len(df) == 0:
        return pd.Series(psar, index=index), pd.Series(trend, index=index), pd.Series(af_series, index=index)

    uptrend = True
    if len(df) > 1 and not np.isnan(close[1]) and not np.isnan(close[0]):
        uptrend = close[1] >= close[0]

    ep = high[0] if uptrend else low[0]
    psar[0] = low[0] if uptrend else high[0]
    trend[0] = 1.0 if uptrend else -1.0
    af = step
    af_series[0] = af

    for i in range(1, len(df)):
        prev_psar = psar[i - 1]

        if uptrend:
            candidate = prev_psar + af * (ep - prev_psar)
            candidate = min(candidate, low[i - 1], low[i - 2] if i > 1 else low[i - 1])
            if low[i] < candidate:
                uptrend = False
                psar[i] = ep
                ep = low[i]
                af = step
            else:
                psar[i] = candidate
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step, max_step)
        else:
            candidate = prev_psar + af * (ep - prev_psar)
            candidate = max(candidate, high[i - 1], high[i - 2] if i > 1 else high[i - 1])
            if high[i] > candidate:
                uptrend = True
                psar[i] = ep
                ep = high[i]
                af = step
            else:
                psar[i] = candidate
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step, max_step)

        trend[i] = 1.0 if uptrend else -1.0
        af_series[i] = af

    return (
        pd.Series(psar, index=index),
        pd.Series(trend, index=index),
        pd.Series(af_series, index=index),
    )


def _nvi_or_pvi(
    close: pd.Series,
    volume: pd.Series,
    start_value: float,
    positive: bool,
) -> pd.Series:
    output = np.full(len(close), np.nan, dtype=float)
    if len(close) == 0:
        return pd.Series(output, index=close.index)

    output[0] = start_value
    close_values = close.to_numpy(dtype=float)
    volume_values = volume.to_numpy(dtype=float)

    for i in range(1, len(close_values)):
        prev_index = output[i - 1]
        if np.isnan(prev_index):
            prev_index = start_value

        current_close = close_values[i]
        prev_close = close_values[i - 1]
        current_volume = volume_values[i]
        prev_volume = volume_values[i - 1]

        if np.isnan(current_close) or np.isnan(prev_close) or prev_close == 0:
            output[i] = prev_index
            continue

        if np.isnan(current_volume) or np.isnan(prev_volume):
            output[i] = prev_index
            continue

        volume_condition = current_volume > prev_volume if positive else current_volume < prev_volume
        if volume_condition:
            output[i] = prev_index * (1.0 + (current_close - prev_close) / prev_close)
        else:
            output[i] = prev_index

    return pd.Series(output, index=close.index)


def _rsx_approximation(series: pd.Series, window: int, smoothing_window: int) -> pd.Series:
    """
    Approximation documentee du RSX :
    on applique un lissage exponentiel au prix avant un RSI Wilder standard,
    puis un second lissage leger sur la sortie pour obtenir un oscillateur
    plus fluide et moins bruite qu'un RSI brut.
    """
    smoothed_price = _ema(series, smoothing_window)
    rsx = _rsi(smoothed_price, window)
    return _ema(rsx, max(2, smoothing_window))


def _emotion_willingness_proxy(
    df: pd.DataFrame,
    window: int,
    atr: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    """
    Approximation documentee :
    - Emotion Index : intensite moyenne des mouvements absolus normalises par l'ATR.
    - Willingness Index : position moyenne du close dans la range roulante.
    """
    rolling_low = df["low"].rolling(window=window, min_periods=window).min()
    rolling_high = df["high"].rolling(window=window, min_periods=window).max()
    range_width = rolling_high - rolling_low

    normalized_move = _safe_divide(df["close"].diff().abs(), atr)
    emotion = 100 * normalized_move.rolling(window=window, min_periods=window).mean()

    willingness = 100 * _safe_divide(df["close"] - rolling_low, range_width)
    return emotion, willingness


def _prepare_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Le DataFrame d'entree est vide.")

    prepared = df.copy()
    prepared.columns = [str(col).strip().lower() for col in prepared.columns]

    if "date" in prepared.columns and not isinstance(prepared.index, pd.DatetimeIndex):
        prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
        prepared = prepared.dropna(subset=["date"]).set_index("date")
    elif not isinstance(prepared.index, pd.DatetimeIndex):
        try:
            prepared.index = pd.to_datetime(prepared.index, errors="raise")
        except Exception as exc:
            raise ValueError(
                "L'index doit etre de type datetime ou une colonne `date` doit etre fournie."
            ) from exc

    missing = [col for col in REQUIRED_OHLC_COLUMNS if col not in prepared.columns]
    if missing:
        raise ValueError(
            "Colonnes indispensables manquantes pour calculer les indicateurs : "
            f"{', '.join(missing)}"
        )

    prepared = prepared.sort_index()
    prepared = prepared[~prepared.index.duplicated(keep="last")]

    numeric_columns = [col for col in prepared.columns if col != "date"]
    prepared[numeric_columns] = prepared[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return prepared


def _build_base_feature_frame(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    general_cfg = config["general"]
    base = _empty_feature_frame(df.index)

    if general_cfg["include_all_original_columns"]:
        base = pd.concat([base, df], axis=1)
    else:
        keep_cols = [col for col in ("open", "high", "low", "close", "volume", "vix") if col in df.columns]
        base = pd.concat([base, df[keep_cols]], axis=1)

    mapping: Dict[str, List[str]] = {}

    if general_cfg["include_base_features"]:
        hlc_average = (df["high"] + df["low"] + df["close"]) / 3.0
        median_price = (df["high"] + df["low"]) / 2.0
        ohlc_average = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
        true_range = _true_range(df)
        simple_return_1 = df["close"].pct_change()
        log_return_1 = np.log(df["close"].where(df["close"] > 0)).diff()

        extra = pd.DataFrame(
            {
                "hlc_average": hlc_average,
                "median_price": median_price,
                "ohlc_average": ohlc_average,
                "true_range": true_range,
                "simple_return_1": simple_return_1,
                "log_return_1": log_return_1,
            },
            index=df.index,
        )
        base = pd.concat([base, extra], axis=1)
        mapping["high_low_close_average"] = ["hlc_average"]
        mapping["median_price"] = ["median_price"]
        mapping["open_high_low_close_average"] = ["ohlc_average"]

    return base, mapping


def compute_band_channel_indicators(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    out = _empty_feature_frame(df.index)
    mapping: Dict[str, List[str]] = {}

    close = df["close"]
    high = df["high"]
    low = df["low"]

    bb_cfg = config["bollinger"]
    bb_n = int(bb_cfg["window"])
    bb_k = bb_cfg["std_multiplier"]
    bb_token = _format_token(bb_k)
    bb_middle = _sma(close, bb_n)
    bb_std = close.rolling(window=bb_n, min_periods=bb_n).std(ddof=0)
    bb_upper = bb_middle + bb_k * bb_std
    bb_lower = bb_middle - bb_k * bb_std
    bb_width = _safe_divide(bb_upper - bb_lower, bb_middle)
    bb_percent_b = _safe_divide(close - bb_lower, bb_upper - bb_lower)
    bb_cols = {
        f"bb_middle_{bb_n}_{bb_token}": bb_middle,
        f"bb_upper_{bb_n}_{bb_token}": bb_upper,
        f"bb_lower_{bb_n}_{bb_token}": bb_lower,
        f"bb_bandwidth_{bb_n}_{bb_token}": bb_width,
        f"bb_percent_b_{bb_n}_{bb_token}": bb_percent_b,
    }
    out = pd.concat([out, pd.DataFrame(bb_cols, index=df.index)], axis=1)
    mapping["bollinger_bands"] = list(bb_cols.keys())

    don_cfg = config["donchian"]
    don_n = int(don_cfg["window"])
    don_upper = high.rolling(window=don_n, min_periods=don_n).max()
    don_lower = low.rolling(window=don_n, min_periods=don_n).min()
    don_middle = (don_upper + don_lower) / 2.0
    don_width = _safe_divide(don_upper - don_lower, don_middle)
    don_cols = {
        f"donchian_upper_{don_n}": don_upper,
        f"donchian_lower_{don_n}": don_lower,
        f"donchian_middle_{don_n}": don_middle,
        f"donchian_width_{don_n}": don_width,
    }
    out = pd.concat([out, pd.DataFrame(don_cols, index=df.index)], axis=1)
    mapping["donchian_channel"] = list(don_cols.keys())

    kel_cfg = config["keltner"]
    kel_ema_n = int(kel_cfg["ema_window"])
    kel_atr_n = int(kel_cfg["atr_window"])
    kel_mult = kel_cfg["multiplier"]
    kel_mult_token = _format_token(kel_mult)
    kel_middle = _ema(close, kel_ema_n)
    kel_atr = _atr(df, kel_atr_n)
    kel_upper = kel_middle + kel_mult * kel_atr
    kel_lower = kel_middle - kel_mult * kel_atr
    kel_width = _safe_divide(kel_upper - kel_lower, kel_middle)
    kel_cols = {
        f"keltner_middle_{kel_ema_n}": kel_middle,
        f"keltner_upper_{kel_ema_n}_{kel_atr_n}_{kel_mult_token}": kel_upper,
        f"keltner_lower_{kel_ema_n}_{kel_atr_n}_{kel_mult_token}": kel_lower,
        f"keltner_width_{kel_ema_n}_{kel_atr_n}_{kel_mult_token}": kel_width,
    }
    out = pd.concat([out, pd.DataFrame(kel_cols, index=df.index)], axis=1)
    mapping["keltner_channel"] = list(kel_cols.keys())

    ab_cfg = config["acceleration_bands"]
    ab_n = int(ab_cfg["window"])
    factor = _safe_divide(4.0 * (high - low), high + low)
    upper_raw = high * (1.0 + factor)
    lower_raw = low * (1.0 - factor)
    ab_upper = _sma(upper_raw, ab_n)
    ab_lower = _sma(lower_raw, ab_n)
    ab_middle = _sma(close, ab_n)
    ab_width = _safe_divide(ab_upper - ab_lower, ab_middle)
    ab_percent_b = _safe_divide(close - ab_lower, ab_upper - ab_lower)
    ab_cols = {
        f"accbands_middle_{ab_n}": ab_middle,
        f"accbands_upper_{ab_n}": ab_upper,
        f"accbands_lower_{ab_n}": ab_lower,
        f"accbands_width_{ab_n}": ab_width,
        f"accbands_percent_b_{ab_n}": ab_percent_b,
    }
    out = pd.concat([out, pd.DataFrame(ab_cols, index=df.index)], axis=1)
    mapping["acceleration_bands"] = list(ab_cols.keys())

    return out, mapping


def compute_centered_oscillators(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    out = _empty_feature_frame(df.index)
    mapping: Dict[str, List[str]] = {}

    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    median_price = (high + low) / 2.0
    has_volume = "volume" in df.columns
    volume = df["volume"] if has_volume else pd.Series(np.nan, index=df.index)

    apo_cfg = config["apo"]
    apo_fast = int(apo_cfg["fast"])
    apo_slow = int(apo_cfg["slow"])
    apo = _ema(close, apo_fast) - _ema(close, apo_slow)
    apo_col = f"apo_{apo_fast}_{apo_slow}"
    out[apo_col] = apo
    mapping["absolute_price_oscillator"] = [apo_col]

    if has_volume:
        ad_line = _chaikin_ad_line(df)
        ado_cfg = config["ado"]
        ado_fast = int(ado_cfg["fast"])
        ado_slow = int(ado_cfg["slow"])
        ado = _ema(ad_line, ado_fast) - _ema(ad_line, ado_slow)
        ado_col = f"ado_{ado_fast}_{ado_slow}"
        out[ado_col] = ado
        mapping["accumulation_distribution_oscillator"] = [ado_col]
    else:
        _warn_optional("Volume absent : Accumulation/Distribution Oscillator rempli avec NaN.")
        ado_cfg = config["ado"]
        ado_col = f"ado_{int(ado_cfg['fast'])}_{int(ado_cfg['slow'])}"
        out[ado_col] = np.nan
        mapping["accumulation_distribution_oscillator"] = [ado_col]

    ao_cfg = config["awesome"]
    ao_fast = int(ao_cfg["fast"])
    ao_slow = int(ao_cfg["slow"])
    ao = _sma(median_price, ao_fast) - _sma(median_price, ao_slow)
    ao_col = f"awesome_oscillator_{ao_fast}_{ao_slow}"
    out[ao_col] = ao
    mapping["awesome_oscillator"] = [ao_col]

    bop = _safe_divide(close - open_, high - low)
    out["bop"] = bop
    mapping["balance_of_power"] = ["bop"]

    bias_cfg = config["bias"]
    bias_n = int(bias_cfg["window"])
    bias_ma = _sma(close, bias_n)
    bias = _safe_divide(close - bias_ma, bias_ma)
    bias_col = f"bias_{bias_n}"
    out[bias_col] = bias
    mapping["bias_indicator"] = [bias_col]

    cfo_cfg = config["cfo"]
    cfo_n = int(cfo_cfg["window"])
    _, _, cfo_fitted = _rolling_linear_regression(close, cfo_n)
    cfo = 100 * _safe_divide(close - cfo_fitted, cfo_fitted)
    cfo_cols = {
        f"cfo_{cfo_n}": cfo,
        f"cfo_fitted_{cfo_n}": cfo_fitted,
    }
    out = pd.concat([out, pd.DataFrame(cfo_cols, index=df.index)], axis=1)
    mapping["chande_forecast_oscillator"] = list(cfo_cols.keys())

    cmo_cfg = config["cmo"]
    cmo_n = int(cmo_cfg["window"])
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    up_sum = up.rolling(window=cmo_n, min_periods=cmo_n).sum()
    down_sum = down.rolling(window=cmo_n, min_periods=cmo_n).sum()
    cmo = 100 * _safe_divide(up_sum - down_sum, up_sum + down_sum)
    cmo_col = f"cmo_{cmo_n}"
    out[cmo_col] = cmo
    mapping["chande_momentum_oscillator"] = [cmo_col]

    coppock_cfg = config["coppock"]
    coppock_long = int(coppock_cfg["long_window"])
    coppock_short = int(coppock_cfg["short_window"])
    coppock_wma = int(coppock_cfg["wma_window"])
    roc_long = 100 * (close / close.shift(coppock_long) - 1)
    roc_short = 100 * (close / close.shift(coppock_short) - 1)
    coppock = _wma(roc_long + roc_short, coppock_wma)
    coppock_col = f"coppock_{coppock_long}_{coppock_short}_{coppock_wma}"
    out[coppock_col] = coppock
    mapping["coppock_curve"] = [coppock_col]

    dpo_cfg = config["dpo"]
    dpo_n = int(dpo_cfg["window"])
    dpo = close.shift(int(dpo_n / 2) + 1) - _sma(close, dpo_n)
    dpo_col = f"dpo_{dpo_n}"
    out[dpo_col] = dpo
    mapping["detrended_price_oscillator"] = [dpo_col]

    eom_cfg = config["eom"]
    eom_n = int(eom_cfg["window"])
    eom_smooth = int(eom_cfg["smoothing_window"])
    if has_volume:
        distance_moved = median_price - median_price.shift(1)
        box_ratio = _safe_divide(volume, high - low)
        eom = _safe_divide(distance_moved, box_ratio)
        eom_sma = _sma(eom, eom_smooth)
    else:
        _warn_optional("Volume absent : Ease of Movement rempli avec NaN.")
        eom = pd.Series(np.nan, index=df.index)
        eom_sma = pd.Series(np.nan, index=df.index)
    eom_cols = {
        f"eom_{eom_n}": eom,
        f"eom_sma_{eom_smooth}": eom_sma,
    }
    out = pd.concat([out, pd.DataFrame(eom_cols, index=df.index)], axis=1)
    mapping["ease_of_movement"] = list(eom_cols.keys())

    efi_cfg = config["efi"]
    efi_n = int(efi_cfg["window"])
    if has_volume:
        efi = close.diff() * volume
        efi_ema = _ema(efi, efi_n)
    else:
        _warn_optional("Volume absent : Elder's Force Index rempli avec NaN.")
        efi = pd.Series(np.nan, index=df.index)
        efi_ema = pd.Series(np.nan, index=df.index)
    efi_cols = {
        "efi_raw": efi,
        f"efi_ema_{efi_n}": efi_ema,
    }
    out = pd.concat([out, pd.DataFrame(efi_cols, index=df.index)], axis=1)
    mapping["elders_force_index"] = list(efi_cols.keys())

    kst_cfg = config["kst"]
    roc_windows = [int(value) for value in kst_cfg["roc_windows"]]
    sma_windows = [int(value) for value in kst_cfg["sma_windows"]]
    weights = [float(value) for value in kst_cfg["weights"]]
    signal_n = int(kst_cfg["signal_window"])
    rocs = [100 * (close / close.shift(n) - 1) for n in roc_windows]
    smoothed_rocs = [_sma(roc, n) for roc, n in zip(rocs, sma_windows)]
    kst = sum(weight * roc for weight, roc in zip(weights, smoothed_rocs))
    kst_signal = _sma(kst, signal_n)
    kst_suffix = "_".join(str(value) for value in roc_windows)
    kst_cols = {
        f"kst_{kst_suffix}": kst,
        f"kst_signal_{kst_suffix}_{signal_n}": kst_signal,
    }
    out = pd.concat([out, pd.DataFrame(kst_cols, index=df.index)], axis=1)
    mapping["know_sure_thing"] = list(kst_cols.keys())

    macd_cfg = config["macd"]
    macd_fast = int(macd_cfg["fast"])
    macd_slow = int(macd_cfg["slow"])
    macd_signal_n = int(macd_cfg["signal_window"])
    macd_line = _ema(close, macd_fast) - _ema(close, macd_slow)
    macd_signal = _ema(macd_line, macd_signal_n)
    macd_hist = macd_line - macd_signal
    macd_cols = {
        f"macd_line_{macd_fast}_{macd_slow}": macd_line,
        f"macd_signal_{macd_fast}_{macd_slow}_{macd_signal_n}": macd_signal,
        f"macd_hist_{macd_fast}_{macd_slow}_{macd_signal_n}": macd_hist,
    }
    out = pd.concat([out, pd.DataFrame(macd_cols, index=df.index)], axis=1)
    mapping["macd"] = list(macd_cols.keys())

    momentum_cfg = config["momentum"]
    momentum_n = int(momentum_cfg["window"])
    momentum = close - close.shift(momentum_n)
    momentum_col = f"momentum_{momentum_n}"
    out[momentum_col] = momentum
    mapping["momentum_indicator"] = [momentum_col]

    ppo_cfg = config["ppo"]
    ppo_fast = int(ppo_cfg["fast"])
    ppo_slow = int(ppo_cfg["slow"])
    ppo_signal_n = int(ppo_cfg["signal_window"])
    ppo_den = _ema(close, ppo_slow)
    ppo = 100 * _safe_divide(_ema(close, ppo_fast) - ppo_den, ppo_den)
    ppo_signal = _ema(ppo, ppo_signal_n)
    ppo_hist = ppo - ppo_signal
    ppo_cols = {
        f"ppo_{ppo_fast}_{ppo_slow}": ppo,
        f"ppo_signal_{ppo_fast}_{ppo_slow}_{ppo_signal_n}": ppo_signal,
        f"ppo_hist_{ppo_fast}_{ppo_slow}_{ppo_signal_n}": ppo_hist,
    }
    out = pd.concat([out, pd.DataFrame(ppo_cols, index=df.index)], axis=1)
    mapping["percentage_price_oscillator"] = list(ppo_cols.keys())

    pvo_cfg = config["pvo"]
    pvo_fast = int(pvo_cfg["fast"])
    pvo_slow = int(pvo_cfg["slow"])
    pvo_signal_n = int(pvo_cfg["signal_window"])
    if has_volume:
        pvo_den = _ema(volume, pvo_slow)
        pvo = 100 * _safe_divide(_ema(volume, pvo_fast) - pvo_den, pvo_den)
        pvo_signal = _ema(pvo, pvo_signal_n)
        pvo_hist = pvo - pvo_signal
    else:
        _warn_optional("Volume absent : Percentage Volume Oscillator rempli avec NaN.")
        pvo = pd.Series(np.nan, index=df.index)
        pvo_signal = pd.Series(np.nan, index=df.index)
        pvo_hist = pd.Series(np.nan, index=df.index)
    pvo_cols = {
        f"pvo_{pvo_fast}_{pvo_slow}": pvo,
        f"pvo_signal_{pvo_fast}_{pvo_slow}_{pvo_signal_n}": pvo_signal,
        f"pvo_hist_{pvo_fast}_{pvo_slow}_{pvo_signal_n}": pvo_hist,
    }
    out = pd.concat([out, pd.DataFrame(pvo_cols, index=df.index)], axis=1)
    mapping["percentage_volume_oscillator"] = list(pvo_cols.keys())

    qstick_cfg = config["qstick"]
    qstick_n = int(qstick_cfg["window"])
    qstick = _sma(close - open_, qstick_n)
    qstick_col = f"qstick_{qstick_n}"
    out[qstick_col] = qstick
    mapping["qstick_indicator"] = [qstick_col]

    roc_cfg = config["roc"]
    roc_n = int(roc_cfg["window"])
    roc = 100 * (close / close.shift(roc_n) - 1)
    roc_col = f"roc_{roc_n}"
    out[roc_col] = roc
    mapping["rate_of_change"] = [roc_col]

    rvi_cfg = config["rvi"]
    rvi_n = int(rvi_cfg["window"])
    rvi_signal_n = int(rvi_cfg["signal_window"])
    rvi_num = _sma(close - open_, rvi_n)
    rvi_den = _sma(high - low, rvi_n)
    rvi = _safe_divide(rvi_num, rvi_den)
    rvi_signal = _sma(rvi, rvi_signal_n)
    rvi_cols = {
        f"rvi_{rvi_n}": rvi,
        f"rvi_signal_{rvi_n}_{rvi_signal_n}": rvi_signal,
    }
    out = pd.concat([out, pd.DataFrame(rvi_cols, index=df.index)], axis=1)
    mapping["relative_vigor_index"] = list(rvi_cols.keys())

    return out, mapping


def compute_context_filters(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    out = _empty_feature_frame(df.index)
    mapping: Dict[str, List[str]] = {}

    close = df["close"]
    high = df["high"]
    low = df["low"]

    atr_cfg = config["atr"]
    atr_n = int(atr_cfg["window"])
    atr = _atr(df, atr_n)
    atr_col = f"atr_{atr_n}"
    out[atr_col] = atr
    mapping["atr"] = [atr_col]

    natr = _safe_divide(atr, close)
    natr_col = f"natr_{atr_n}"
    out[natr_col] = natr
    mapping["normalized_atr"] = [natr_col]

    adx_cfg = config["adx"]
    adx_n = int(adx_cfg["window"])
    adx, plus_di, minus_di, dx = _adx(df, adx_n)
    adx_cols = {
        f"adx_{adx_n}": adx,
        f"plus_di_{adx_n}": plus_di,
        f"minus_di_{adx_n}": minus_di,
        f"dx_{adx_n}": dx,
    }
    out = pd.concat([out, pd.DataFrame(adx_cols, index=df.index)], axis=1)
    mapping["adx"] = list(adx_cols.keys())

    chop_cfg = config["choppiness"]
    chop_n = int(chop_cfg["window"])
    tr_sum = _true_range(df).rolling(window=chop_n, min_periods=chop_n).sum()
    range_n = high.rolling(window=chop_n, min_periods=chop_n).max() - low.rolling(
        window=chop_n, min_periods=chop_n
    ).min()
    choppiness = 100 * np.log10(_safe_divide(tr_sum, range_n)) / math.log10(chop_n)
    chop_col = f"choppiness_{chop_n}"
    out[chop_col] = choppiness
    mapping["choppiness_index"] = [chop_col]

    entropy_cfg = config["entropy"]
    entropy_n = int(entropy_cfg["window"])
    log_returns = np.log(close.where(close > 0)).diff()
    entropy = _rolling_entropy(
        log_returns,
        window=entropy_n,
        mode=str(entropy_cfg.get("mode", "sign")),
        bins=int(entropy_cfg.get("bins", 10)),
    )
    entropy_col = f"entropy_{entropy_n}"
    out[entropy_col] = entropy
    mapping["entropy_indicator"] = [entropy_col]

    kurt_cfg = config["kurtosis"]
    kurt_n = int(kurt_cfg["window"])
    kurtosis = close.pct_change().rolling(window=kurt_n, min_periods=kurt_n).kurt()
    kurt_col = f"kurtosis_{kurt_n}"
    out[kurt_col] = kurtosis
    mapping["kurtosis"] = [kurt_col]

    mass_cfg = config["mass_index"]
    mass_ema_n = int(mass_cfg["ema_window"])
    mass_sum_n = int(mass_cfg["sum_window"])
    range_series = high - low
    ema1 = _ema(range_series, mass_ema_n)
    ema2 = _ema(ema1, mass_ema_n)
    mass_ratio = _safe_divide(ema1, ema2)
    mass_index = mass_ratio.rolling(window=mass_sum_n, min_periods=mass_sum_n).sum()
    mass_col = f"mass_index_{mass_ema_n}_{mass_sum_n}"
    out[mass_col] = mass_index
    mapping["mass_index"] = [mass_col]

    mad_cfg = config["mad"]
    mad_n = int(mad_cfg["window"])
    mad_source = str(mad_cfg.get("source", "close")).lower()
    if mad_source == "typical_price":
        mad_input = (high + low + close) / 3.0
    elif mad_source == "median_price":
        mad_input = (high + low) / 2.0
    else:
        mad_input = close
        mad_source = "close"
    mad = _rolling_mad(mad_input, mad_n)
    mad_col = f"mad_{mad_source}_{mad_n}"
    out[mad_col] = mad
    mapping["mean_absolute_deviation"] = [mad_col]

    midpoint_cfg = config["midpoint"]
    midpoint_n = int(midpoint_cfg["window"])
    midpoint = (
        close.rolling(window=midpoint_n, min_periods=midpoint_n).max()
        + close.rolling(window=midpoint_n, min_periods=midpoint_n).min()
    ) / 2.0
    midpoint_col = f"midpoint_{midpoint_n}"
    out[midpoint_col] = midpoint
    mapping["midpoint_indicator"] = [midpoint_col]

    midprice_cfg = config["midprice"]
    midprice_n = int(midprice_cfg["window"])
    midprice = (
        high.rolling(window=midprice_n, min_periods=midprice_n).max()
        + low.rolling(window=midprice_n, min_periods=midprice_n).min()
    ) / 2.0
    midprice_col = f"midprice_{midprice_n}"
    out[midprice_col] = midprice
    mapping["midprice_indicator"] = [midprice_col]

    price_distance_cfg = config["price_distance"]
    distance_n = int(price_distance_cfg["window"])
    reference_kind = str(price_distance_cfg.get("reference", "sma")).lower()
    if reference_kind == "ema":
        reference = _ema(close, distance_n)
    else:
        reference_kind = "sma"
        reference = _sma(close, distance_n)
    distance = _safe_divide(close - reference, reference)
    distance_cols = {
        f"price_distance_{reference_kind}_{distance_n}": distance,
        f"price_distance_ref_{reference_kind}_{distance_n}": reference,
    }
    out = pd.concat([out, pd.DataFrame(distance_cols, index=df.index)], axis=1)
    mapping["price_distance"] = list(distance_cols.keys())

    quantile_cfg = config["quantile"]
    quantile_n = int(quantile_cfg["window"])
    quantile_rank = _rolling_percent_rank(close, quantile_n)
    quantile_col = f"quantile_rank_{quantile_n}"
    out[quantile_col] = quantile_rank
    mapping["quantile_indicator"] = [quantile_col]

    vix_cfg = config["vix"]
    vix_roc_n = int(vix_cfg["roc_window"])
    vix_z_n = int(vix_cfg["zscore_window"])
    vix_sma_n = int(vix_cfg["sma_window"])
    if "vix" in df.columns:
        vix = df["vix"]
        vix_roc = 100 * (vix / vix.shift(vix_roc_n) - 1)
        vix_sma = _sma(vix, vix_sma_n)
        vix_std = vix.rolling(window=vix_z_n, min_periods=vix_z_n).std(ddof=0)
        vix_zscore = _safe_divide(vix - _sma(vix, vix_z_n), vix_std)
    else:
        _warn_optional("Colonne `vix` absente : les features VIX sont remplies avec NaN.")
        vix = pd.Series(np.nan, index=df.index)
        vix_roc = pd.Series(np.nan, index=df.index)
        vix_sma = pd.Series(np.nan, index=df.index)
        vix_zscore = pd.Series(np.nan, index=df.index)
    vix_cols = {
        "vix": vix,
        f"vix_roc_{vix_roc_n}": vix_roc,
        f"vix_sma_{vix_sma_n}": vix_sma,
        f"vix_zscore_{vix_z_n}": vix_zscore,
    }
    out = pd.concat([out, pd.DataFrame(vix_cols, index=df.index)], axis=1)
    mapping["vix"] = list(vix_cols.keys())

    return out, mapping


def compute_extreme_zone_oscillators(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    out = _empty_feature_frame(df.index)
    mapping: Dict[str, List[str]] = {}

    close = df["close"]
    high = df["high"]
    low = df["low"]
    typical_price = (high + low + close) / 3.0
    has_volume = "volume" in df.columns
    volume = df["volume"] if has_volume else pd.Series(np.nan, index=df.index)

    cog_cfg = config["center_of_gravity"]
    cog_n = int(cog_cfg["window"])
    cog = _center_of_gravity(close, cog_n)
    cog_col = f"cog_{cog_n}"
    out[cog_col] = cog
    mapping["center_of_gravity"] = [cog_col]

    cci_cfg = config["cci"]
    cci_n = int(cci_cfg["window"])
    tp_sma = _sma(typical_price, cci_n)
    tp_mad = _rolling_mad(typical_price, cci_n)
    cci = _safe_divide(typical_price - tp_sma, 0.015 * tp_mad)
    cci_col = f"cci_{cci_n}"
    out[cci_col] = cci
    mapping["commodity_channel_index"] = [cci_col]

    atr_proxy = _atr(df, int(config["atr"]["window"]))
    emotion_cfg = config["emotion_willingness"]
    emotion_n = int(emotion_cfg["window"])
    emotion, willingness = _emotion_willingness_proxy(df, emotion_n, atr_proxy)
    emotion_cols = {
        f"emotion_index_{emotion_n}": emotion,
        f"willingness_index_{emotion_n}": willingness,
    }
    out = pd.concat([out, pd.DataFrame(emotion_cols, index=df.index)], axis=1)
    mapping["emotion_index_willingness_index"] = list(emotion_cols.keys())

    mfi_cfg = config["mfi"]
    mfi_n = int(mfi_cfg["window"])
    if has_volume:
        money_flow = typical_price * volume
        tp_delta = typical_price.diff()
        positive_flow = money_flow.where(tp_delta > 0, 0.0)
        negative_flow = money_flow.where(tp_delta < 0, 0.0).abs()
        positive_sum = positive_flow.rolling(window=mfi_n, min_periods=mfi_n).sum()
        negative_sum = negative_flow.rolling(window=mfi_n, min_periods=mfi_n).sum()
        money_ratio = _safe_divide(positive_sum, negative_sum)
        mfi = 100 - (100 / (1 + money_ratio))
    else:
        _warn_optional("Volume absent : Money Flow Index rempli avec NaN.")
        mfi = pd.Series(np.nan, index=df.index)
    mfi_col = f"mfi_{mfi_n}"
    out[mfi_col] = mfi
    mapping["money_flow_index"] = [mfi_col]

    pgo_cfg = config["pgo"]
    pgo_n = int(pgo_cfg["window"])
    pgo = _safe_divide(close - _sma(close, pgo_n), _atr(df, pgo_n))
    pgo_col = f"pgo_{pgo_n}"
    out[pgo_col] = pgo
    mapping["pretty_good_oscillator"] = [pgo_col]

    rsi_cfg = config["rsi"]
    rsi_n = int(rsi_cfg["window"])
    rsi = _rsi(close, rsi_n)
    rsi_col = f"rsi_{rsi_n}"
    out[rsi_col] = rsi
    mapping["rsi"] = [rsi_col]

    rsx_cfg = config["rsx"]
    rsx_n = int(rsx_cfg["window"])
    rsx_smooth_n = int(rsx_cfg["smoothing_window"])
    rsx = _rsx_approximation(close, rsx_n, rsx_smooth_n)
    rsx_col = f"rsx_{rsx_n}"
    out[rsx_col] = rsx
    mapping["rsx"] = [rsx_col]

    stoch_cfg = config["stochastic"]
    stoch_n = int(stoch_cfg["window"])
    stoch_d_n = int(stoch_cfg["signal_window"])
    lowest_low = low.rolling(window=stoch_n, min_periods=stoch_n).min()
    highest_high = high.rolling(window=stoch_n, min_periods=stoch_n).max()
    stoch_k = 100 * _safe_divide(close - lowest_low, highest_high - lowest_low)
    stoch_d = _sma(stoch_k, stoch_d_n)
    stoch_cols = {
        f"stochastic_k_{stoch_n}_{stoch_d_n}": stoch_k,
        f"stochastic_d_{stoch_n}_{stoch_d_n}": stoch_d,
    }
    out = pd.concat([out, pd.DataFrame(stoch_cols, index=df.index)], axis=1)
    mapping["stochastic_oscillator"] = list(stoch_cols.keys())

    return out, mapping


def compute_trend_filters(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    out = _empty_feature_frame(df.index)
    mapping: Dict[str, List[str]] = {}

    close = df["close"]
    high = df["high"]
    low = df["low"]

    ma_cfg = config["ma"]
    ma_columns: List[str] = []
    sma_windows = [int(value) for value in ma_cfg["sma_windows"]]
    ema_windows = [int(value) for value in ma_cfg["ema_windows"]]
    for window in sma_windows:
        col = f"sma_{window}"
        out[col] = _sma(close, window)
        ma_columns.append(col)
    for window in ema_windows:
        col = f"ema_{window}"
        out[col] = _ema(close, window)
        ma_columns.append(col)
    for first, second in zip(sma_windows[:-1], sma_windows[1:]):
        col = f"sma_spread_{first}_{second}"
        out[col] = _safe_divide(out[f"sma_{first}"] - out[f"sma_{second}"], out[f"sma_{second}"])
        ma_columns.append(col)
    for first, second in zip(ema_windows[:-1], ema_windows[1:]):
        col = f"ema_spread_{first}_{second}"
        out[col] = _safe_divide(out[f"ema_{first}"] - out[f"ema_{second}"], out[f"ema_{second}"])
        ma_columns.append(col)
    mapping["moving_averages"] = ma_columns

    alma_cfg = config["alma"]
    alma_n = int(alma_cfg["window"])
    alma_offset = float(alma_cfg["offset"])
    alma_sigma = float(alma_cfg["sigma"])
    alma_col = f"alma_{alma_n}_{_format_token(alma_offset)}_{_format_token(alma_sigma)}"
    out[alma_col] = _alma(close, alma_n, alma_offset, alma_sigma)
    mapping["alma"] = [alma_col]

    aroon_cfg = config["aroon"]
    aroon_n = int(aroon_cfg["window"])
    aroon_up = high.rolling(window=aroon_n, min_periods=aroon_n).apply(
        lambda values: float((aroon_n - 1 - (aroon_n - 1 - np.argmax(values))) / aroon_n * 100),
        raw=True,
    )
    aroon_down = low.rolling(window=aroon_n, min_periods=aroon_n).apply(
        lambda values: float((aroon_n - 1 - (aroon_n - 1 - np.argmin(values))) / aroon_n * 100),
        raw=True,
    )
    aroon_osc = aroon_up - aroon_down
    aroon_cols = {
        f"aroon_up_{aroon_n}": aroon_up,
        f"aroon_down_{aroon_n}": aroon_down,
        f"aroon_oscillator_{aroon_n}": aroon_osc,
    }
    out = pd.concat([out, pd.DataFrame(aroon_cols, index=df.index)], axis=1)
    mapping["aroon_indicator"] = list(aroon_cols.keys())

    cti_cfg = config["cti"]
    cti_n = int(cti_cfg["window"])
    cti = _rolling_corr_with_time(close, cti_n)
    cti_col = f"cti_{cti_n}"
    out[cti_col] = cti
    mapping["correlation_trend_indicator"] = [cti_col]

    er_cfg = config["efficiency_ratio"]
    er_n = int(er_cfg["window"])
    er_change = (close - close.shift(er_n)).abs()
    er_volatility = close.diff().abs().rolling(window=er_n, min_periods=er_n).sum()
    efficiency_ratio = _safe_divide(er_change, er_volatility)
    er_col = f"efficiency_ratio_{er_n}"
    out[er_col] = efficiency_ratio
    mapping["efficiency_ratio"] = [er_col]

    ss_cfg = config["super_smoother"]
    ss_period = int(ss_cfg["period"])
    ss_col = f"super_smoother_{ss_period}"
    out[ss_col] = _super_smoother(close, ss_period)
    mapping["ehlers_super_smoother_filter"] = [ss_col]

    ebsw_cfg = config["even_better_sinewave"]
    ebsw_period = int(ebsw_cfg["period"])
    ebsw_signal_n = int(ebsw_cfg["signal_window"])
    ebsw, ebsw_signal = _even_better_sinewave(close, ebsw_period, ebsw_signal_n)
    ebsw_cols = {
        f"even_better_sinewave_{ebsw_period}": ebsw,
        f"even_better_sinewave_signal_{ebsw_period}_{ebsw_signal_n}": ebsw_signal,
    }
    out = pd.concat([out, pd.DataFrame(ebsw_cols, index=df.index)], axis=1)
    mapping["even_better_sinewave"] = list(ebsw_cols.keys())

    fib_cfg = config["fibonacci_wma"]
    fib_n = int(fib_cfg["window"])
    fib_col = f"fib_wma_{fib_n}"
    out[fib_col] = _wma(close, fib_n, weights=_fibonacci_weights(fib_n))
    mapping["fibonacci_weighted_moving_average"] = [fib_col]

    ich_cfg = config["ichimoku"]
    tenkan_n = int(ich_cfg["tenkan_window"])
    kijun_n = int(ich_cfg["kijun_window"])
    span_b_n = int(ich_cfg["span_b_window"])
    tenkan = (
        high.rolling(window=tenkan_n, min_periods=tenkan_n).max()
        + low.rolling(window=tenkan_n, min_periods=tenkan_n).min()
    ) / 2.0
    kijun = (
        high.rolling(window=kijun_n, min_periods=kijun_n).max()
        + low.rolling(window=kijun_n, min_periods=kijun_n).min()
    ) / 2.0
    span_a = (tenkan + kijun) / 2.0
    span_b = (
        high.rolling(window=span_b_n, min_periods=span_b_n).max()
        + low.rolling(window=span_b_n, min_periods=span_b_n).min()
    ) / 2.0
    ich_cols = {
        f"ichimoku_tenkan_{tenkan_n}": tenkan,
        f"ichimoku_kijun_{kijun_n}": kijun,
        f"ichimoku_span_a_{tenkan_n}_{kijun_n}": span_a,
        f"ichimoku_span_b_{span_b_n}": span_b,
        f"ichimoku_cloud_delta_{tenkan_n}_{kijun_n}_{span_b_n}": span_a - span_b,
    }
    out = pd.concat([out, pd.DataFrame(ich_cols, index=df.index)], axis=1)
    mapping["ichimoku"] = list(ich_cols.keys())

    inertia_cfg = config["inertia"]
    inertia_n = int(inertia_cfg["window"])
    slope, _, fitted = _rolling_linear_regression(close, inertia_n)
    cti_inertia = _rolling_corr_with_time(close, inertia_n)
    close_mean = close.rolling(window=inertia_n, min_periods=inertia_n).mean()
    normalized_slope = _safe_divide(slope, close_mean)
    inertia = 100 * normalized_slope * (cti_inertia ** 2)
    inertia_cols = {
        f"inertia_{inertia_n}": inertia,
        f"inertia_fitted_{inertia_n}": fitted,
    }
    out = pd.concat([out, pd.DataFrame(inertia_cols, index=df.index)], axis=1)
    mapping["inertia_indicator"] = list(inertia_cols.keys())

    kama_cfg = config["kama"]
    kama_er_n = int(kama_cfg["er_window"])
    kama_fast = int(kama_cfg["fast"])
    kama_slow = int(kama_cfg["slow"])
    kama_col = f"kama_{kama_er_n}_{kama_fast}_{kama_slow}"
    out[kama_col] = _kama(close, kama_er_n, kama_fast, kama_slow)
    mapping["kaufman_adaptive_moving_average"] = [kama_col]

    decay_cfg = config["linear_decay"]
    decay_n = int(decay_cfg["window"])
    decay_col = f"linear_decay_{decay_n}"
    out[decay_col] = _wma(close, decay_n, weights=np.arange(1, decay_n + 1, dtype=float))
    mapping["linear_decay"] = [decay_col]

    linreg_cfg = config["linear_regression"]
    linreg_n = int(linreg_cfg["window"])
    slope, intercept, fitted = _rolling_linear_regression(close, linreg_n)
    linreg_cols = {
        f"linreg_slope_{linreg_n}": slope,
        f"linreg_intercept_{linreg_n}": intercept,
        f"linreg_fitted_{linreg_n}": fitted,
    }
    out = pd.concat([out, pd.DataFrame(linreg_cols, index=df.index)], axis=1)
    mapping["linear_regression"] = list(linreg_cols.keys())

    psar_cfg = config["psar"]
    psar_step = float(psar_cfg["step"])
    psar_max = float(psar_cfg["max_step"])
    psar_value, psar_trend, psar_af = _psar(df, psar_step, psar_max)
    psar_token_step = _format_token(psar_step)
    psar_token_max = _format_token(psar_max)
    psar_cols = {
        f"psar_{psar_token_step}_{psar_token_max}": psar_value,
        f"psar_trend_{psar_token_step}_{psar_token_max}": psar_trend,
        f"psar_af_{psar_token_step}_{psar_token_max}": psar_af,
        f"psar_distance_{psar_token_step}_{psar_token_max}": _safe_divide(close - psar_value, close),
    }
    out = pd.concat([out, pd.DataFrame(psar_cols, index=df.index)], axis=1)
    mapping["parabolic_sar"] = list(psar_cols.keys())

    return out, mapping


def compute_volume_confirmation_indicators(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    out = _empty_feature_frame(df.index)
    mapping: Dict[str, List[str]] = {}

    has_volume = "volume" in df.columns
    close = df["close"]

    cmf_n = int(config["cmf"]["window"])
    elder_ray_n = int(config["elder_ray"]["window"])
    pvr_n = int(config["price_volume_rank"]["window"])
    nvi_start = float(config["nvi"]["start_value"])
    pvi_start = float(config["pvi"]["start_value"])

    ad_line_col = "ad_line"
    cmf_col = f"cmf_{cmf_n}"
    bull_col = f"elder_ray_bull_{elder_ray_n}"
    bear_col = f"elder_ray_bear_{elder_ray_n}"
    obv_col = "obv"
    pvt_col = "pvt"
    nvi_col = "nvi"
    pvi_col = "pvi"
    pvr_col = f"price_volume_rank_{pvr_n}"

    if not has_volume:
        _warn_optional("Volume absent : les indicateurs de confirmation volume sont remplis avec NaN.")
        nan_cols = [
            ad_line_col,
            cmf_col,
            bull_col,
            bear_col,
            obv_col,
            pvt_col,
            nvi_col,
            pvi_col,
            pvr_col,
        ]
        out = _nan_frame(df.index, nan_cols)
        mapping["accumulation_distribution_line"] = [ad_line_col]
        mapping["chaikin_money_flow"] = [cmf_col]
        mapping["elder_ray_index"] = [bull_col, bear_col]
        mapping["on_balance_volume"] = [obv_col]
        mapping["price_volume_trend"] = [pvt_col]
        mapping["negative_volume_index"] = [nvi_col]
        mapping["positive_volume_index"] = [pvi_col]
        mapping["price_volume_rank"] = [pvr_col]
        return out, mapping

    volume = df["volume"]
    money_flow_multiplier, money_flow_volume = _money_flow_components(df)
    ad_line = money_flow_volume.cumsum()
    out[ad_line_col] = ad_line
    mapping["accumulation_distribution_line"] = [ad_line_col]

    cmf = _safe_divide(
        money_flow_volume.rolling(window=cmf_n, min_periods=cmf_n).sum(),
        volume.rolling(window=cmf_n, min_periods=cmf_n).sum(),
    )
    out[cmf_col] = cmf
    mapping["chaikin_money_flow"] = [cmf_col]

    ema_close = _ema(close, elder_ray_n)
    out[bull_col] = df["high"] - ema_close
    out[bear_col] = df["low"] - ema_close
    mapping["elder_ray_index"] = [bull_col, bear_col]

    close_diff = close.diff()
    signed_volume = np.sign(close_diff).fillna(0.0) * volume
    out[obv_col] = signed_volume.cumsum()
    mapping["on_balance_volume"] = [obv_col]

    pvt = (volume * _safe_divide(close_diff, close.shift(1))).fillna(0.0).cumsum()
    out[pvt_col] = pvt
    mapping["price_volume_trend"] = [pvt_col]

    out[nvi_col] = _nvi_or_pvi(close, volume, nvi_start, positive=False)
    mapping["negative_volume_index"] = [nvi_col]

    out[pvi_col] = _nvi_or_pvi(close, volume, pvi_start, positive=True)
    mapping["positive_volume_index"] = [pvi_col]

    return_rank = _rolling_percent_rank(close.pct_change(), pvr_n)
    volume_rank = _rolling_percent_rank(volume, pvr_n)
    out[pvr_col] = (2 * return_rank - 1) * volume_rank
    mapping["price_volume_rank"] = [pvr_col]

    return out, mapping


def _merge_indicator_maps(
    target: Dict[str, List[str]],
    source: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    merged = dict(target)
    merged.update(source)
    return merged


def _build_family_column_map(indicator_column_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
    family_map: Dict[str, List[str]] = {}
    for indicator_key, columns in indicator_column_map.items():
        if indicator_key not in INDICATOR_REGISTRY:
            continue
        family = INDICATOR_REGISTRY[indicator_key]["family"]
        family_map.setdefault(family, [])
        family_map[family].extend(columns)

    for family, columns in family_map.items():
        family_map[family] = sorted(dict.fromkeys(columns))
    return family_map


def summarize_feature_output(features: pd.DataFrame) -> Dict[str, Any]:
    """
    Construit un petit resume structurant le nombre de colonnes par famille.
    """
    family_map = features.attrs.get("family_column_map", {})
    indicator_map = features.attrs.get("indicator_column_map", {})
    summary = {
        "num_rows": int(features.shape[0]),
        "num_columns": int(features.shape[1]),
        "num_indicators": len(indicator_map),
        "family_feature_counts": {family: len(columns) for family, columns in family_map.items()},
    }
    return summary


def print_feature_summary(features: pd.DataFrame) -> None:
    """
    Affiche un resume compact du nombre de features generees.
    """
    summary = summarize_feature_output(features)
    print(
        f"Features generees : {summary['num_columns']} colonnes "
        f"sur {summary['num_rows']} lignes pour {summary['num_indicators']} indicateurs."
    )
    for family, count in summary["family_feature_counts"].items():
        print(f"- {family}: {count} colonnes")


def smoke_test_indicators(df: pd.DataFrame, config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """
    Lance un test rapide de bout en bout sur un DataFrame OHLCV.
    """
    features = compute_all_indicators(df, config=config)
    summary = summarize_feature_output(features)
    summary["added_columns"] = features.attrs.get("added_feature_columns", [])
    return summary


def _compute_all_indicators_single_df(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    prepared = _prepare_input_dataframe(df)

    base_features, indicator_column_map = _build_base_feature_frame(prepared, cfg)

    band_df, band_map = compute_band_channel_indicators(prepared, cfg)
    centered_df, centered_map = compute_centered_oscillators(prepared, cfg)
    context_df, context_map = compute_context_filters(prepared, cfg)
    extreme_df, extreme_map = compute_extreme_zone_oscillators(prepared, cfg)
    trend_df, trend_map = compute_trend_filters(prepared, cfg)
    volume_df, volume_map = compute_volume_confirmation_indicators(prepared, cfg)

    indicator_column_map = _merge_indicator_maps(indicator_column_map, band_map)
    indicator_column_map = _merge_indicator_maps(indicator_column_map, centered_map)
    indicator_column_map = _merge_indicator_maps(indicator_column_map, context_map)
    indicator_column_map = _merge_indicator_maps(indicator_column_map, extreme_map)
    indicator_column_map = _merge_indicator_maps(indicator_column_map, trend_map)
    indicator_column_map = _merge_indicator_maps(indicator_column_map, volume_map)

    features = pd.concat(
        [base_features, band_df, centered_df, context_df, extreme_df, trend_df, volume_df],
        axis=1,
    )
    features = features.loc[:, ~features.columns.duplicated()]
    features = features.sort_index()

    family_column_map = _build_family_column_map(indicator_column_map)
    original_columns = list(base_features.columns)
    added_columns = [col for col in features.columns if col not in original_columns]

    features.attrs["indicator_registry"] = get_indicator_registry()
    features.attrs["indicator_column_map"] = indicator_column_map
    features.attrs["family_column_map"] = family_column_map
    features.attrs["config"] = cfg
    features.attrs["added_feature_columns"] = added_columns
    return features


def _flatten_panel_input(
    data: Any,
    prefix: Tuple[Any, ...] = (),
) -> Dict[Tuple[Any, ...], pd.DataFrame]:
    if isinstance(data, pd.DataFrame):
        return {prefix: data}

    if isinstance(data, ABCMapping):
        flattened: Dict[Tuple[Any, ...], pd.DataFrame] = {}
        for key, value in data.items():
            flattened.update(_flatten_panel_input(value, prefix + (key,)))
        return flattened

    raise TypeError(
        "compute_all_indicators attend un DataFrame OHLCV ou un dictionnaire "
        "imbrique de DataFrames. Type recu : "
        f"{type(data).__name__}"
    )


def _infer_panel_level_names(keys: Sequence[Tuple[Any, ...]]) -> List[str]:
    if not keys:
        return ["asset"]

    depth = len(keys[0])
    if depth <= 1:
        return ["asset"]
    if depth == 2:
        return ["timeframe", "asset"]
    return [f"group_{i}" for i in range(depth - 1)] + ["asset"]


def _compute_all_indicators_panel(
    panel: ABCMapping,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    flattened = _flatten_panel_input(panel)
    if not flattened:
        raise ValueError("Le panel d'entree est vide.")

    computed: Dict[Tuple[Any, ...], pd.DataFrame] = {}
    sample_features: Optional[pd.DataFrame] = None

    for key, frame in flattened.items():
        features = _compute_all_indicators_single_df(frame, cfg)
        computed[key] = features
        if sample_features is None:
            sample_features = features

    level_names = _infer_panel_level_names(list(computed.keys()))
    panel_features = pd.concat(computed, names=level_names)
    panel_features = panel_features.sort_index()

    if sample_features is not None:
        panel_features.attrs["indicator_registry"] = sample_features.attrs.get("indicator_registry", {})
        panel_features.attrs["indicator_column_map"] = sample_features.attrs.get("indicator_column_map", {})
        panel_features.attrs["family_column_map"] = sample_features.attrs.get("family_column_map", {})
        panel_features.attrs["config"] = cfg
        panel_features.attrs["added_feature_columns"] = sample_features.attrs.get("added_feature_columns", [])
        panel_features.attrs["panel_keys"] = list(computed.keys())

    return panel_features


def compute_all_indicators(
    df: pd.DataFrame | Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """
    Calcule l'ensemble des indicateurs techniques de maniere causale.

    Parameters
    ----------
    df : pd.DataFrame | Mapping[str, Any]
        Soit un DataFrame OHLCV indexe par datetime ou contenant une colonne `date`,
        soit un dictionnaire de DataFrames, par exemple `{asset: dataframe}` ou
        `{timeframe: {asset: dataframe}}`.
    config : Optional[Mapping[str, Any]]
        Surcharge optionnelle de la configuration par defaut.

    Returns
    -------
    pd.DataFrame
        DataFrame final trie par datetime, sans fuite d'information,
        contenant les colonnes originales utiles et toutes les features.
        Si l'entree est un dictionnaire, le retour est un DataFrame concatene
        avec MultiIndex (`asset`, `date`) ou (`timeframe`, `asset`, `date`).
    """
    cfg = _resolve_config(config)

    if isinstance(df, pd.DataFrame):
        features = _compute_all_indicators_single_df(df, cfg)
    elif isinstance(df, ABCMapping):
        features = _compute_all_indicators_panel(df, cfg)
    else:
        raise TypeError(
            "compute_all_indicators attend un DataFrame pandas ou un dictionnaire "
            f"de DataFrames. Type recu : {type(df).__name__}"
        )

    if cfg["general"].get("print_summary", False):
        print_feature_summary(features)

    return features


__all__ = [
    "compute_all_indicators",
    "compute_band_channel_indicators",
    "compute_centered_oscillators",
    "compute_context_filters",
    "compute_extreme_zone_oscillators",
    "compute_trend_filters",
    "compute_volume_confirmation_indicators",
    "get_default_indicator_config",
    "get_indicator_registry",
    "print_feature_summary",
    "smoke_test_indicators",
    "summarize_feature_output",
]

import glob
import os
from typing import Dict, List, Tuple

import pandas as pd


EXPECTED_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "buy_volume",
    "sell_volume",
    "volume",
]

DEFAULT_TIMEFRAMES = ("1H", "4H", "1D")
EXPECTED_INDICATOR_COLUMNS = [
    "indicator_name",
    "signal_family",
    "natural_direction",
]


def _extract_asset_name(path: str) -> str:
    """
    Extrait le ticker depuis un nom de fichier du type
    `BINANCE_BTCUSDT_future.csv`.
    """
    filename = os.path.basename(path)
    parts = filename.split("_")
    if len(parts) < 3:
        raise ValueError(f"Nom de fichier inattendu: {filename}")
    return parts[1]


def _read_crypto_csv(path: str) -> pd.DataFrame:
    """
    Lit un CSV crypto et retourne un DataFrame proprement indexe par `date`.
    """
    df = pd.read_csv(path, parse_dates=["date"])

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        filename = os.path.basename(path)
        raise ValueError(
            f"Colonnes manquantes dans {filename}: {', '.join(missing_cols)}"
        )

    df = df[EXPECTED_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")
    df = df.drop_duplicates(subset="date", keep="last")
    df = df.set_index("date")

    return df


def _build_ohlcv_agg_map() -> Dict[str, str]:
    """
    Definit l'agregation standard pour resampler des donnees OHLCV.
    """
    return {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "buy_volume": "sum",
        "sell_volume": "sum",
        "volume": "sum",
    }


def _resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample un DataFrame OHLCV vers une timeframe cible.
    """
    pandas_timeframe = timeframe.replace("H", "h")
    resampled = df.resample(pandas_timeframe).agg(_build_ohlcv_agg_map())
    resampled = resampled.dropna(subset=["open", "high", "low", "close"], how="all")
    return resampled


def load_data(
    data_dir: str = "DATA_CRYPTO",
    timeframes: Tuple[str, ...] = DEFAULT_TIMEFRAMES,
) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], List[str]]:
    """
    Charge les donnees crypto depuis les CSV du dossier `data_dir`
    et construit plusieurs DataFrames resamples par timeframe.

    Chaque fichier doit respecter le schema suivant:
    `date, open, high, low, close, buy_volume, sell_volume, volume`

    Retour
    ------
    Tuple[Dict[str, Dict[str, pd.DataFrame]], List[str]]
        - Un dictionnaire `{timeframe: {asset: dataframe}}`.
        - La liste ordonnee des assets detectes.
    """
    pattern = os.path.join(data_dir, "BINANCE_*_future.csv")
    csv_paths = sorted(glob.glob(pattern))

    if not csv_paths:
        print(f"Aucun fichier trouve avec le pattern: {pattern}")
        return {}, []

    raw_data: Dict[str, pd.DataFrame] = {}

    for path in csv_paths:
        asset = _extract_asset_name(path)
        raw_data[asset] = _read_crypto_csv(path)

    assets = sorted(raw_data.keys())
    multi_tf_data: Dict[str, Dict[str, pd.DataFrame]] = {}

    for timeframe in timeframes:
        multi_tf_data[timeframe] = {
            asset: _resample_ohlcv(df, timeframe)
            for asset, df in raw_data.items()
        }

    print(
        f"Donnees crypto chargees: {len(assets)} assets | timeframes: {', '.join(timeframes)}"
    )
    return multi_tf_data, assets


def load_indicator_mapping(
    csv_path: str = "level2_indicator_mapping_model_ready_v2.csv",
) -> pd.DataFrame:
    """
    Charge le fichier de mapping des indicateurs.

    Le fichier doit etre separe par `;` et contenir les colonnes:
    `indicator_name`, `signal_family`, `natural_direction`.
    """
    indicators = pd.read_csv(csv_path, sep=";")

    missing_cols = [
        col for col in EXPECTED_INDICATOR_COLUMNS if col not in indicators.columns
    ]
    if missing_cols:
        raise ValueError(
            f"Colonnes manquantes dans {csv_path}: {', '.join(missing_cols)}"
        )

    indicators = indicators[EXPECTED_INDICATOR_COLUMNS].copy()
    indicators = indicators.dropna(subset=["indicator_name"])
    indicators = indicators.drop_duplicates(subset="indicator_name", keep="last")
    indicators = indicators.reset_index(drop=True)

    print(f"Mapping indicateurs charge: {len(indicators)} lignes depuis {csv_path}")
    return indicators


def inspect_nans(data_dict: Dict[str, pd.DataFrame], data_type: str = "") -> pd.DataFrame:
    """
    Inspecte le nombre de valeurs manquantes par asset et par colonne.
    """
    title = f"INSPECTION {data_type.upper()}".strip()
    print(f"\n{'=' * 20} {title} {'=' * 20}")

    if not data_dict:
        print("(data_dict est vide)")
        return pd.DataFrame()

    summaries = []
    for asset, df in data_dict.items():
        print(f"\n===== {asset} =====")
        print("Nombre de NaN par colonne :")

        nan_counts = df.isna().sum()
        for col, count in nan_counts.items():
            print(f"{col}: {int(count)}")

        summaries.append(nan_counts.rename(asset))

    return pd.DataFrame(summaries).fillna(0).astype(int)

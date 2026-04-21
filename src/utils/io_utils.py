"""
Shared I/O helpers: load settings, save/load CSV and Parquet.
"""
from pathlib import Path
import yaml
import pandas as pd


_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
_settings_cache: dict | None = None


def load_settings() -> dict:
    global _settings_cache
    if _settings_cache is None:
        with open(_SETTINGS_PATH) as f:
            _settings_cache = yaml.safe_load(f)
    return _settings_cache


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

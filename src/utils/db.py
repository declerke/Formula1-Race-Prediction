"""
Database connection factory and upsert helpers.
Reads credentials from environment variables (set via .env).
"""
import os
import logging
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

log = logging.getLogger(__name__)

_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        host = os.environ.get("DB_HOST", "localhost")
        port = os.environ.get("DB_PORT", "5432")
        name = os.environ.get("DB_NAME", "f1_db")
        user = os.environ.get("DB_USER", "f1_user")
        password = os.environ.get("DB_PASSWORD", "f1_password")
        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"
        _engine = create_engine(url, pool_pre_ping=True)
    return _engine


def upsert_df(df: pd.DataFrame, table: str, conflict_cols: list[str]) -> int:
    """
    Insert rows from df into table; skip rows that conflict on conflict_cols.
    Returns the number of rows inserted.
    """
    if df.empty:
        return 0
    engine = get_engine()
    cols = list(df.columns)
    col_names = ", ".join(cols)
    placeholders = ", ".join([f":{c}" for c in cols])
    conflict_target = ", ".join(conflict_cols)

    stmt = text(
        f"INSERT INTO {table} ({col_names}) VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict_target}) DO NOTHING"
    )
    with engine.begin() as conn:
        records = df.where(pd.notnull(df), None).to_dict(orient="records")
        conn.execute(stmt, records)
    return len(df)


def table_row_count(table: str) -> int:
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
        return result.scalar()

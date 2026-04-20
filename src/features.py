from __future__ import annotations
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from .utils import RAW_DIR, PROCESSED_DIR, save_dataframe
RAW_PATH = RAW_DIR / "online_retail_ii.csv"
FEATURES_PATH = PROCESSED_DIR / "customer_features.csv"
SQLITE_PATH = PROCESSED_DIR / "segmentation.db"
def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate", "CustomerID", "Quantity", "UnitPrice"])
    df = df[df["CustomerID"] > 0]
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C", na=False)]
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df
def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    grouped = df.groupby("CustomerID").agg(
        recency_days=("InvoiceDate", lambda s: (snapshot_date - s.max()).days),
        frequency=("InvoiceNo", "nunique"),
        monetary=("Revenue", "sum"),
        total_quantity=("Quantity", "sum"),
        avg_order_value=("Revenue", "mean"),
        active_span_days=("InvoiceDate", lambda s: max((s.max() - s.min()).days, 0)),
        country_count=("Country", "nunique"),
    ).reset_index()
    grouped["avg_items_per_order"] = grouped["total_quantity"] / grouped["frequency"].clip(lower=1)
    grouped["monetary_per_order"] = grouped["monetary"] / grouped["frequency"].clip(lower=1)
    return grouped.replace([np.inf, -np.inf], np.nan).fillna(0)
def persist_sqlite(df: pd.DataFrame, path: Path) -> None:
    with sqlite3.connect(path) as conn:
        df.to_sql("customer_features", conn, if_exists="replace", index=False)
def main() -> None:
    df = pd.read_csv(RAW_PATH)
    cleaned = clean_transactions(df)
    features = build_customer_features(cleaned)
    save_dataframe(cleaned, PROCESSED_DIR / "clean_transactions.csv")
    save_dataframe(features, FEATURES_PATH)
    persist_sqlite(features, SQLITE_PATH)
    print(f"Saved features to {FEATURES_PATH}")
if __name__ == "__main__":
    main()

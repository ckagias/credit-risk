import sqlite3
import time
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
DB = Path("data/home_credit.db")
CHUNK_ROWS = 200_000

TABLES = {
    "application_train": "application_train.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash_balance": "POS_CASH_balance.csv",
    "installments_payments": "installments_payments.csv",
    "credit_card_balance": "credit_card_balance.csv",
}

INDEXES = {
    "application_train": ["SK_ID_CURR"],
    "bureau": ["SK_ID_CURR", "SK_ID_BUREAU"],
    "bureau_balance": ["SK_ID_BUREAU"],
    "previous_application": ["SK_ID_CURR", "SK_ID_PREV"],
    "pos_cash_balance": ["SK_ID_CURR", "SK_ID_PREV"],
    "installments_payments": ["SK_ID_CURR", "SK_ID_PREV"],
    "credit_card_balance": ["SK_ID_CURR", "SK_ID_PREV"],
}


def load_table(con, table, csv_name):
    rows = 0
    first = True
    for chunk in pd.read_csv(RAW / csv_name, chunksize=CHUNK_ROWS):
        chunk.to_sql(table, con, if_exists="replace" if first else "append", index=False)
        first = False
        rows += len(chunk)
    return rows


def add_indexes(con, table, columns):
    for col in columns:
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_{col} ON {table} ({col})")


def main():
    con = sqlite3.connect(DB)
    con.execute("PRAGMA journal_mode = OFF")
    con.execute("PRAGMA synchronous = OFF")
    for table, csv_name in TABLES.items():
        start = time.time()
        rows = load_table(con, table, csv_name)
        con.commit()
        print(f"{table}: {rows:,} rows loaded in {time.time() - start:.0f}s", flush=True)
    for table, columns in INDEXES.items():
        start = time.time()
        add_indexes(con, table, columns)
        con.commit()
        print(f"{table}: indexed {columns} in {time.time() - start:.0f}s", flush=True)
    con.close()


if __name__ == "__main__":
    main()

import sqlite3
import statistics
from pathlib import Path

import pandas as pd

DB = Path("data/home_credit.db")
SECONDARY = ["bureau", "previous_application", "pos_cash_balance", "installments_payments", "credit_card_balance"]
BY_PREV = ["pos_cash_balance", "installments_payments", "credit_card_balance"]


def heading(text):
    print(f"\n=== {text} ===")


def default_rate(con):
    heading("Default rate in application_train")
    for target, n, pct in con.execute(
        "SELECT TARGET, COUNT(*), ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM application_train), 2) "
        "FROM application_train GROUP BY TARGET"
    ):
        print(f"TARGET={target}: {n:,} ({pct}%)")


def missing_values(con):
    heading("Missing values per column (share of rows, only columns with any missing)")
    tables = [t for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
    for table in tables:
        columns = [row[1] for row in con.execute(f"PRAGMA table_info({table})")]
        sums = ", ".join(f'SUM("{c}" IS NULL)' for c in columns)
        result = con.execute(f"SELECT COUNT(*), {sums} FROM {table}").fetchone()
        total, nulls = result[0], result[1:]
        missing = sorted(((n / total, c) for c, n in zip(columns, nulls) if n), reverse=True)
        over_half = sum(1 for share, _ in missing if share > 0.5)
        print(f"\n{table}: {len(columns)} columns, {len(missing)} with missing, {over_half} above 50%")
        for share, column in missing:
            print(f"  {column:34s} {share:6.1%}")


def join_coverage(con):
    heading("Applicants in application_train that have at least one row in each table")
    total = con.execute("SELECT COUNT(*) FROM application_train").fetchone()[0]
    for table in SECONDARY:
        n = con.execute(
            f"SELECT COUNT(*) FROM application_train a WHERE EXISTS "
            f"(SELECT 1 FROM {table} t WHERE t.SK_ID_CURR = a.SK_ID_CURR)"
        ).fetchone()[0]
        print(f"{table:24s} {n:>8,} of {total:,} ({n / total:.1%})")


def orphans(con):
    heading("Rows whose key has no parent row")
    test_ids = set(pd.read_csv("data/raw/application_test.csv", usecols=["SK_ID_CURR"])["SK_ID_CURR"])
    for table in SECONDARY:
        ids = [r[0] for r in con.execute(
            f"SELECT DISTINCT SK_ID_CURR FROM {table} WHERE SK_ID_CURR NOT IN (SELECT SK_ID_CURR FROM application_train)"
        )]
        in_test = sum(1 for i in ids if i in test_ids)
        print(f"{table:24s} {len(ids):>7,} SK_ID_CURR not in train ({in_test:,} are in application_test, {len(ids) - in_test:,} in neither)")
    n = con.execute(
        "SELECT COUNT(*) FROM bureau_balance b WHERE NOT EXISTS (SELECT 1 FROM bureau u WHERE u.SK_ID_BUREAU = b.SK_ID_BUREAU)"
    ).fetchone()[0]
    print(f"bureau_balance rows with SK_ID_BUREAU not in bureau: {n:,}")
    for table in BY_PREV:
        n = con.execute(
            f"SELECT COUNT(*) FROM (SELECT DISTINCT SK_ID_PREV FROM {table}) t WHERE NOT EXISTS "
            f"(SELECT 1 FROM previous_application p WHERE p.SK_ID_PREV = t.SK_ID_PREV)"
        ).fetchone()[0]
        print(f"{table} SK_ID_PREV not in previous_application: {n:,}")


def unique_keys(con):
    heading("Is the key unique?")
    for table, key in [("application_train", "SK_ID_CURR"), ("bureau", "SK_ID_BUREAU"), ("previous_application", "SK_ID_PREV")]:
        rows, distinct = con.execute(f"SELECT COUNT(*), COUNT(DISTINCT {key}) FROM {table}").fetchone()
        print(f"{table}.{key}: {rows:,} rows, {distinct:,} distinct")


def rows_per_key(con):
    heading("Rows per key (min / median / mean / max)")
    for table, key in [(t, "SK_ID_CURR") for t in SECONDARY] + [("bureau_balance", "SK_ID_BUREAU")]:
        counts = [r[0] for r in con.execute(f"SELECT COUNT(*) FROM {table} GROUP BY {key}")]
        print(f"{table:24s} per {key:12s} keys={len(counts):>9,}  {min(counts)} / {statistics.median(counts):.0f} / {statistics.mean(counts):.1f} / {max(counts)}")


def main():
    con = sqlite3.connect(DB)
    default_rate(con)
    unique_keys(con)
    join_coverage(con)
    orphans(con)
    rows_per_key(con)
    missing_values(con)


if __name__ == "__main__":
    main()

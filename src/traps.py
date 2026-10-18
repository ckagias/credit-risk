import sqlite3
from pathlib import Path

DB = Path("data/home_credit.db")
PLACEHOLDER = 365243


def heading(text):
    print(f"\n=== {text} ===")


def columns(con, table):
    return [(row[1], row[2]) for row in con.execute(f"PRAGMA table_info({table})")]


def tables(con):
    return [t for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]


def days_employed(con):
    heading("DAYS_EMPLOYED placeholder in application_train")
    total = con.execute("SELECT COUNT(*) FROM application_train").fetchone()[0]
    n, rate = con.execute(
        f"SELECT COUNT(*), AVG(TARGET) FROM application_train WHERE DAYS_EMPLOYED = {PLACEHOLDER}"
    ).fetchone()
    other_rate, low, high = con.execute(
        f"SELECT AVG(TARGET), MIN(DAYS_EMPLOYED), MAX(DAYS_EMPLOYED) FROM application_train WHERE DAYS_EMPLOYED <> {PLACEHOLDER}"
    ).fetchone()
    print(f"{PLACEHOLDER} appears in {n:,} rows ({n / total:.1%}), default rate {rate:.2%}")
    print(f"other rows: default rate {other_rate:.2%}, DAYS_EMPLOYED from {low:,} to {high:,}")
    print("income type of the placeholder rows:")
    for income, count in con.execute(
        f"SELECT NAME_INCOME_TYPE, COUNT(*) FROM application_train WHERE DAYS_EMPLOYED = {PLACEHOLDER} GROUP BY 1 ORDER BY 2 DESC"
    ):
        print(f"  {income:24s} {count:>7,}")
    print(f"  placeholder rows that are not Pensioner or Unemployed: "
          f"{con.execute(f'SELECT COUNT(*) FROM application_train WHERE DAYS_EMPLOYED = {PLACEHOLDER} AND NAME_INCOME_TYPE NOT IN (\"Pensioner\", \"Unemployed\")').fetchone()[0]:,}")


def days_placeholder_everywhere(con):
    heading(f"Columns holding {PLACEHOLDER} in any table")
    for table in tables(con):
        for column, _ in columns(con, table):
            if not column.startswith("DAYS_"):
                continue
            n = con.execute(f'SELECT COUNT(*) FROM {table} WHERE "{column}" = {PLACEHOLDER}').fetchone()[0]
            if n:
                total = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"{table}.{column}: {n:,} rows ({n / total:.1%})")


def xna(con):
    heading("XNA and XAP values in text columns")
    for table in tables(con):
        for column, kind in columns(con, table):
            if kind != "TEXT":
                continue
            for marker in ("XNA", "XAP"):
                n = con.execute(f'SELECT COUNT(*) FROM {table} WHERE "{column}" = ?', (marker,)).fetchone()[0]
                if not n:
                    continue
                total = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                line = f"{table}.{column} = {marker}: {n:,} rows ({n / total:.1%})"
                if table == "application_train":
                    rate = con.execute(f'SELECT AVG(TARGET) FROM application_train WHERE "{column}" = ?', (marker,)).fetchone()[0]
                    line += f", default rate {rate:.2%}"
                print(line)


def half_missing(con):
    heading("application_train columns missing in more than half of the rows: default rate when missing vs present")
    cols = [c for c, _ in columns(con, "application_train") if c not in ("SK_ID_CURR", "TARGET")]
    total = con.execute("SELECT COUNT(*) FROM application_train").fetchone()[0]
    sums = ", ".join(f'SUM("{c}" IS NULL)' for c in cols)
    nulls = dict(zip(cols, con.execute(f"SELECT {sums} FROM application_train").fetchone()))
    rows = []
    for column in cols:
        if nulls[column] / total <= 0.5:
            continue
        miss, present = con.execute(
            f'SELECT AVG(TARGET) FILTER (WHERE "{column}" IS NULL), AVG(TARGET) FILTER (WHERE "{column}" IS NOT NULL) FROM application_train'
        ).fetchone()
        rows.append((nulls[column] / total, column, miss, present))
    print(f"{len(rows)} columns")
    print(f"{'column':32s} {'missing':>8s} {'default | missing':>18s} {'default | present':>18s}")
    for share, column, miss, present in sorted(rows, reverse=True):
        print(f"{column:32s} {share:8.1%} {miss:18.2%} {present:18.2%}")


def main():
    con = sqlite3.connect(DB)
    days_employed(con)
    days_placeholder_everywhere(con)
    xna(con)
    half_missing(con)


if __name__ == "__main__":
    main()

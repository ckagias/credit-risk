import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

from split import DB

SQL_DIR = Path("sql")
GROUPS = ["bureau", "previous_application", "pos_cash", "installments", "credit_card", "application"]


def table_of(group):
    return f"feat_{group}"


def load_group(con, group, part):
    return pd.read_sql(
        f"SELECT f.* FROM {table_of(group)} f JOIN split s USING (SK_ID_CURR) WHERE s.part = ? ORDER BY f.SK_ID_CURR",
        con,
        params=(part,),
    )


def build_group(con, group):
    con.executescript((SQL_DIR / f"{group}.sql").read_text())
    con.commit()


def row_count_problems(con, group):
    table = table_of(group)
    expected = con.execute("SELECT COUNT(*) FROM application_train").fetchone()[0]
    rows, distinct, missing = con.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT f.SK_ID_CURR), "
        f"(SELECT COUNT(*) FROM application_train a WHERE a.SK_ID_CURR NOT IN (SELECT SK_ID_CURR FROM {table})) "
        f"FROM {table} f"
    ).fetchone()
    problems = []
    if rows != expected:
        problems.append(f"{table} has {rows:,} rows, application_train has {expected:,}")
    if distinct != rows:
        problems.append(f"{table} repeats {rows - distinct:,} applicants")
    if missing:
        problems.append(f"{table} lacks {missing:,} applicants")
    return problems


def build_all(groups=GROUPS):
    con = sqlite3.connect(DB)
    con.execute("PRAGMA cache_size = -2000000")
    con.execute("PRAGMA temp_store = MEMORY")
    for group in groups:
        start = time.time()
        build_group(con, group)
        problems = row_count_problems(con, group)
        columns = len(pd.read_sql(f"SELECT * FROM {table_of(group)} LIMIT 1", con).columns) - 1
        print(f"{group}: {columns} columns in {time.time() - start:.0f}s", "OK" if not problems else problems, flush=True)
    con.close()


if __name__ == "__main__":
    build_all(sys.argv[1:] or GROUPS)

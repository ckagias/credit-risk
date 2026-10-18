import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DB = Path("data/home_credit.db")
SEED = 42
HOLDOUT_SHARE = 0.2


def make_split(applicants):
    train, holdout = train_test_split(
        applicants,
        test_size=HOLDOUT_SHARE,
        stratify=applicants["TARGET"],
        random_state=SEED,
    )
    return pd.concat(
        [
            pd.DataFrame({"SK_ID_CURR": train["SK_ID_CURR"], "part": "train"}),
            pd.DataFrame({"SK_ID_CURR": holdout["SK_ID_CURR"], "part": "holdout"}),
        ]
    ).sort_values("SK_ID_CURR")


def load_split(con):
    return pd.read_sql("SELECT SK_ID_CURR, part FROM split", con)


def main():
    con = sqlite3.connect(DB)
    applicants = pd.read_sql("SELECT SK_ID_CURR, TARGET FROM application_train ORDER BY SK_ID_CURR", con)
    split = make_split(applicants)
    split.to_sql("split", con, if_exists="replace", index=False)
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_split_SK_ID_CURR ON split (SK_ID_CURR)")
    con.commit()
    summary = pd.read_sql(
        "SELECT s.part, COUNT(*) AS applicants, SUM(a.TARGET) AS defaults, ROUND(100.0 * AVG(a.TARGET), 3) AS default_rate "
        "FROM split s JOIN application_train a USING (SK_ID_CURR) GROUP BY s.part",
        con,
    )
    print(summary.to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()

"""Download DOE contracts from USAspending.gov and build the labelled dataset (about 10 minutes).

    python scripts/prepare_data.py                 # FY2018-FY2025, up to 8,000 contracts per year
    python scripts/prepare_data.py --per-year 2000 # quicker try
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spendcoder import data  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--first-fy", type=int, default=2018)
    p.add_argument("--last-fy", type=int, default=2025)
    p.add_argument("--per-year", type=int, default=8000)
    p.add_argument("--out", default="data/spend_lines.csv.gz")
    a = p.parse_args()

    raw = data.fetch(a.first_fy, a.last_fy, a.per_year)
    df = data.build_dataset(raw)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    train, val, test = data.split(df)
    print(f"Saved {len(df):,} labelled lines to {a.out}")
    print(data.summary(train, val, test).to_string())


if __name__ == "__main__":
    main()

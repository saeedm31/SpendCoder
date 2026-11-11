"""Real purchase data: US Department of Energy contracts from USAspending.gov (public domain, no key).

Each contract has a short free-text description written by a buyer, for example
"IGF::OT::IGF ANNUAL MAINTENANCE OF HV SWITCHGEAR", and an official Product and Service Code (PSC).
The description becomes the model input, the PSC (grouped into 14 categories) the label.

Split by time, like production: train on older contracts, validate on the next year, test on the newest.
A test line whose exact text also appears in training is removed, so the test measures new wording.
"""

import re
import time

import pandas as pd
import requests

from .taxonomy import LABEL2ID, psc_to_category

API_URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
FIELDS = ["Award ID", "Description", "Award Amount", "Start Date", "PSC", "generated_internal_id"]
CONTRACT_TYPES = ["A", "B", "C", "D"]      # BPA call, purchase order, delivery order, definitive contract
AGENCY = "Department of Energy"
PAGE_SIZE = 100                            # the API maximum

_IGF = re.compile(r"IGF::[A-Z,\s]*::IGF", re.IGNORECASE)   # "inherently governmental function" markers
_SPACES = re.compile(r"\s+")


def clean_text(text):
    """Remove administrative markers and extra spaces; keep the buyer's wording and casing."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = _IGF.sub(" ", str(text))
    return _SPACES.sub(" ", text).strip(" .-:;,")


def _psc_code(value):
    """The PSC field comes back as {'code': 'R425', 'description': ...} (or a plain string)."""
    if isinstance(value, dict):
        return value.get("code")
    return value if isinstance(value, str) else None


def fetch_year(fiscal_year, max_rows=8000, session=None, pause=0.2, log=print):
    """Download DOE contracts with activity in one US fiscal year (October to September)."""
    session = session or requests.Session()
    body = {
        "subawards": False, "limit": PAGE_SIZE, "page": 1, "sort": "Award ID", "order": "asc",
        "fields": FIELDS,
        "filters": {
            "award_type_codes": CONTRACT_TYPES,
            "agencies": [{"type": "awarding", "tier": "toptier", "name": AGENCY}],
            "time_period": [{"start_date": f"{fiscal_year - 1}-10-01", "end_date": f"{fiscal_year}-09-30"}],
        },
    }
    rows = []
    while len(rows) < max_rows:
        for attempt in range(4):
            r = session.post(API_URL, json=body, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            break
        if r.status_code != 200:
            raise RuntimeError(f"USAspending API error {r.status_code} for FY{fiscal_year}: {r.text[:300]}")
        payload = r.json()
        results = payload.get("results", [])
        rows += results
        has_next = payload.get("page_metadata", {}).get("hasNext", len(results) == PAGE_SIZE)
        if not results or not has_next:
            break
        body["page"] += 1
        time.sleep(pause)
    log(f"FY{fiscal_year}: {len(rows)} contracts")
    return rows[:max_rows]


def fetch(first_fy=2018, last_fy=2025, max_rows_per_year=8000, session=None, log=print):
    """All years as one raw DataFrame (one row per contract per year it appears in)."""
    frames = []
    for fy in range(first_fy, last_fy + 1):
        df = pd.DataFrame(fetch_year(fy, max_rows_per_year, session=session, log=log))
        df["fiscal_year"] = fy
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_dataset(raw):
    """Raw API rows -> one clean, labelled row per contract: text, category, label, psc, start_date."""
    df = pd.DataFrame({
        "award_id": raw.get("generated_internal_id", raw.get("Award ID")),
        "text": raw["Description"].map(clean_text),
        "psc": raw["PSC"].map(_psc_code),
        "start_date": pd.to_datetime(raw["Start Date"], errors="coerce"),
        "amount": pd.to_numeric(raw.get("Award Amount"), errors="coerce"),
    })
    df["category"] = df["psc"].map(psc_to_category)
    df = df.dropna(subset=["category", "start_date"])
    df = df[df["text"].str.len() >= 4]
    df = df.sort_values("start_date").drop_duplicates("award_id", keep="first")   # a contract counts once
    df["label"] = df["category"].map(LABEL2ID).astype(int)
    return df.reset_index(drop=True)


def split(df, val_start="2023-10-01", test_start="2024-10-01"):
    """Time split by contract start date. Removes val/test lines whose exact text is in train."""
    train = df[df["start_date"] < val_start]
    val = df[(df["start_date"] >= val_start) & (df["start_date"] < test_start)]
    test = df[df["start_date"] >= test_start]
    seen = set(train["text"].str.lower())
    val = val[~val["text"].str.lower().isin(seen)]
    test = test[~test["text"].str.lower().isin(seen)]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def summary(train, val, test):
    """Rows per category and split, for a quick look at the label balance."""
    parts = {name: part["category"].value_counts() for name, part in
             (("train", train), ("val", val), ("test", test))}
    return pd.DataFrame(parts).fillna(0).astype(int).sort_values("train", ascending=False)

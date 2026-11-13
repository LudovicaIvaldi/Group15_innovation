"""
extract_sample_carlo.py - Extract a Balanced 100-Patent Sample directly from Carlo's Step 0 Output.

Run this script or notebook cell directly on the EPO University Server where `epo.tipdata` is installed.
"""

from pathlib import Path
import pandas as pd
from sqlalchemy import and_, func, or_

# 1. EPO Connection
from epo.tipdata.patstat import PatstatClient
from epo.tipdata.patstat.database.models import (
    TLS201_APPLN,
    TLS202_APPLN_TITLE,
    TLS203_APPLN_ABSTR,
)

ENV = "PROD"  # Environment (PROD or TEST)
START_YEAR, END_YEAR = 1990, 2023
IPR_TYPE = "PI"
N_POS, N_NEG = 50, 50
RANDOM_STATE = 42
NEG_MOD, NEG_REMAINDER = 1000, 7

# Path to Carlo's Step 0 output
ROOT_DIR = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "outputs").exists()), Path.cwd())
CARLO_FILE = ROOT_DIR / f"outputs/step0/qc_families_{ENV}.csv"

OUTPUT_DIR = ROOT_DIR / "outputs/validation_carlo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_CSV = OUTPUT_DIR / "validation_sample.csv"
BATCH1_FILE = OUTPUT_DIR / "patents_batch1.txt"
BATCH2_FILE = OUTPUT_DIR / "patents_batch2.txt"

print(f"Loading Carlo's Step 0 file from: {CARLO_FILE}")
df_carlo = pd.read_csv(CARLO_FILE)
carlo_family_ids = df_carlo["docdb_family_id"].dropna().unique().tolist()
print(f"Total families in Carlo's sector: {len(carlo_family_ids):,}")

patstat = PatstatClient(env=ENV)
db = patstat.orm()

has_text = or_(
    TLS202_APPLN_TITLE.appln_title.isnot(None),
    TLS203_APPLN_ABSTR.appln_abstract.isnot(None),
)

def eligible_families(*filters):
    return patstat.df(
        db.query(TLS201_APPLN.docdb_family_id.label("docdb_family_id"))
        .outerjoin(TLS202_APPLN_TITLE, TLS201_APPLN.appln_id == TLS202_APPLN_TITLE.appln_id)
        .outerjoin(TLS203_APPLN_ABSTR, TLS201_APPLN.appln_id == TLS203_APPLN_ABSTR.appln_id)
        .filter(TLS201_APPLN.docdb_family_id > 0)
        .filter(TLS201_APPLN.earliest_filing_year.between(START_YEAR, END_YEAR))
        .filter(has_text)
        .filter(*filters)
        .filter(TLS201_APPLN.ipr_type == IPR_TYPE)
        .distinct()
    )

print("Sampling 50 POSITIVES from Carlo's families...")
in_carlo = TLS201_APPLN.docdb_family_id.in_(carlo_family_ids)
pool_pos = eligible_families(in_carlo)
df_pos = pool_pos.sample(n=min(N_POS, len(pool_pos)), random_state=RANDOM_STATE).copy()
df_pos["query_label"] = 1

print("Sampling 50 NEGATIVES from outside Carlo's families...")
pool_neg = eligible_families(~in_carlo, TLS201_APPLN.docdb_family_id % NEG_MOD == NEG_REMAINDER)
df_neg = pool_neg.sample(n=min(N_NEG, len(pool_neg)), random_state=RANDOM_STATE).copy()
df_neg["query_label"] = 0

df_sample = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
all_sample_ids = df_sample["docdb_family_id"].tolist()

print(f"Fetching titles and abstracts from PATSTAT for {len(all_sample_ids)} families...")
text_query = (
    db.query(
        TLS201_APPLN.docdb_family_id,
        TLS201_APPLN.appln_id,
        func.max(TLS202_APPLN_TITLE.appln_title).label("appln_title"),
        func.max(TLS203_APPLN_ABSTR.appln_abstract).label("appln_abstract"),
    )
    .outerjoin(TLS202_APPLN_TITLE, TLS201_APPLN.appln_id == TLS202_APPLN_TITLE.appln_id)
    .outerjoin(TLS203_APPLN_ABSTR, TLS201_APPLN.appln_id == TLS203_APPLN_ABSTR.appln_id)
    .filter(TLS201_APPLN.docdb_family_id.in_(all_sample_ids))
    .filter(has_text)
    .group_by(TLS201_APPLN.docdb_family_id, TLS201_APPLN.appln_id)
)

df_texts = patstat.df(text_query)
df_texts["has_abs"] = df_texts["appln_abstract"].notna()
df_texts["has_title"] = df_texts["appln_title"].notna()
df_texts = (
    df_texts.sort_values(["docdb_family_id", "has_abs", "has_title", "appln_id"],
                         ascending=[True, False, False, True])
    .drop_duplicates("docdb_family_id")
    .drop(columns=["has_abs", "has_title"])
)

df_sample = df_sample.merge(df_texts, on="docdb_family_id", how="left")
df_sample.to_csv(SAMPLE_CSV, index=False)
print(f"Saved 100-patent sample to: {SAMPLE_CSV}")

# Generate LLM evaluation prompt files
def create_batch_file(filepath, batch_df, batch_num, start_idx):
    lines = [
        f"# BATCH {batch_num} (Patents {start_idx} to {start_idx + len(batch_df) - 1})",
        "",
        "Please evaluate each patent below according to the Quantum Computing domain rules.",
        "Output format: CSV with columns: docdb_family_id,prediction,confidence_score,reason",
        "",
        "=" * 60,
        ""
    ]
    for i, (_, row) in enumerate(batch_df.iterrows(), start_idx):
        lines.append(f"--- Patent [{i}/100] ---")
        lines.append(f"docdb_family_id: {row['docdb_family_id']}")
        lines.append(f"Title: {row.get('appln_title', '')}")
        lines.append(f"Abstract: {row.get('appln_abstract', '')}")
        lines.append("")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

create_batch_file(BATCH1_FILE, df_sample.iloc[:50], 1, 1)
create_batch_file(BATCH2_FILE, df_sample.iloc[50:], 2, 51)
print(f"Generated LLM prompt files:\n  - {BATCH1_FILE}\n  - {BATCH2_FILE}")
print("\nDone! You can now download these files to your laptop or evaluate them directly with the models.")

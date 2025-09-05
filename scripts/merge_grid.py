#!/usr/bin/env python3
import os, glob, re
import pandas as pd

# ---- configure roots you want to scan ----
ROOTS = ["results/tfs", "results"]   # edit as needed
OUT_DIR = "results/combined"
OUT_BASE = "all_spaces_summary_with_paths"   # we'll reuse this base name

# Regex to extract subject from filename suffix: all_spaces_summary_<SUBJECT>.csv
SUBJ_RE = re.compile(r"all_spaces_summary_([^/\\]+)\.csv$")

def find_subject_csvs():
    files = []
    for root in ROOTS:
        if not os.path.isdir(root):
            continue
        files.extend(glob.glob(os.path.join(root, "**", "all_spaces_summary_*.csv"), recursive=True))
    return sorted(set(files))

def extract_subject(path):
    m = SUBJ_RE.search(path)
    if not m:
        return None
    return m.group(1)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    for path in find_subject_csvs():
        subj = extract_subject(path)
        if subj is None:
            print(f"[skip/name] couldn't parse subject from: {path}")
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[skip/read] {path}: {e}")
            continue

        # keep just one row (you said all rows are the same)
        if len(df) == 0:
            print(f"[skip/empty] {path}")
            continue
        one = df.copy()
        one["subject"] = subj
        one["source_csv"] = path
        rows.append(one)

    if not rows:
        print("[merge] no subject CSVs found. Nothing written.")
        return

    big = pd.concat(rows, ignore_index=True)

    csv_out = os.path.join(OUT_DIR, f"{OUT_BASE}.csv")
    pkl_out = os.path.join(OUT_DIR, f"{OUT_BASE}.pkl")
    big.to_csv(csv_out, index=False)
    big.to_pickle(pkl_out)

    try:
        pq_out = os.path.join(OUT_DIR, f"{OUT_BASE}.parquet")
        big.to_parquet(pq_out, index=False)
        pq_msg = f", parquet → {pq_out}"
    except Exception as e:
        pq_msg = f" (parquet skipped: {e})"

    print(f"[merge] wrote {len(big)} subjects; CSV → {csv_out}{pq_msg}")

if __name__ == "__main__":
    main()

"""
Step 2 — Event extraction (no datetime, Apple Silicon safe).
Uses string-based time arithmetic to avoid segfault.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config as C
from . import io_utils as io


def safe_t_hours(charttime_series, intime_series):
    """Compute elapsed hours without triggering Apple Silicon segfault."""
    ct = pd.to_datetime(charttime_series, errors='coerce', utc=False)
    it = pd.to_datetime(intime_series, errors='coerce', utc=False)
    return (ct - it).dt.total_seconds() / 3600.0


def _apply_bounds(df):
    for name, (lo, hi) in C.PHYS_BOUNDS.items():
        mask = (df["name"] == name) & ((df["value"] < lo) | (df["value"] > hi))
        df = df[~mask]
    return df


def extract_vitals(cohort):
    stay_intime = cohort.set_index("stay_id")["intime"]
    stay_ids = set(cohort["stay_id"].tolist())
    itemid_to_name = C.flat_vital_map()
    keep_itemids = set(itemid_to_name)

    out = []
    chunk_n = 0
    for chunk in io.iter_chunks(
        C.MIMIC_ROOT, "chartevents",
        usecols=["subject_id", "stay_id", "charttime", "itemid", "valuenum"],
        chunksize=1_000_000,
        dtype={"stay_id": "Int64", "itemid": "Int64", "valuenum": "float32"},
    ):
        chunk = chunk[chunk["stay_id"].isin(stay_ids)]
        chunk = chunk[chunk["itemid"].isin(keep_itemids)]
        chunk = chunk.dropna(subset=["valuenum"])
        if chunk.empty:
            chunk_n += 1
            continue
        chunk["name"] = chunk["itemid"].map(itemid_to_name)
        intime_mapped = chunk["stay_id"].map(stay_intime)
        chunk["t_hours"] = safe_t_hours(chunk["charttime"], intime_mapped)
        chunk = chunk.dropna(subset=["t_hours"])
        chunk = chunk[(chunk["t_hours"] >= 0) &
                      (chunk["t_hours"] <= C.PRED_HORIZON_HOURS)]
        chunk = chunk.rename(columns={"valuenum": "value"})
        out.append(chunk[["subject_id", "stay_id", "t_hours", "name", "value"]])
        chunk_n += 1
        print(f"  vitals chunk {chunk_n}: {len(chunk):,} rows kept")

    if not out:
        return pd.DataFrame(columns=["subject_id","stay_id","t_hours","name","value"])
    df = pd.concat(out, ignore_index=True)
    df = _apply_bounds(df)
    if "temperature_f" in df["name"].values:
        m = df["name"] == "temperature_f"
        df.loc[m, "value"] = (df.loc[m, "value"] - 32.0) * 5.0 / 9.0
        df.loc[m, "name"] = "temperature_c"
    return df.reset_index(drop=True)


def extract_labs(cohort):
    hadm_intime = cohort.set_index("hadm_id")["intime"]
    hadm_ids = set(cohort["hadm_id"].tolist())
    itemid_to_name = C.flat_lab_map()
    keep_itemids = set(itemid_to_name)

    out = []
    for chunk in io.iter_chunks(
        C.MIMIC_ROOT, "labevents",
        usecols=["subject_id", "hadm_id", "charttime", "itemid", "valuenum"],
        chunksize=1_000_000,
        dtype={"hadm_id": "Int64", "itemid": "Int64", "valuenum": "float32"},
    ):
        chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]
        chunk = chunk[chunk["itemid"].isin(keep_itemids)]
        chunk = chunk.dropna(subset=["valuenum", "hadm_id"])
        if chunk.empty:
            continue
        chunk["name"] = chunk["itemid"].map(itemid_to_name)
        intime_mapped = chunk["hadm_id"].map(hadm_intime)
        chunk["t_hours"] = safe_t_hours(chunk["charttime"], intime_mapped)
        chunk = chunk.dropna(subset=["t_hours"])
        chunk = chunk[(chunk["t_hours"] >= 0) &
                      (chunk["t_hours"] <= C.PRED_HORIZON_HOURS)]
        chunk = chunk.rename(columns={"valuenum": "value"})
        out.append(chunk[["subject_id", "hadm_id", "t_hours", "name", "value"]])

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["subject_id","hadm_id","t_hours","name","value"])


def extract_meds(cohort):
    stay_intime = cohort.set_index("stay_id")["intime"]
    stay_ids = set(cohort["stay_id"].tolist())

    out = []
    for chunk in io.iter_chunks(
        C.MIMIC_ROOT, "inputevents",
        usecols=["subject_id", "stay_id", "starttime",
                 "itemid", "amount", "rate", "ordercategoryname"],
        chunksize=500_000,
        dtype={"stay_id": "Int64"},
    ):
        chunk = chunk[chunk["stay_id"].isin(stay_ids)]
        chunk = chunk.dropna(subset=["starttime"])
        if chunk.empty:
            continue
        intime_mapped = chunk["stay_id"].map(stay_intime)
        chunk["t_hours"] = safe_t_hours(chunk["starttime"], intime_mapped)
        chunk = chunk.dropna(subset=["t_hours"])
        chunk = chunk[(chunk["t_hours"] >= 0) &
                      (chunk["t_hours"] <= C.PRED_HORIZON_HOURS)]
        chunk = chunk.rename(columns={"ordercategoryname": "name"})
        out.append(chunk[["subject_id", "stay_id",
                          "t_hours", "name", "amount", "rate"]])

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["subject_id","stay_id","t_hours","name","amount","rate"])


def extract_diagnoses(cohort):
    hadm_ids = set(cohort["hadm_id"].tolist())
    dx = io.read(C.MIMIC_ROOT, "diagnoses_icd",
        usecols=["subject_id","hadm_id","icd_code","icd_version","seq_num"])
    return dx[dx["hadm_id"].isin(hadm_ids)].reset_index(drop=True)


def main():
    cohort = pd.read_parquet(C.OUT_ROOT / "cohort.parquet")
    out = C.OUT_ROOT

    print("[1/4] vitals (chartevents) ...")
    vitals = extract_vitals(cohort)
    vitals.to_parquet(out / "vitals.parquet", index=False)
    print(f"  vitals total: {len(vitals):,} rows")

    print("[2/4] labs (labevents) ...")
    labs = extract_labs(cohort)
    labs.to_parquet(out / "labs.parquet", index=False)
    print(f"  labs total: {len(labs):,} rows")

    print("[3/4] meds (inputevents) ...")
    meds = extract_meds(cohort)
    meds.to_parquet(out / "meds.parquet", index=False)
    print(f"  meds total: {len(meds):,} rows")

    print("[4/4] diagnoses ...")
    dx = extract_diagnoses(cohort)
    dx.to_parquet(out / "diagnoses.parquet", index=False)
    print(f"  diagnoses: {len(dx):,} rows")

    print("Step 2 complete!")


if __name__ == "__main__":
    main()

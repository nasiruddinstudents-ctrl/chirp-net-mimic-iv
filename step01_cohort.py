"""
Step 1 — Cohort extraction (segfault-safe).
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from . import config as C
from . import io_utils as io

def build_cohort() -> pd.DataFrame:
    patients = io.read(C.MIMIC_ROOT, "patients",
        usecols=["subject_id", "gender", "anchor_age"])
    
    admissions = io.read(C.MIMIC_ROOT, "admissions",
        usecols=["subject_id", "hadm_id", "admittime", 
                 "deathtime", "hospital_expire_flag", "race"])
    
    icustays = io.read(C.MIMIC_ROOT, "icustays",
        usecols=["subject_id", "hadm_id", "stay_id", 
                 "intime", "outtime", "los"])

    df = (icustays
        .merge(patients, on="subject_id", how="left")
        .merge(admissions, on=["subject_id", "hadm_id"], how="left"))

    # Adults only
    df = df[df["anchor_age"] >= C.MIN_AGE]

    # First stay per patient
    df = df.sort_values(["subject_id", "intime"])
    df = df.drop_duplicates(subset=["subject_id"], keep="first")

    # LOS filter
    df["los_hours"] = df["los"] * 24.0
    df = df[df["los_hours"] >= C.MIN_LOS_HOURS]

    # Mortality label - use hospital_expire_flag directly
    df["mortality_label"] = df["hospital_expire_flag"].fillna(0).astype(int)
    
    # pred_window_end as string offset (avoid datetime on Apple Silicon)
    df["intime_str"] = df["intime"].astype(str)
    df["pred_window_end"] = df["intime_str"]  # placeholder

    keep = ["subject_id", "hadm_id", "stay_id",
            "gender", "anchor_age", "race",
            "intime", "outtime", "los_hours",
            "hospital_expire_flag", "mortality_label",
            "pred_window_end"]
    return df[keep].reset_index(drop=True)

def main():
    C.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cohort = build_cohort()
    out = C.OUT_ROOT / "cohort.parquet"
    cohort.to_parquet(out, index=False)
    n = len(cohort)
    pos = int(cohort["mortality_label"].sum())
    print(f"cohort size: {n:,}   mortality: {pos:,} ({pos/n:.1%})")
    print(f"median age:  {cohort['anchor_age'].median():.0f}")
    print(f"wrote {out}")

if __name__ == "__main__":
    main()

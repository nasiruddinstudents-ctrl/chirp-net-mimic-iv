"""
Step 3 — Build the Continuous-Time Heterogeneous EHR Graph (CT-HEG).

For each cohort ICU stay we materialize one `torch_geometric.data.HeteroData`
object that mirrors the schema in Section III of the CHIRP draft:

  Node types:
    visit        1 node (index stay). Feature = demographics + admission stats.
    vital        one node per unique vital *variable* observed in the window
                 (heart_rate, sbp, ...). Feature = summary stats over its
                 time series in the window (last, mean, std, min, max, count).
    lab_event    one node per unique lab *variable* observed. Feature = same
                 summary stats.
    med_event    one node per unique medication *label* administered.
                 Feature = (total_amount, mean_rate, n_admins).
    diagnosis    one node per ICD code assigned to the hosp admission.
                 Feature = one-hot on top-K vocab (or code index).

  Edge types (all typed & timestamped):
    (visit, has_vital,     vital)      edge_attr = per-observation (t_hours, value)
    (visit, has_lab,       lab_event)  edge_attr = per-observation (t_hours, value)
    (visit, has_med,       med_event)  edge_attr = per-admin      (t_hours, amount, rate)
    (visit, has_diagnosis, diagnosis)  edge_attr = (seq_num,)

Because in CHIRP the message passing is *timestamped*, we do NOT aggregate
observations into a single edge per (visit, variable). We keep one edge per
raw observation. The variable node still exists as the type-conditioned
recipient of many timestamped edges; the Neural-ODE inter-event dynamics
in CHIRP-Net then evolve the variable-node state between edges.

This script writes:
    graphs/<stay_id>.pt
    graphs_index.parquet   (stay_id, mortality_label, n_vital_edges, ...)
    vocab_diagnoses.parquet
    vocab_meds.parquet
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

from . import config as C


# ---------- vocab helpers -------------------------------------------------

TOP_K_DIAG = 500
TOP_K_MED  = 200


def build_vocab(series: pd.Series, top_k: int) -> dict[str, int]:
    counts = series.value_counts().head(top_k)
    return {v: i for i, v in enumerate(counts.index.tolist())}


# ---------- feature helpers ----------------------------------------------

def variable_features(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Summary stats for one variable's observations within the window."""
    if len(values) == 0:
        return np.zeros(6, dtype=np.float32)
    return np.array([
        values[-1] if len(values) else 0.0,   # last
        values.mean(),
        values.std() if len(values) > 1 else 0.0,
        values.min(),
        values.max(),
        len(values),
    ], dtype=np.float32)


def visit_features(row: pd.Series) -> np.ndarray:
    return np.array([
        float(row["anchor_age"]),
        1.0 if row["gender"] == "M" else 0.0,
        float(row["los_hours"]),
    ], dtype=np.float32)


# ---------- per-stay graph builder ---------------------------------------

def build_graph_for_stay(
    stay_row: pd.Series,
    vitals: pd.DataFrame,
    labs: pd.DataFrame,
    meds: pd.DataFrame,
    dx: pd.DataFrame,
    vocab_dx: dict,
    vocab_med: dict,
) -> HeteroData | None:
    g = HeteroData()

    # ----- visit node
    g["visit"].x = torch.tensor(visit_features(stay_row)).unsqueeze(0)
    g["visit"].y = torch.tensor([int(stay_row["mortality_label"])], dtype=torch.long)

    # ----- vital nodes + edges
    if len(vitals):
        vital_names = sorted(vitals["name"].unique().tolist())
        v_idx = {n: i for i, n in enumerate(vital_names)}
        v_feats = np.stack([
            variable_features(
                vitals.loc[vitals["name"] == n, "value"].to_numpy(),
                vitals.loc[vitals["name"] == n, "t_hours"].to_numpy(),
            )
            for n in vital_names
        ])
        g["vital"].x = torch.tensor(v_feats)
        src = torch.zeros(len(vitals), dtype=torch.long)  # single visit node
        dst = torch.tensor([v_idx[n] for n in vitals["name"]], dtype=torch.long)
        g["visit", "has_vital", "vital"].edge_index = torch.stack([src, dst], dim=0)
        g["visit", "has_vital", "vital"].edge_attr = torch.tensor(
            vitals[["t_hours", "value"]].to_numpy(dtype=np.float32)
        )
    else:
        g["vital"].x = torch.zeros((0, 6))
        g["visit", "has_vital", "vital"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        g["visit", "has_vital", "vital"].edge_attr = torch.zeros((0, 2))

    # ----- lab nodes + edges
    if len(labs):
        lab_names = sorted(labs["name"].unique().tolist())
        l_idx = {n: i for i, n in enumerate(lab_names)}
        l_feats = np.stack([
            variable_features(
                labs.loc[labs["name"] == n, "value"].to_numpy(),
                labs.loc[labs["name"] == n, "t_hours"].to_numpy(),
            )
            for n in lab_names
        ])
        g["lab_event"].x = torch.tensor(l_feats)
        src = torch.zeros(len(labs), dtype=torch.long)
        dst = torch.tensor([l_idx[n] for n in labs["name"]], dtype=torch.long)
        g["visit", "has_lab", "lab_event"].edge_index = torch.stack([src, dst], dim=0)
        g["visit", "has_lab", "lab_event"].edge_attr = torch.tensor(
            labs[["t_hours", "value"]].to_numpy(dtype=np.float32)
        )
    else:
        g["lab_event"].x = torch.zeros((0, 6))
        g["visit", "has_lab", "lab_event"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        g["visit", "has_lab", "lab_event"].edge_attr = torch.zeros((0, 2))

    # ----- med nodes + edges (drop unknown vocab entries)
    if len(meds):
        meds = meds[meds["name"].isin(vocab_med)].copy()
    if len(meds):
        med_names = sorted(meds["name"].unique().tolist())
        m_idx = {n: i for i, n in enumerate(med_names)}
        m_feats = []
        for n in med_names:
            sub = meds[meds["name"] == n]
            m_feats.append([
                float(sub["amount"].sum(skipna=True) or 0.0),
                float(sub["rate"].mean(skipna=True) or 0.0),
                float(len(sub)),
            ])
        g["med_event"].x = torch.tensor(np.array(m_feats, dtype=np.float32))
        src = torch.zeros(len(meds), dtype=torch.long)
        dst = torch.tensor([m_idx[n] for n in meds["name"]], dtype=torch.long)
        g["visit", "has_med", "med_event"].edge_index = torch.stack([src, dst], dim=0)
        amt = meds["amount"].fillna(0.0).to_numpy(dtype=np.float32)
        rate = meds["rate"].fillna(0.0).to_numpy(dtype=np.float32)
        g["visit", "has_med", "med_event"].edge_attr = torch.tensor(
            np.stack([meds["t_hours"].to_numpy(dtype=np.float32), amt, rate], axis=1)
        )
    else:
        g["med_event"].x = torch.zeros((0, 3))
        g["visit", "has_med", "med_event"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        g["visit", "has_med", "med_event"].edge_attr = torch.zeros((0, 3))

    # ----- diagnosis nodes + edges
    if len(dx):
        dx = dx[dx["icd_code"].isin(vocab_dx)].copy()
    if len(dx):
        dx_codes = sorted(dx["icd_code"].unique().tolist())
        d_idx = {c: i for i, c in enumerate(dx_codes)}
        d_feats = np.array([[vocab_dx[c]] for c in dx_codes], dtype=np.float32)
        g["diagnosis"].x = torch.tensor(d_feats)
        src = torch.zeros(len(dx), dtype=torch.long)
        dst = torch.tensor([d_idx[c] for c in dx["icd_code"]], dtype=torch.long)
        g["visit", "has_diagnosis", "diagnosis"].edge_index = torch.stack([src, dst], dim=0)
        seq = dx["seq_num"].fillna(0.0).to_numpy(dtype=np.float32).reshape(-1, 1)
        g["visit", "has_diagnosis", "diagnosis"].edge_attr = torch.tensor(seq)
    else:
        g["diagnosis"].x = torch.zeros((0, 1))
        g["visit", "has_diagnosis", "diagnosis"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        g["visit", "has_diagnosis", "diagnosis"].edge_attr = torch.zeros((0, 1))

    return g


# ---------- main pipeline -------------------------------------------------

def main():
    out_root = C.OUT_ROOT
    (out_root / "graphs").mkdir(parents=True, exist_ok=True)

    cohort = pd.read_parquet(out_root / "cohort.parquet")
    vitals = pd.read_parquet(out_root / "vitals.parquet")
    labs   = pd.read_parquet(out_root / "labs.parquet")
    meds   = pd.read_parquet(out_root / "meds.parquet")
    dx     = pd.read_parquet(out_root / "diagnoses.parquet")

    # ---- vocab
    vocab_dx  = build_vocab(dx["icd_code"], TOP_K_DIAG)
    vocab_med = build_vocab(meds["name"],   TOP_K_MED)
    pd.DataFrame({"icd_code": list(vocab_dx),  "idx": list(vocab_dx.values())}
                 ).to_parquet(out_root / "vocab_diagnoses.parquet", index=False)
    pd.DataFrame({"name":     list(vocab_med), "idx": list(vocab_med.values())}
                 ).to_parquet(out_root / "vocab_meds.parquet", index=False)

    # ---- group event tables once for O(1) per-stay lookup
    vitals_by = dict(tuple(vitals.groupby("stay_id"))) if len(vitals) else {}
    meds_by   = dict(tuple(meds.groupby("stay_id")))   if len(meds)   else {}
    labs_by   = dict(tuple(labs.groupby("hadm_id")))   if len(labs)   else {}
    dx_by     = dict(tuple(dx.groupby("hadm_id")))     if len(dx)     else {}

    index_rows = []
    empty = pd.DataFrame()
    for _, row in tqdm(cohort.iterrows(), total=len(cohort), desc="graphs"):
        sid = int(row["stay_id"])
        hid = int(row["hadm_id"])
        g = build_graph_for_stay(
            stay_row=row,
            vitals=vitals_by.get(sid, empty),
            labs=labs_by.get(hid, empty),
            meds=meds_by.get(sid, empty),
            dx=dx_by.get(hid, empty),
            vocab_dx=vocab_dx,
            vocab_med=vocab_med,
        )
        torch.save(g, out_root / "graphs" / f"{sid}.pt")

        index_rows.append({
            "stay_id": sid,
            "hadm_id": hid,
            "subject_id": int(row["subject_id"]),
            "mortality_label": int(row["mortality_label"]),
            "n_vital_edges": int(g["visit", "has_vital", "vital"].edge_index.shape[1]),
            "n_lab_edges":   int(g["visit", "has_lab", "lab_event"].edge_index.shape[1]),
            "n_med_edges":   int(g["visit", "has_med", "med_event"].edge_index.shape[1]),
            "n_dx_edges":    int(g["visit", "has_diagnosis", "diagnosis"].edge_index.shape[1]),
        })

    idx = pd.DataFrame(index_rows)
    idx.to_parquet(out_root / "graphs_index.parquet", index=False)
    print(f"wrote {len(idx):,} graphs to {out_root/'graphs'}")


if __name__ == "__main__":
    main()

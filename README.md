# CHIRP-Net: Bidirectional, Timestamp-Attributed Event Graphs for ICU Mortality Prediction

Code accompanying the manuscript submitted to *BMC Medical Informatics and Decision Making*:
**"CT-HEG: A Bidirectional, Timestamp-Attributed Event Graph for ICU In-Hospital Mortality
Prediction — An Architectural Ablation Study"**

## Overview

This repository contains the training scripts for CHIRP-Net, three architectural ablations,
four baseline comparators, and the calibration/ensemble evaluation script, all run on
MIMIC-IV v3.1 (31,142 ICU stays, LOS≥48h, 13.4% in-hospital mortality).

## Contents

| File | Purpose |
|---|---|
| `train_chirp_v5.py` | Full CHIRP-Net model: 4-layer HeteroConv, GATv2Conv, bidirectional edges |
| `ablation_no_reverse.py` | Ablation: remove reverse edges (connectivity/reachability check) |
| `ablation_no_time.py` | Ablation: zero out timestamp edge attribute |
| `ablation_homogeneous.py` | Ablation: collapse heterogeneous edge types into one relation |
| `baseline_grud.py` | GRU-D baseline |
| `baseline_mtand.py` | mTAND baseline |
| `baseline_transformer.py` | 4-layer Transformer baseline |
| `baseline_lr_FIXED.py` | Logistic regression baseline (corrected cohort version) |
| `calibration_ensemble_proper.py` | Validation-fitted temperature scaling + 5-checkpoint ensemble evaluation |

## Cohort

Adult patients (age ≥18), first eligible ICU stay only (one stay per `subject_id`),
LOS ≥48h. 31,142 stays; 13.4% in-hospital mortality. Cohort construction retains
exactly one row per patient, so patient-level split disjointness is guaranteed
by construction — verified directly against `cohort_clean.parquet`
(31,142 rows, 31,142 unique `subject_id` values).

## Reproducing Table II / III / IV

All scripts read from `graphs_index_clean.parquet` and `cohort_clean.parquet`
(the deduplicated cohort), use `random_state=42` for the 70/15/15 split, and
train 5 seeds (42–46) unless noted as a single run (GRU-D).

```bash
# Full model, one seed
SEED=42 python3 train_chirp_v5.py

# Ablations (5 seeds each)
for SEED in 42 43 44 45 46; do
  SEED=$SEED python3 ablation_no_reverse.py
  SEED=$SEED python3 ablation_no_time.py
  SEED=$SEED python3 ablation_homogeneous.py
done

# Baselines
python3 baseline_lr_FIXED.py
for SEED in 42 43 44 45 46; do
  SEED=$SEED python3 baseline_grud.py
  SEED=$SEED python3 baseline_mtand.py
  SEED=$SEED python3 baseline_transformer.py
done

# Calibration + ensemble (requires saved checkpoints from all 5 CHIRP-Net seeds)
python3 calibration_ensemble_proper.py
```

## Known data-provenance note

An earlier version of the logistic-regression baseline (`baseline_lr.py`, not included
here) was trained on an un-deduplicated cohort file. `baseline_lr_FIXED.py` is the
corrected version, reading from `graphs_index_clean.parquet`, and is what the
manuscript's Table II figures are drawn from.

## License

BSD 3-Clause. See `LICENSE`.

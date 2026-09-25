# CT-HEG / CHIRP-Net — post-48-hour ICU mortality on MIMIC-IV v3.1

Code for *CT-HEG: Timestamp-Conditioned Message Passing for Post-48-Hour ICU Mortality Prediction — An Architectural Ablation Study*.

No MIMIC-IV data or MIMIC-derived files are included. You need credentialed PhysioNet access to MIMIC-IV v3.1
(https://physionet.org/content/mimiciv/3.1/). Do not commit data, graphs, checkpoints or per-patient predictions to this repository.

## Pipeline (run in order)
| Step | Script | Output |
|---|---|---|
| 1 | `step01_cohort.py` | adult first ICU stay per patient, LOS >= 48 h: 31,142 stays / 31,142 patients (`cohort.parquet`) |
| 2 | `step02_events.py` | vitals, labs and medication events within the first 48 h |
| 3 | `step03_build_ct_heg.py` | one PyTorch Geometric `HeteroData` graph per stay (`graphs/<stay_id>.pt`) + `graphs_index.parquet` |
| 4 | `run_queue.sh` -> `run_unified_nodx.py` | trains every CHIRP-Net condition, 5 seeds each, under one protocol; writes per-seed test predictions |
| 5 | `paired_bootstrap.py` | paired patient-level bootstrap CIs reported in Tables 5-6 |

Notes
- The training script expects the step-1 and step-3 outputs under the names `cohort_clean.parquet` and `graphs_index_clean.parquet`
  (identical content; copy or rename them).
- Step 3 also builds diagnosis nodes; the training script removes them before use because diagnosis-code timing
  cannot be verified at the 48-hour cutoff (see the paper's Leakage Audit).
- The train/validation/test split (70/15/15, stratified, seed 42) and all normalization statistics are computed inside
  `run_unified_nodx.py`; `step04_split.py` is not used for the reported results.
- Baselines: `baseline_grud.py`, `baseline_mtand.py`, `baseline_transformer.py`, `baseline_lr_FIXED.py`.

## Environment
PyTorch 2.x, PyTorch Geometric 2.8, scikit-learn, pandas, pyarrow. Experiments were run on a single RTX 5090.

## License
BSD-3-Clause.

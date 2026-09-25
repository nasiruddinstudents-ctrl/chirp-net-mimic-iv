"""
CT-HEG / CHIRP-Net -- UNIFIED-PROTOCOL runs (no diagnosis), Vast.ai.
Author: Mohammad Nasir Uddin

Every condition uses the SAME protocol as train_chirp_nodx_vastai.py (the
full model): AdamW lr=3e-4 wd=3e-4, 500-step linear warmup + per-step cosine
over 80 epochs, max 80 epochs, early-stop patience 15 on val AUROC, best-val
checkpoint, WeightedRandomSampler, batch 32 (train) / 64 (eval), dropout 0.25,
aux LOS weight 0.05 (except no_aux), identical split (random_state=42).
Only the CONDITION-specific change differs.

CONDITION (env var):
  full                 heterogeneous, hidden 192 (the reported main model)
  no_aux               full, auxiliary LOS loss weight = 0
  vitlab_only          full, medication nodes/edges removed
  no_reverse           full, reverse edges never added
  edge_time_zeroed     full, edge-attr timestamp column = 0
  edge_value_zeroed    full, edge-attr value column = 0
  edge_time_permuted   full, edge-attr timestamp shuffled within graph/edge type
  homogeneous_small    relation-collapsed, hidden 192
  homogeneous_large    relation-collapsed, hidden 448
  heterogeneous_small  heterogeneous, hidden 80
  diagnose             no training; prints edge types and raw edge-attr stats

EDGE_NORM=zlog (default): edge value -> sign*log1p(|v|), z-scored per relation on
  training graphs, then clamped to [-10,10]. EDGE_NORM=clamp: legacy behaviour
  (raw value clamped to [-10,10]), identical to the original scripts.
SMOKE=1 -> small subset, 1 epoch, outputs to *_smoke dirs (pipeline check only).

Outputs:
  /workspace/models_unified/<COND>_seed<S>.pt
  /workspace/predictions_unified/<COND>_seed<S>_test_predictions.csv
  /workspace/results_unified/<COND>_seed<S>.csv
"""
import os, sys, time, json
import torch, torch.nn as nn, numpy as np, pandas as pd
from pathlib import Path
from torch_geometric.nn import HeteroConv, GATv2Conv
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import torch_geometric.transforms as T

COND  = os.environ.get("CONDITION", "full")
SEED  = int(os.environ.get("SEED", 42))
SMOKE = os.environ.get("SMOKE", "0") == "1"
EDGE_NORM = os.environ.get("EDGE_NORM", "zlog")  # "zlog" (fixed) or "clamp" (legacy: raw value clamped to [-10,10])
assert EDGE_NORM in ("zlog", "clamp")
CONDS = {  # name: (arch, hidden, aux_weight)
    "full": ("hetero", 192, 0.05), "no_aux": ("hetero", 192, 0.0),
    "vitlab_only": ("hetero", 192, 0.05), "no_reverse": ("hetero", 192, 0.05),
    "edge_time_zeroed": ("hetero", 192, 0.05), "edge_value_zeroed": ("hetero", 192, 0.05),
    "edge_time_permuted": ("hetero", 192, 0.05),
    "homogeneous_small": ("homog", 192, 0.05), "homogeneous_large": ("homog", 448, 0.05),
    "heterogeneous_small": ("hetero", 80, 0.05),
    # grid repeated without the auxiliary LOS loss (full-size hetero no-aux = "no_aux")
    "heterogeneous_small_noaux": ("hetero", 80, 0.0),
    "homogeneous_small_noaux": ("homog", 192, 0.0),
    "homogeneous_large_noaux": ("homog", 448, 0.0),
    # ablations repeated on the validation-selected small heterogeneous model (hidden 80)
    "vitlab_only_s80": ("hetero", 80, 0.05), "no_reverse_s80": ("hetero", 80, 0.05),
    "edge_time_zeroed_s80": ("hetero", 80, 0.05), "edge_value_zeroed_s80": ("hetero", 80, 0.05),
    "edge_time_permuted_s80": ("hetero", 80, 0.05),
}
MODE = COND[:-4] if COND.endswith("_s80") else COND  # behaviour key (vitlab_only, edge_*, no_reverse, ...)
assert COND in CONDS or COND == "diagnose", f"unknown CONDITION={COND}"

MAX_EPOCHS, PATIENCE, WARMUP = (1, 15, 500) if SMOKE else (80, 15, 500)

torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED); torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | CONDITION={COND} | SEED={SEED} | SMOKE={SMOKE} | EDGE_NORM={EDGE_NORM}", flush=True)

BASE = Path("/workspace"); GRAPHS_DIR = BASE / "graphs"
sfx = "_smoke" if SMOKE else ""
MODEL_DIR = BASE / f"models_unified{sfx}"; PRED_DIR = BASE / f"predictions_unified{sfx}"
RES_DIR = BASE / f"results_unified{sfx}"
for d in (MODEL_DIR, PRED_DIR, RES_DIR): d.mkdir(parents=True, exist_ok=True)

idx = pd.read_parquet(BASE / "graphs_index_clean.parquet")
cohort = pd.read_parquet(BASE / "cohort_clean.parquet")
label_map = dict(zip(idx["stay_id"], idx["mortality_label"]))
los_map = dict(zip(cohort["stay_id"], cohort["los_hours"]))

pos = idx[idx["mortality_label"] == 1]; neg = idx[idx["mortality_label"] == 0]
def split3(df):
    tr, tmp = train_test_split(df, test_size=0.30, random_state=42)
    va, te = train_test_split(tmp, test_size=0.50, random_state=42)
    return tr, va, te
ptr, pva, pte = split3(pos); ntr, nva, nte = split3(neg)
train_ids = pd.concat([ptr, ntr])["stay_id"].tolist()
val_ids = pd.concat([pva, nva])["stay_id"].tolist()
test_ids = pd.concat([pte, nte])["stay_id"].tolist()
if SMOKE or COND == "diagnose":
    # random mixed subsets (the split lists all positives first, so a head slice would be single-class)
    _r = np.random.RandomState(0)
    train_ids = list(_r.choice(train_ids, 1500, replace=False))
    val_ids = list(_r.choice(val_ids, 600, replace=False))
    test_ids = list(_r.choice(test_ids, 600, replace=False))
print(f"Train {len(train_ids):,} | Val {len(val_ids):,} | Test {len(test_ids):,}", flush=True)

def strip_types(g, names):
    for nt in names:
        if nt in g.node_types: del g[nt]
    for et in list(g.edge_types):
        if any(nt in et for nt in names): del g[et]
    return g

STRIP = ["diagnosis"] + (["med_event"] if MODE == "vitlab_only" else [])
t0 = time.time(); graphs_ram = {}
for sid in train_ids + val_ids + test_ids:
    try:
        g = torch.load(GRAPHS_DIR / f"{sid}.pt", map_location="cpu", weights_only=False)
        graphs_ram[sid] = strip_types(g, STRIP)
    except Exception as e:
        print(f"  load err {sid}: {str(e)[:60]}")
print(f"Loaded {len(graphs_ram):,} graphs in {time.time()-t0:.0f}s", flush=True)

if COND == "diagnose":
    g0 = next(iter(graphs_ram.values()))
    print("Stored (pre-ToUndirected) edge types:", [str(e) for e in g0.edge_types])
    for et in g0.edge_types:
        ei = g0[et].edge_index
        print(f"  {et}: edge_index row0 max={int(ei[0].max())} row1 max={int(ei[1].max())} "
              f"| n_src={g0[et[0]].num_nodes} n_dst={g0[et[2]].num_nodes}")
    stats = {}
    for g in graphs_ram.values():
        for et in g.edge_types:
            ea = g[et].edge_attr
            if ea is None or ea.shape[0] == 0: continue
            stats.setdefault(str(et), []).append(ea.float())
    for et, lst in stats.items():
        ea = torch.cat(lst); ea = torch.nan_to_num(ea, nan=float("nan"))
        print(f"\n{et}: dims={ea.shape[1]} rows={ea.shape[0]:,}")
        for c in range(ea.shape[1]):
            col = ea[:, c]; col = col[torch.isfinite(col)]
            q = torch.quantile(col, torch.tensor([0.01, 0.5, 0.99]))
            print(f"  col{c}: mean={col.mean():.3f} sd={col.std():.3f} "
                  f"p1={q[0]:.3f} median={q[1]:.3f} p99={q[2]:.3f} min={col.min():.3f} max={col.max():.3f}")
    sys.exit(0)

ARCH, HIDDEN, AUX_W = CONDS[COND]

vital_vals, lab_vals, med_vals = [], [], []
for sid in train_ids:
    g = graphs_ram.get(sid)
    if g is None: continue
    for nt, lst in [("vital", vital_vals), ("lab_event", lab_vals), ("med_event", med_vals)]:
        if nt in g.node_types and g[nt].x is not None and g[nt].x.shape[0] > 0:
            lst.append(g[nt].x)
def compute_stats(ts):
    if not ts: return None, None
    cat = torch.cat(ts, 0); cat = cat[torch.isfinite(cat).all(1)]
    return cat.mean(0), cat.std(0).clamp(min=1e-6)
vm, vs = compute_stats(vital_vals); lm, ls = compute_stats(lab_vals); mm, ms = compute_stats(med_vals)

def slog(v): return torch.sign(v) * torch.log1p(v.abs())
edge_stats = {}
if EDGE_NORM == "zlog":
    acc = {}
    for sid in train_ids:
        g = graphs_ram.get(sid)
        if g is None: continue
        for et in g.edge_types:
            ea = g[et].edge_attr
            if ea is None or ea.shape[0] == 0 or ea.shape[1] < 2: continue
            v = ea[:, 1].float(); v = v[torch.isfinite(v)]
            if v.numel(): acc.setdefault(et, []).append(slog(v))
    for et, lst in acc.items():
        v = torch.cat(lst); edge_stats[et] = (v.mean(), v.std().clamp(min=1e-6))
        print(f"  edge-value stats {et}: slog mean={edge_stats[et][0]:.3f} sd={edge_stats[et][1]:.3f} (n={v.numel():,})")

def perm_gen(sid, salt):
    g = torch.Generator(); g.manual_seed((SEED * 1_000_003 + int(sid) * 97 + salt) % (2**31 - 1)); return g

to_undirected = T.ToUndirected()
def normalize_graph(g, sid):
    def n(x, m, s):
        if m is None or x is None or x.shape[0] == 0: return x
        return ((x - m) / s).clamp(-10, 10)
    if "vital" in g.node_types: g["vital"].x = n(g["vital"].x, vm, vs)
    if "lab_event" in g.node_types: g["lab_event"].x = n(g["lab_event"].x, lm, ls)
    if "med_event" in g.node_types: g["med_event"].x = n(g["med_event"].x, mm, ms)
    for nt in g.node_types:
        if g[nt].x is not None:
            g[nt].x = torch.nan_to_num(g[nt].x, nan=0.0, posinf=3.0, neginf=-3.0)
    for salt, et in enumerate(list(g.edge_types)):
        ea = g[et].edge_attr
        if ea is None or ea.shape[0] == 0: continue
        ea = torch.nan_to_num(ea, nan=0.0)
        if ea.shape[1] >= 1: ea[:, 0] = (ea[:, 0] / 48.0).clamp(0, 1)
        if ea.shape[1] >= 2:
            if EDGE_NORM == "zlog" and et in edge_stats:
                m_, s_ = edge_stats[et]; ea[:, 1] = ((slog(ea[:, 1]) - m_) / s_).clamp(-10, 10)
            else:
                ea[:, 1] = ea[:, 1].clamp(-10, 10)
        if MODE == "edge_time_zeroed" and ea.shape[1] >= 1: ea[:, 0] = 0.0
        elif MODE == "edge_value_zeroed" and ea.shape[1] >= 2: ea[:, 1] = 0.0
        elif MODE == "edge_time_permuted" and ea.shape[0] > 1 and ea.shape[1] >= 1:
            p = torch.randperm(ea.shape[0], generator=perm_gen(sid, salt)); ea[:, 0] = ea[p, 0]
        g[et].edge_attr = ea
    if MODE != "no_reverse":
        g = to_undirected(g)
    g.y = torch.tensor([float(label_map.get(sid, 0))], dtype=torch.float)
    g.los = torch.tensor([float(los_map.get(sid, 48.0)) / 100.0], dtype=torch.float)
    return g

for sid in list(graphs_ram.keys()):
    graphs_ram[sid] = normalize_graph(graphs_ram[sid], sid)
sample_g = next(iter(graphs_ram.values()))
node_types = list(sample_g.node_types); edge_types = list(sample_g.edge_types)
print(f"Node types: {node_types}\nEdge types ({len(edge_types)}): {[str(e) for e in edge_types]}", flush=True)
assert "diagnosis" not in node_types
exp_et = {"no_reverse": 3, "vitlab_only": 4}.get(MODE, 6)
assert len(edge_types) == exp_et, f"expected {exp_et} edge types, got {len(edge_types)}"

train_graphs = [graphs_ram[s] for s in train_ids if s in graphs_ram]
val_graphs = [graphs_ram[s] for s in val_ids if s in graphs_ram]
test_graphs = [graphs_ram[s] for s in test_ids if s in graphs_ram]
test_ids_used = [s for s in test_ids if s in graphs_ram]
labels_arr = np.array([label_map[s] for s in train_ids if s in graphs_ram])
mr = labels_arr.mean()
weights = np.where(labels_arr == 1, 1.0 / mr, 1.0 / (1 - mr))
sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(weights, dtype=torch.float),
                                                 num_samples=len(train_graphs), replacement=True)
train_loader = DataLoader(train_graphs, batch_size=32, sampler=sampler, num_workers=0)
val_loader = DataLoader(val_graphs, batch_size=64, shuffle=False, num_workers=0)
test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False, num_workers=0)

FD = {"visit": 3, "vital": 6, "lab_event": 6, "med_event": 3, "diagnosis": 1}

def make_conv(h, d, ets):
    return HeteroConv({et: GATv2Conv((-1, -1), h // 4, heads=4, edge_dim=2, dropout=d,
                                     add_self_loops=False, concat=True) for et in ets}, aggr="sum")

class Hetero(nn.Module):
    """Identical to CHIRPNetNoDx in train_chirp_nodx_vastai.py."""
    def __init__(self, hidden, dropout, node_types, edge_types):
        super().__init__()
        self.enc = nn.ModuleDict({nt: nn.Sequential(nn.Linear(FD.get(nt, 4), hidden), nn.LayerNorm(hidden), nn.ReLU())
                                  for nt in node_types})
        self.type_emb = nn.Embedding(len(node_types), hidden)
        self.type_map = {nt: i for i, nt in enumerate(node_types)}
        self.convs = nn.ModuleList([make_conv(hidden, dropout, edge_types) for _ in range(4)])
        self.norms = nn.ModuleList([nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in node_types}) for _ in range(4)])
        self.drop = nn.Dropout(dropout)
        self.head_mort = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))
        self.head_los = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
    def forward(self, data):
        hd = {}
        for nt in self.enc:
            if nt not in data.node_types: continue
            x = data[nt].x
            if x is None or x.shape[0] == 0: continue
            hd[nt] = self.enc[nt](x) + self.type_emb(torch.tensor(self.type_map[nt], device=device))
        if "visit" not in hd: return torch.zeros(1, device=device), torch.zeros(1, device=device)
        ei_d, ea_d = {}, {}
        for et in data.edge_types:
            ei = data[et].edge_index
            if ei is None or ei.numel() == 0: continue
            ei_d[et] = ei
            ea = data[et].edge_attr
            if ea is not None and ea.shape[0] > 0:
                if ea.shape[1] < 2: ea = torch.cat([ea, torch.zeros(ea.shape[0], 2 - ea.shape[1], device=ea.device)], 1)
                ea_d[et] = ea[:, :2].float()
            else: ea_d[et] = torch.zeros(ei.shape[1], 2, device=device)
        for i, conv in enumerate(self.convs):
            hn = conv(hd, ei_d, ea_d)
            for nt in hd:
                if nt in hn and hn[nt] is not None:
                    hd[nt] = self.norms[i][nt](hd[nt] + self.drop(torch.nan_to_num(hn[nt], nan=0.0)))
        v = hd["visit"]
        return self.head_mort(v).squeeze(-1), self.head_los(v).squeeze(-1)

class Homog(nn.Module):
    """Identical to CHIRPHomogeneousNoDx in homogeneous_and_grid_nodx_vastai.py:
    type-specific encoders + type embedding kept; one shared GATv2Conv and one
    shared LayerNorm per layer across all relations (relation-collapsed)."""
    def __init__(self, hidden, dropout, node_types):
        super().__init__()
        self.node_types = node_types
        self.enc = nn.ModuleDict({nt: nn.Sequential(nn.Linear(FD.get(nt, 4), hidden), nn.LayerNorm(hidden), nn.ReLU())
                                  for nt in node_types})
        self.type_emb = nn.Embedding(len(node_types), hidden)
        self.type_map = {nt: i for i, nt in enumerate(node_types)}
        self.convs = nn.ModuleList([GATv2Conv(hidden, hidden // 4, heads=4, edge_dim=2, dropout=dropout,
                                              add_self_loops=False, concat=True) for _ in range(4)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(4)])
        self.drop = nn.Dropout(dropout)
        self.head_mort = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))
        self.head_los = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
    def forward(self, data):
        offsets, h_list, running = {}, [], 0
        for nt in self.node_types:
            if nt not in data.node_types: offsets[nt] = (running, running); continue
            x = data[nt].x; n = 0 if x is None else x.shape[0]
            if n == 0: offsets[nt] = (running, running); continue
            h_list.append(self.enc[nt](x) + self.type_emb(torch.tensor(self.type_map[nt], device=device)))
            offsets[nt] = (running, running + n); running += n
        if running == 0 or offsets.get("visit", (0, 0))[1] == offsets.get("visit", (0, 0))[0]:
            return torch.zeros(1, device=device), torch.zeros(1, device=device)
        h = torch.cat(h_list, 0); ei_list, ea_list = [], []
        for et in data.edge_types:
            s, _, d = et; ei = data[et].edge_index
            if ei is None or ei.numel() == 0 or s not in offsets or d not in offsets: continue
            eg = ei.clone(); eg[0] = ei[0] + offsets[s][0]; eg[1] = ei[1] + offsets[d][0]; ei_list.append(eg)
            ea = data[et].edge_attr
            if ea is not None and ea.shape[0] > 0:
                if ea.shape[1] < 2: ea = torch.cat([ea, torch.zeros(ea.shape[0], 2 - ea.shape[1], device=ea.device)], 1)
                ea_list.append(ea[:, :2].float())
            else: ea_list.append(torch.zeros(ei.shape[1], 2, device=device))
        if not ei_list: return torch.zeros(1, device=device), torch.zeros(1, device=device)
        edge_index = torch.cat(ei_list, 1); edge_attr = torch.cat(ea_list, 0)
        for i, conv in enumerate(self.convs):
            h = self.norms[i](h + self.drop(torch.nan_to_num(conv(h, edge_index, edge_attr), nan=0.0)))
        a, b = offsets["visit"]; v = h[a:b]
        return self.head_mort(v).squeeze(-1), self.head_los(v).squeeze(-1)

model = (Hetero(HIDDEN, 0.25, node_types, edge_types) if ARCH == "hetero" else Homog(HIDDEN, 0.25, node_types)).to(device)
with torch.no_grad(): _ = model(next(iter(train_loader)).to(device))
n_params = sum(p.numel() for p in model.parameters())
print(f"Arch={ARCH} hidden={HIDDEN} aux_w={AUX_W} params={n_params:,}", flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=3e-4)
total_steps = len(train_loader) * 80  # schedule horizon fixed at 80 epochs, as in the full model
warm = LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / WARMUP))
cos = CosineAnnealingLR(optimizer, T_max=max(total_steps - WARMUP, 1))
scheduler = SequentialLR(optimizer, [warm, cos], milestones=[WARMUP])
criterion = nn.BCEWithLogitsLoss()

def evaluate(loader):
    model.eval(); probs, labels = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device); mort, _ = model(b)
            for pi, yi in zip(torch.sigmoid(mort).cpu().numpy().flatten(), b.y.cpu().numpy().flatten()):
                if np.isfinite(pi): probs.append(float(pi)); labels.append(int(yi))
    if len(set(labels)) < 2: return float("nan"), float("nan")
    return roc_auc_score(labels, probs), average_precision_score(labels, probs)

def predict_with_ids(loader, ids):
    model.eval(); rows, ptr = [], 0
    with torch.no_grad():
        for b in loader:
            b = b.to(device); mort, _ = model(b)
            p = torch.sigmoid(mort).cpu().numpy().flatten(); y = b.y.cpu().numpy().flatten()
            assert len(p) == len(y) == b.num_graphs, "prediction/graph count mismatch"
            for sid, pi, yi in zip(ids[ptr:ptr + len(p)], p, y):
                rows.append((sid, int(yi), float(pi)))
            ptr += len(p)
    assert ptr == len(ids)
    return pd.DataFrame(rows, columns=["stay_id", "label", "prob"])

ckpt = MODEL_DIR / f"{COND}_seed{SEED}.pt"
best_auc, best_epoch, patience, total_skipped, t0 = 0.0, 0, 0, 0, time.time()
for epoch in range(1, MAX_EPOCHS + 1):
    model.train(); tl, cnt, skip = 0.0, 0, 0
    for b in train_loader:
        b = b.to(device); optimizer.zero_grad()
        try:
            mo, lo = model(b); ym = b.y.to(device).flatten(); mo = mo.flatten()
            n = min(mo.shape[0], ym.shape[0]); mo, ym = mo[:n], ym[:n]
            if not torch.isfinite(mo).all(): skip += 1; continue
            lm_ = criterion(mo, ym)
            loss = lm_ + AUX_W * nn.functional.mse_loss(lo.flatten()[:n], b.los.to(device).flatten()[:n]) if AUX_W > 0 else lm_
            if not torch.isfinite(loss): skip += 1; continue
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); tl += lm_.item(); cnt += 1
        except Exception as e:
            print(f"  batch err: {str(e)[:100]}"); skip += 1; continue
    total_skipped += skip
    va, vp = evaluate(val_loader)
    print(f"Epoch {epoch:3d} | loss {tl/max(cnt,1):.4f} | val AUROC {va:.4f} AUPRC {vp:.4f} | batches {cnt} skipped {skip}", flush=True)
    if np.isfinite(va) and va > best_auc:
        best_auc, best_epoch, patience = va, epoch, 0; torch.save(model.state_dict(), ckpt)
    else:
        patience += 1
        if patience >= PATIENCE: print(f"Early stop at epoch {epoch}"); break

model.load_state_dict(torch.load(ckpt))
ta, tp = evaluate(test_loader)
pred = predict_with_ids(test_loader, test_ids_used)
pred.to_csv(PRED_DIR / f"{COND}_seed{SEED}_test_predictions.csv", index=False)
pd.DataFrame([dict(condition=COND, seed=SEED, edge_norm=EDGE_NORM, arch=ARCH, hidden=HIDDEN, aux_weight=AUX_W, params=n_params,
                   best_epoch=best_epoch, epochs_run=epoch, best_val_auroc=best_auc, test_auroc=ta,
                   test_auprc=tp, skipped_batches=total_skipped, n_test=len(pred),
                   minutes=round((time.time() - t0) / 60, 1))]).to_csv(RES_DIR / f"{COND}_seed{SEED}.csv", index=False)
print(f"DONE {COND} seed {SEED}: test AUROC {ta:.4f} AUPRC {tp:.4f} | params {n_params:,} | "
      f"best epoch {best_epoch} | skipped {total_skipped}", flush=True)

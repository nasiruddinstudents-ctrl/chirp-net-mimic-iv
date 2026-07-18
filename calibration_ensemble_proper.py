"""
Proper calibration + ensemble statistics for CHIRP-Net.
- Fits ONE temperature on VAL logits only (never touches test during fitting)
- Applies that temperature to TEST logits (untouched until this point)
- Computes ensemble AUROC/AUPRC/Brier/ECE from averaged per-seed logits
  (distinct from the across-seed mean of five separate metrics)
- Reports patient-level bootstrap 95% CI on the ensemble metrics
"""
import torch, torch.nn as nn, numpy as np, pandas as pd, os
from pathlib import Path
from torch_geometric.nn import HeteroConv, GATv2Conv
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

GRAPHS_DIR = Path("/root/graphs")
MODEL_DIR  = Path("/root/models")

idx    = pd.read_parquet("/root/graphs_index_clean.parquet")
cohort = pd.read_parquet("/root/cohort_clean.parquet")
label_map = dict(zip(idx['stay_id'], idx['mortality_label']))
los_map   = dict(zip(cohort['stay_id'], cohort['los_hours']))

pos = idx[idx['mortality_label']==1]
neg = idx[idx['mortality_label']==0]
def split3(df):
    tr, tmp = train_test_split(df, test_size=0.30, random_state=42)
    va, te  = train_test_split(tmp, test_size=0.50, random_state=42)
    return tr, va, te
ptr,pva,pte = split3(pos); ntr,nva,nte = split3(neg)
train_ids = pd.concat([ptr,ntr])['stay_id'].tolist()
val_ids   = pd.concat([pva,nva])['stay_id'].tolist()
test_ids  = pd.concat([pte,nte])['stay_id'].tolist()
print(f"Train: {len(train_ids):,} Val: {len(val_ids):,} Test: {len(test_ids):,}")

print("Preloading raw graphs...")
graphs_ram = {}
for sid in idx['stay_id']:
    try:
        g = torch.load(GRAPHS_DIR / f"{sid}.pt", map_location='cpu', weights_only=False)
        graphs_ram[sid] = g
    except Exception:
        continue
print(f"Loaded: {len(graphs_ram):,}")

print("Computing normalization stats on train split (same convention as train_chirp_v5.py)...")
vital_vals, lab_vals, med_vals = [], [], []
for sid in train_ids:
    g = graphs_ram.get(sid)
    if g is None: continue
    for nt, lst in [('vital',vital_vals),('lab_event',lab_vals),('med_event',med_vals)]:
        if nt in g.node_types and g[nt].x is not None and g[nt].x.shape[0] > 0:
            lst.append(g[nt].x)

def compute_stats(tensors):
    if not tensors: return None, None
    cat = torch.cat(tensors, dim=0)
    cat = cat[torch.isfinite(cat).all(dim=1)]
    return cat.mean(dim=0), cat.std(dim=0).clamp(min=1e-6)

vital_mean, vital_std = compute_stats(vital_vals)
lab_mean, lab_std     = compute_stats(lab_vals)
med_mean, med_std     = compute_stats(med_vals)

to_undirected = T.ToUndirected()

def normalize_graph(g, sid):
    def norm(x, m, s):
        if m is None or x is None or x.shape[0] == 0: return x
        return ((x - m) / s).clamp(-10, 10)
    if 'vital'     in g.node_types: g['vital'].x     = norm(g['vital'].x,     vital_mean, vital_std)
    if 'lab_event' in g.node_types: g['lab_event'].x = norm(g['lab_event'].x, lab_mean,   lab_std)
    if 'med_event' in g.node_types: g['med_event'].x = norm(g['med_event'].x, med_mean,   med_std)
    for nt in g.node_types:
        if g[nt].x is not None:
            g[nt].x = torch.nan_to_num(g[nt].x, nan=0.0, posinf=3.0, neginf=-3.0)
    for et in g.edge_types:
        ea = g[et].edge_attr
        if ea is not None and ea.shape[0] > 0:
            ea = torch.nan_to_num(ea, nan=0.0)
            if ea.shape[1] >= 1: ea[:, 0] = (ea[:, 0] / 48.0).clamp(0, 1)
            if ea.shape[1] >= 2: ea[:, 1] = ea[:, 1].clamp(-10, 10)
            g[et].edge_attr = ea
    g = to_undirected(g)
    g.y   = torch.tensor([float(label_map.get(sid, 0))], dtype=torch.float)
    g.los = torch.tensor([float(los_map.get(sid, 48.0))/100.0], dtype=torch.float)
    return g

print("Normalizing...")
for sid in list(graphs_ram.keys()):
    graphs_ram[sid] = normalize_graph(graphs_ram[sid], sid)
print("Done.")

sample_g   = next(iter(graphs_ram.values()))
node_types = list(sample_g.node_types)
edge_types = list(sample_g.edge_types)
print(f"Node types: {node_types} | Edge types: {len(edge_types)}")

val_graphs  = [graphs_ram[s] for s in val_ids  if s in graphs_ram]
test_graphs = [graphs_ram[s] for s in test_ids if s in graphs_ram]
val_loader  = DataLoader(val_graphs,  batch_size=64, shuffle=False, num_workers=0)
test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False, num_workers=0)


def make_conv(h, d, ets):
    return HeteroConv({
        et: GATv2Conv((-1,-1), h//4, heads=4, edge_dim=2,
                     dropout=d, add_self_loops=False, concat=True)
        for et in ets}, aggr='sum')

class CHIRPNet(nn.Module):
    def __init__(self, hidden=192, dropout=0.25, node_types=None, edge_types=None):
        super().__init__()
        fd = {'visit':3,'vital':6,'lab_event':6,'med_event':3,'diagnosis':1}
        self.enc = nn.ModuleDict({
            nt: nn.Sequential(nn.Linear(fd.get(nt,4), hidden),
                             nn.LayerNorm(hidden), nn.ReLU())
            for nt in node_types})
        self.type_emb = nn.Embedding(len(node_types), hidden)
        self.type_map = {nt: i for i,nt in enumerate(node_types)}
        self.convs = nn.ModuleList([make_conv(hidden, dropout, edge_types) for _ in range(4)])
        self.norms = nn.ModuleList([
            nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in node_types})
            for _ in range(4)])
        self.drop = nn.Dropout(dropout)
        self.head_mort = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden//2, 1))
        self.head_los = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Linear(hidden//2, 1))

    def forward(self, data):
        hd = {}
        for nt in self.enc:
            if nt not in data.node_types: continue
            x = data[nt].x
            if x is None or x.shape[0]==0: continue
            hd[nt] = self.enc[nt](x) + self.type_emb(
                torch.tensor(self.type_map[nt], device=device))
        if 'visit' not in hd:
            return torch.zeros(1,device=device), torch.zeros(1,device=device)
        ei_d, ea_d = {}, {}
        for et in data.edge_types:
            ei = data[et].edge_index
            if ei is None or ei.numel()==0: continue
            ei_d[et] = ei
            ea = data[et].edge_attr
            if ea is not None and ea.shape[0]>0:
                if ea.shape[1]<2:
                    ea = torch.cat([ea,torch.zeros(ea.shape[0],2-ea.shape[1],device=ea.device)],dim=1)
                ea_d[et] = ea[:,:2].float()
            else: ea_d[et] = torch.zeros(ei.shape[1],2,device=device)
        ea_d = {et:ea_d[et] for et in ei_d if et in ea_d}
        for i,conv in enumerate(self.convs):
            hn = conv(hd, ei_d, ea_d)
            for nt in hd:
                if nt in hn and hn[nt] is not None:
                    hd[nt] = self.norms[i][nt](hd[nt]+self.drop(
                        torch.nan_to_num(hn[nt], nan=0.0)))
        v = hd['visit']
        return self.head_mort(v).squeeze(-1), self.head_los(v).squeeze(-1)


@torch.no_grad()
def get_logits(model, loader):
    model.eval()
    logits, labels = [], []
    for b in loader:
        b = b.to(device)
        mort, _ = model(b)
        logits.append(mort.cpu())
        labels.append(b.y.cpu())
    return torch.cat(logits).flatten(), torch.cat(labels).flatten()


SEEDS = [42, 43, 44, 45, 46]
val_logits_per_seed, test_logits_per_seed = [], []
val_labels, test_labels = None, None

for seed in SEEDS:
    print(f"Loading seed {seed}...")
    model = CHIRPNet(hidden=192, dropout=0.25, node_types=node_types, edge_types=edge_types).to(device)
    with torch.no_grad():
        _ = model(next(iter(val_loader)).to(device))
    state = torch.load(MODEL_DIR / f"best_chirp_v5_seed{seed}.pt", map_location=device)
    model.load_state_dict(state)

    v_logits, v_labels = get_logits(model, val_loader)
    t_logits, t_labels = get_logits(model, test_loader)
    val_logits_per_seed.append(v_logits)
    test_logits_per_seed.append(t_logits)
    if val_labels is None:
        val_labels, test_labels = v_labels, t_labels

val_logits_stack  = torch.stack(val_logits_per_seed)
test_logits_stack = torch.stack(test_logits_per_seed)

ensemble_val_logit  = val_logits_stack.mean(dim=0)
ensemble_test_logit = test_logits_stack.mean(dim=0)

print("\nFitting temperature scaling on VAL set only...")
T_param = torch.nn.Parameter(torch.ones(1) * 1.0)
optimizer = torch.optim.LBFGS([T_param], lr=0.01, max_iter=200)
criterion = nn.BCEWithLogitsLoss()

def closure():
    optimizer.zero_grad()
    loss = criterion(ensemble_val_logit / T_param, val_labels)
    loss.backward()
    return loss

optimizer.step(closure)
T_fitted = T_param.item()
print(f"Fitted temperature T = {T_fitted:.4f} (fit on val, N={len(val_labels)})")

calibrated_test_logit = ensemble_test_logit / T_fitted
calibrated_test_prob  = torch.sigmoid(calibrated_test_logit).numpy()
raw_test_prob         = torch.sigmoid(ensemble_test_logit).numpy()
test_labels_np        = test_labels.numpy().astype(int)

def brier_score(probs, labels):
    return np.mean((probs - labels) ** 2)

def expected_calibration_error(probs, labels, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins-1 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0: continue
        conf = probs[mask].mean()
        acc  = labels[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(conf - acc)
    return ece

ens_auroc = roc_auc_score(test_labels_np, raw_test_prob)
ens_auprc = average_precision_score(test_labels_np, raw_test_prob)
brier_before = brier_score(raw_test_prob, test_labels_np)
brier_after  = brier_score(calibrated_test_prob, test_labels_np)
ece_before   = expected_calibration_error(raw_test_prob, test_labels_np)
ece_after    = expected_calibration_error(calibrated_test_prob, test_labels_np)

print("\n" + "="*60)
print("ENSEMBLE RESULTS (averaged logits across 5 seeds, test set)")
print("="*60)
print(f"Ensemble AUROC: {ens_auroc:.4f}")
print(f"Ensemble AUPRC: {ens_auprc:.4f}")
print(f"Brier score  — before calibration: {brier_before:.4f} | after (T={T_fitted:.3f}): {brier_after:.4f}")
print(f"ECE (15 bins)— before calibration: {ece_before:.4f} | after (T={T_fitted:.3f}): {ece_after:.4f}")

print("\nBootstrapping 95% CI (1000 resamples, patient-level, test set)...")
n = len(test_labels_np)
rng = np.random.default_rng(42)
boot_auroc, boot_auprc, boot_brier, boot_ece = [], [], [], []
for _ in range(1000):
    idxs = rng.integers(0, n, n)
    yb, pb = test_labels_np[idxs], calibrated_test_prob[idxs]
    if len(set(yb)) < 2:
        continue
    boot_auroc.append(roc_auc_score(yb, pb))
    boot_auprc.append(average_precision_score(yb, pb))
    boot_brier.append(brier_score(pb, yb))
    boot_ece.append(expected_calibration_error(pb, yb))

def ci(arr):
    return np.percentile(arr, 2.5), np.percentile(arr, 97.5)

print(f"AUROC 95% CI: {ci(boot_auroc)}")
print(f"AUPRC 95% CI: {ci(boot_auprc)}")
print(f"Brier (post-calibration) 95% CI: {ci(boot_brier)}")
print(f"ECE (post-calibration) 95% CI: {ci(boot_ece)}")

pd.DataFrame([{
    'temperature_fitted_on_val': T_fitted,
    'ensemble_auroc_test': ens_auroc,
    'ensemble_auprc_test': ens_auprc,
    'brier_pre_calibration': brier_before,
    'brier_post_calibration': brier_after,
    'ece_pre_calibration': ece_before,
    'ece_post_calibration': ece_after,
    'auroc_ci_lo': ci(boot_auroc)[0], 'auroc_ci_hi': ci(boot_auroc)[1],
    'auprc_ci_lo': ci(boot_auprc)[0], 'auprc_ci_hi': ci(boot_auprc)[1],
    'brier_ci_lo': ci(boot_brier)[0], 'brier_ci_hi': ci(boot_brier)[1],
    'ece_ci_lo': ci(boot_ece)[0], 'ece_ci_hi': ci(boot_ece)[1],
}]).to_csv('/root/results/calibration_ensemble_proper.csv', index=False)
print("\nSaved: /root/results/calibration_ensemble_proper.csv")

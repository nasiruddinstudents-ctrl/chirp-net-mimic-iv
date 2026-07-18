"""
CHIRP Training v5 - CRITICAL FIX: reverse edges + lazy init
Author: Mohammad Nasir Uddin
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from torch_geometric.nn import HeteroConv, GATv2Conv
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import torch_geometric.transforms as T

import os
SEED = int(os.environ.get("SEED", 42))
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

GRAPHS_DIR = Path("/root/graphs")
MODEL_DIR  = Path("/root/models")
MODEL_DIR.mkdir(exist_ok=True)

idx    = pd.read_parquet("/root/graphs_index_clean.parquet")
cohort = pd.read_parquet("/root/cohort_clean.parquet")
print(f"Total: {len(idx):,} | Mortality: {idx['mortality_label'].mean()*100:.1f}%")

label_map = dict(zip(idx['stay_id'], idx['mortality_label']))
los_map   = dict(zip(cohort['stay_id'], cohort['los_hours']))

# Stratified split
pos = idx[idx['mortality_label']==1]
neg = idx[idx['mortality_label']==0]
def split3(df):
    tr, tmp = train_test_split(df, test_size=0.30, random_state=42)
    va, te  = train_test_split(tmp, test_size=0.50, random_state=42)
    return tr, va, te
ptr,pva,pte = split3(pos)
ntr,nva,nte = split3(neg)
train_ids = pd.concat([ptr,ntr])['stay_id'].tolist()
val_ids   = pd.concat([pva,nva])['stay_id'].tolist()
test_ids  = pd.concat([pte,nte])['stay_id'].tolist()
print(f"Train: {len(train_ids):,} Val: {len(val_ids):,} Test: {len(test_ids):,}")

# ── Preload ──────────────────────────────────────────────────────────────────
print("Preloading graphs...")
graphs_ram = {}
for sid in idx['stay_id']:
    try:
        g = torch.load(GRAPHS_DIR / f"{sid}.pt",
                      map_location='cpu', weights_only=False)
        graphs_ram[sid] = g
    except Exception as e:
        print(f"  load err {sid}: {str(e)[:60]}")
print(f"Loaded: {len(graphs_ram):,}")

# ── Normalization ────────────────────────────────────────────────────────────
print("Computing normalization stats on full train split...")
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
lab_mean,   lab_std   = compute_stats(lab_vals)
med_mean,   med_std   = compute_stats(med_vals)

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

    # CRITICAL FIX: add reverse edges so visit receives messages
    g = to_undirected(g)

    g.y   = torch.tensor([float(label_map.get(sid, 0))], dtype=torch.float)
    g.los = torch.tensor([float(los_map.get(sid, 48.0))/100.0], dtype=torch.float)
    return g

print("Normalizing + adding reverse edges...")
for sid in list(graphs_ram.keys()):
    graphs_ram[sid] = normalize_graph(graphs_ram[sid], sid)
print("Done.")

# Get metadata AFTER normalization (includes reverse edges)
sample_g   = next(iter(graphs_ram.values()))
node_types = list(sample_g.node_types)
edge_types = list(sample_g.edge_types)
print(f"Node types ({len(node_types)}): {node_types}")
print(f"Edge types ({len(edge_types)}): {[str(et) for et in edge_types]}")

# ── DataLoader ───────────────────────────────────────────────────────────────
train_graphs = [graphs_ram[s] for s in train_ids if s in graphs_ram]
val_graphs   = [graphs_ram[s] for s in val_ids   if s in graphs_ram]
test_graphs  = [graphs_ram[s] for s in test_ids  if s in graphs_ram]

labels_arr = np.array([label_map[s] for s in train_ids if s in graphs_ram])
mort_rate  = labels_arr.mean()
w_pos = 1.0 / mort_rate
w_neg = 1.0 / (1 - mort_rate)
weights = np.where(labels_arr == 1, w_pos, w_neg)
sampler = torch.utils.data.WeightedRandomSampler(
    torch.tensor(weights, dtype=torch.float),
    num_samples=len(train_graphs), replacement=True)

BATCH = 32
train_loader = DataLoader(train_graphs, batch_size=BATCH,
                          sampler=sampler, num_workers=0)
val_loader   = DataLoader(val_graphs,   batch_size=64,
                          shuffle=False, num_workers=0)
test_loader  = DataLoader(test_graphs,  batch_size=64,
                          shuffle=False, num_workers=0)
print(f"Train batches: {len(train_loader)}")

# ── Model ────────────────────────────────────────────────────────────────────
def make_hetero_conv(hidden, dropout, edge_types):
    return HeteroConv({
        et: GATv2Conv((-1,-1), hidden//4, heads=4,
                     edge_dim=2, dropout=dropout,
                     add_self_loops=False, concat=True)
        for et in edge_types
    }, aggr='sum')

class CHIRPNet(nn.Module):
    def __init__(self, hidden=192, dropout=0.25,
                 node_types=None, edge_types=None):
        super().__init__()
        self.hidden = hidden
        feat_dims = {'visit':3,'vital':6,'lab_event':6,
                    'med_event':3,'diagnosis':1}
        self.enc = nn.ModuleDict({
            nt: nn.Sequential(
                nn.Linear(feat_dims.get(nt,4), hidden),
                nn.LayerNorm(hidden), nn.ReLU())
            for nt in node_types
        })
        self.type_emb = nn.Embedding(len(node_types), hidden)
        self.type_map = {nt: i for i,nt in enumerate(node_types)}

        self.convs = nn.ModuleList([
            make_hetero_conv(hidden, dropout, edge_types)
            for _ in range(4)
        ])
        self.norms = nn.ModuleList([
            nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in node_types})
            for _ in range(4)
        ])
        self.drop = nn.Dropout(dropout)

        self.head_mort = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden//2, 1))
        self.head_los = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.GELU(),
            nn.Linear(hidden//2, 1))

    def forward(self, data):
        h_dict = {}
        for nt in self.enc:
            if nt not in data.node_types: continue
            x = data[nt].x
            if x is None or x.shape[0] == 0: continue
            h = self.enc[nt](x)
            h = h + self.type_emb(
                torch.tensor(self.type_map[nt], device=device))
            h_dict[nt] = h

        if 'visit' not in h_dict:
            return torch.zeros(1,device=device), torch.zeros(1,device=device)

        # Build edge dicts
        edge_index_dict = {}
        edge_attr_dict  = {}
        for et in data.edge_types:
            ei = data[et].edge_index
            if ei is None or ei.numel() == 0: continue
            edge_index_dict[et] = ei
            ea = data[et].edge_attr
            if ea is not None and ea.shape[0] > 0:
                if ea.shape[1] < 2:
                    ea = torch.cat([ea,
                        torch.zeros(ea.shape[0], 2-ea.shape[1],
                                   device=ea.device)], dim=1)
                edge_attr_dict[et] = ea[:,:2].float()
            else:
                edge_attr_dict[et] = torch.zeros(
                    ei.shape[1], 2, device=device)

        # 4 HeteroConv layers
        for layer_idx, conv in enumerate(self.convs):
            h_new = conv(h_dict, edge_index_dict, edge_attr_dict)
            for nt in h_dict:
                if nt in h_new and h_new[nt] is not None:
                    v = torch.nan_to_num(h_new[nt], nan=0.0)
                    h_dict[nt] = self.norms[layer_idx][nt](
                        h_dict[nt] + self.drop(v))

        visit_emb = h_dict['visit']
        return (self.head_mort(visit_emb).squeeze(-1),
                self.head_los(visit_emb).squeeze(-1))

model = CHIRPNet(hidden=192, dropout=0.25,
                node_types=node_types,
                edge_types=edge_types).to(device)

# Initialize lazy modules with dummy batch
print("Initializing lazy modules...")
with torch.no_grad():
    dummy = next(iter(train_loader)).to(device)
    _ = model(dummy)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-4, weight_decay=3e-4)

total_steps  = len(train_loader) * 80
warmup_steps = 500
warmup_sched = LambdaLR(optimizer,
    lambda s: min(1.0, (s+1) / warmup_steps))
cosine_sched = CosineAnnealingLR(optimizer,
    T_max=max(total_steps - warmup_steps, 1))
scheduler = SequentialLR(optimizer,
    [warmup_sched, cosine_sched], milestones=[warmup_steps])

criterion = nn.BCEWithLogitsLoss()

def evaluate(loader):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            mort, _ = model(batch)
            p = torch.sigmoid(mort).cpu().numpy().flatten()
            y = batch.y.cpu().numpy().flatten()
            for pi, yi in zip(p, y):
                if np.isfinite(pi):
                    probs.append(float(pi))
                    labels.append(int(yi))
    probs  = np.array(probs)
    labels = np.array(labels)
    mask   = np.isfinite(probs)
    if mask.sum() < 50 or len(set(labels[mask])) < 2:
        return float('nan'), float('nan')
    return (roc_auc_score(labels[mask], probs[mask]),
            average_precision_score(labels[mask], probs[mask]))

log_rows = []
print("\n" + "="*60)
print("CHIRP v5 — Reverse Edges Fixed, Visit Receives Messages")
print("="*60)

best_auc = 0
patience_count = 0

for epoch in range(1, 81):
    model.train()
    total_loss, count, nan_skip = 0, 0, 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        try:
            mort_out, los_out = model(batch)
            y_mort   = batch.y.to(device).flatten()
            mort_out = mort_out.flatten()
            n = min(mort_out.shape[0], y_mort.shape[0])
            mort_out, y_mort = mort_out[:n], y_mort[:n]

            if not torch.isfinite(mort_out).all():
                nan_skip += 1
                continue

            loss_mort = criterion(mort_out, y_mort)
            if hasattr(batch, 'los') and batch.los is not None:
                y_los   = batch.los.to(device).flatten()[:n]
                los_out = los_out.flatten()[:n]
                loss    = loss_mort + 0.05 * nn.functional.mse_loss(los_out, y_los)
            else:
                loss = loss_mort

            if not torch.isfinite(loss):
                nan_skip += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss_mort.item()
            count += 1
        except Exception as e:
            print(f"  batch err: {str(e)[:100]}")
            continue

    val_auc, val_auprc = evaluate(val_loader)
    avg_loss = total_loss / max(count, 1)
    log_rows.append({'epoch':epoch,'loss':avg_loss,
                    'val_auc':val_auc,'val_auprc':val_auprc})
    pd.DataFrame(log_rows).to_csv('/root/training_log_v5.csv', index=False)

    print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
          f"AUROC: {val_auc:.4f} | AUPRC: {val_auprc:.4f} | "
          f"Batches: {count} | NaN: {nan_skip}")

    if np.isfinite(val_auc) and val_auc > best_auc:
        best_auc = val_auc
        patience_count = 0
        torch.save(model.state_dict(), MODEL_DIR/f'best_chirp_v5_seed{SEED}.pt')
        print(f"           *** Best AUROC: {best_auc:.4f} ***")
    else:
        patience_count += 1
        if patience_count >= 15:
            print(f"Early stopping at epoch {epoch}")
            break

print(f"\nBest Val AUROC: {best_auc:.4f}")
model.load_state_dict(torch.load(MODEL_DIR/f'best_chirp_v5_seed{SEED}.pt'))
test_auc, test_auprc = evaluate(test_loader)
print(f"Test AUROC:  {test_auc:.4f}")
print(f"Test AUPRC:  {test_auprc:.4f}")
print(f"\nThis is your CHIRP paper result!")

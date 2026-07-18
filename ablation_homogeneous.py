"""
Ablation: CHIRP-Net with homogeneous message passing
(heterogeneous edge types collapsed to a single relation)
Author: Mohammad Nasir Uddin
"""
import torch, torch.nn as nn, numpy as np, pandas as pd, os
from pathlib import Path
from torch_geometric.nn import GATv2Conv
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SEED = int(os.environ.get('SEED', 42))
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)

GRAPHS_DIR = Path("/root/graphs")
MODEL_DIR  = Path("/root/models/ablations")
MODEL_DIR.mkdir(exist_ok=True, parents=True)

idx       = pd.read_parquet("/root/graphs_index_clean.parquet")
cohort    = pd.read_parquet("/root/cohort_clean.parquet")
label_map = dict(zip(idx['stay_id'], idx['mortality_label']))
los_map   = dict(zip(cohort['stay_id'], cohort['los_hours']))

pos = idx[idx['mortality_label']==1]
neg = idx[idx['mortality_label']==0]
def split3(df):
    tr,tmp = train_test_split(df, test_size=0.30, random_state=42)
    va,te  = train_test_split(tmp, test_size=0.50, random_state=42)
    return tr,va,te
ptr,pva,pte = split3(pos); ntr,nva,nte = split3(neg)
train_ids = pd.concat([ptr,ntr])['stay_id'].tolist()
val_ids   = pd.concat([pva,nva])['stay_id'].tolist()
test_ids  = pd.concat([pte,nte])['stay_id'].tolist()

to_undirected = T.ToUndirected()

print("Preloading raw graphs (homogeneous-GAT ablation)...")
graphs_ram = {}
for sid in idx['stay_id']:
    try:
        g = torch.load(GRAPHS_DIR/f"{sid}.pt", map_location='cpu', weights_only=False)
        graphs_ram[sid] = g
    except Exception as e:
        continue
print(f"Loaded: {len(graphs_ram):,}")

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

vm, vs = compute_stats(vital_vals)
lm, ls = compute_stats(lab_vals)
mm, ms = compute_stats(med_vals)

def norm_g(g, sid):
    def n(x,m,s):
        if m is None or x is None or x.shape[0]==0: return x
        return ((x-m)/s).clamp(-10,10)
    if 'vital'     in g.node_types: g['vital'].x     = n(g['vital'].x,vm,vs)
    if 'lab_event' in g.node_types: g['lab_event'].x = n(g['lab_event'].x,lm,ls)
    if 'med_event' in g.node_types: g['med_event'].x = n(g['med_event'].x,mm,ms)
    for nt in g.node_types:
        if g[nt].x is not None:
            g[nt].x = torch.nan_to_num(g[nt].x, nan=0.0, posinf=3.0, neginf=-3.0)
    for et in g.edge_types:
        ea = g[et].edge_attr
        if ea is not None and ea.shape[0]>0:
            ea = torch.nan_to_num(ea, nan=0.0)
            if ea.shape[1]>=1: ea[:,0]=(ea[:,0]/48.0).clamp(0,1)
            if ea.shape[1]>=2: ea[:,1]=ea[:,1].clamp(-10,10)
            g[et].edge_attr = ea
    g = to_undirected(g)
    g.y   = torch.tensor([float(label_map.get(sid,0))], dtype=torch.float)
    g.los = torch.tensor([float(los_map.get(sid,48.0))/100.0], dtype=torch.float)
    return g

print("Normalizing (reverse edges kept, edge types will be collapsed in-model)...")
for sid in list(graphs_ram.keys()):
    graphs_ram[sid] = norm_g(graphs_ram[sid], sid)
print("Done.")

sample_g   = next(iter(graphs_ram.values()))
node_types = list(sample_g.node_types)
edge_types = list(sample_g.edge_types)
print(f"Node types: {node_types}")
print(f"Edge types: {len(edge_types)} (will be merged into 1 relation for message passing)")

train_graphs = [graphs_ram[s] for s in train_ids if s in graphs_ram]
val_graphs   = [graphs_ram[s] for s in val_ids   if s in graphs_ram]
test_graphs  = [graphs_ram[s] for s in test_ids  if s in graphs_ram]

labels_arr = np.array([label_map[s] for s in train_ids if s in graphs_ram])
w_pos = 1.0/labels_arr.mean(); w_neg = 1.0/(1-labels_arr.mean())
weights = np.where(labels_arr==1, w_pos, w_neg)
sampler = torch.utils.data.WeightedRandomSampler(
    torch.tensor(weights, dtype=torch.float), len(train_graphs), replacement=True)

train_loader = DataLoader(train_graphs, batch_size=32, sampler=sampler, num_workers=0)
val_loader   = DataLoader(val_graphs,   batch_size=64, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_graphs,  batch_size=64, shuffle=False, num_workers=0)


class CHIRPHomogeneous(nn.Module):
    def __init__(self, hidden=192, dropout=0.25, node_types=None):
        super().__init__()
        fd = {'visit':3,'vital':6,'lab_event':6,'med_event':3,'diagnosis':1}
        self.hidden = hidden
        self.node_types = node_types
        self.enc = nn.ModuleDict({
            nt: nn.Sequential(nn.Linear(fd.get(nt,4), hidden),
                             nn.LayerNorm(hidden), nn.ReLU())
            for nt in node_types})
        self.type_emb = nn.Embedding(len(node_types), hidden)
        self.type_map = {nt: i for i, nt in enumerate(node_types)}

        self.convs = nn.ModuleList([
            GATv2Conv(hidden, hidden // 4, heads=4, edge_dim=2,
                      dropout=dropout, add_self_loops=False, concat=True)
            for _ in range(4)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(4)])
        self.drop = nn.Dropout(dropout)
        self.head_mort = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden//2, 1))
        self.head_los = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Linear(hidden//2, 1))

    def forward(self, data):
        offsets = {}
        h_list = []
        running = 0
        for nt in self.node_types:
            if nt not in data.node_types:
                offsets[nt] = (running, running)
                continue
            x = data[nt].x
            n = 0 if x is None else x.shape[0]
            if n == 0:
                offsets[nt] = (running, running)
                continue
            h = self.enc[nt](x) + self.type_emb(
                torch.tensor(self.type_map[nt], device=device))
            h_list.append(h)
            offsets[nt] = (running, running + n)
            running += n

        if running == 0 or offsets.get('visit', (0,0))[1] == offsets.get('visit', (0,0))[0]:
            return torch.zeros(1, device=device), torch.zeros(1, device=device)

        h_all = torch.cat(h_list, dim=0)

        ei_list, ea_list = [], []
        for et in data.edge_types:
            src_type, _, dst_type = et
            ei = data[et].edge_index
            if ei is None or ei.numel() == 0: continue
            if src_type not in offsets or dst_type not in offsets: continue
            src_off = offsets[src_type][0]
            dst_off = offsets[dst_type][0]
            ei_global = ei.clone()
            ei_global[0] = ei[0] + src_off
            ei_global[1] = ei[1] + dst_off
            ei_list.append(ei_global)

            ea = data[et].edge_attr
            if ea is not None and ea.shape[0] > 0:
                if ea.shape[1] < 2:
                    ea = torch.cat([ea, torch.zeros(ea.shape[0], 2-ea.shape[1], device=ea.device)], dim=1)
                ea_list.append(ea[:, :2].float())
            else:
                ea_list.append(torch.zeros(ei.shape[1], 2, device=device))

        if not ei_list:
            return torch.zeros(1, device=device), torch.zeros(1, device=device)

        edge_index = torch.cat(ei_list, dim=1)
        edge_attr  = torch.cat(ea_list, dim=0)

        h = h_all
        for i, conv in enumerate(self.convs):
            hn = conv(h, edge_index, edge_attr)
            hn = torch.nan_to_num(hn, nan=0.0)
            h = self.norms[i](h + self.drop(hn))

        vs_, ve_ = offsets['visit']
        v = h[vs_:ve_]
        return self.head_mort(v).squeeze(-1), self.head_los(v).squeeze(-1)


model = CHIRPHomogeneous(hidden=192, dropout=0.25, node_types=node_types).to(device)
with torch.no_grad():
    _ = model(next(iter(train_loader)).to(device))
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
criterion = nn.BCEWithLogitsLoss()

full_aurocs = {42:0.8514, 43:0.8463, 44:0.8324, 45:0.8429, 46:0.8517}

def evaluate(loader):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            mort, _ = model(b)
            p = torch.sigmoid(mort).cpu().numpy().flatten()
            y = b.y.cpu().numpy().flatten()
            for pi,yi in zip(p,y):
                if np.isfinite(pi): probs.append(float(pi)); labels.append(int(yi))
    if len(set(labels))<2: return float('nan'), float('nan')
    return roc_auc_score(labels,probs), average_precision_score(labels,probs)

print(f"\n{'='*55}")
print(f"Ablation: HOMOGENEOUS GAT (edge types collapsed) | Seed {SEED}")
print(f"{'='*55}")

best_auc, patience = 0, 0
for epoch in range(1, 51):
    model.train()
    total_loss, count, skipped = 0, 0, 0
    for b in train_loader:
        try:
            b = b.to(device)
            optimizer.zero_grad()
            mort_out, los_out = model(b)
            y_mort = b.y.to(device).flatten()
            mort_out = mort_out.flatten()
            n = min(mort_out.shape[0], y_mort.shape[0])
            loss_mort = criterion(mort_out[:n], y_mort[:n])
            if hasattr(b, 'los') and b.los is not None:
                y_los = b.los.to(device).flatten()[:n]
                loss = loss_mort + 0.05 * nn.functional.mse_loss(los_out.flatten()[:n], y_los)
            else:
                loss = loss_mort
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss_mort.item(); count += 1
        except Exception as e:
            skipped += 1
            continue
    scheduler.step()
    val_auc, _ = evaluate(val_loader)
    print(f"Epoch {epoch:3d} | Loss: {total_loss/max(count,1):.4f} | Val AUROC: {val_auc:.4f} | Skipped: {skipped}")
    if np.isfinite(val_auc) and val_auc > best_auc:
        best_auc = val_auc
        patience = 0
        torch.save(model.state_dict(), MODEL_DIR/f'ablation_homogeneous_seed{SEED}.pt')
        print(f"           *** Best: {best_auc:.4f} ***")
    else:
        patience += 1
        if patience >= 8:
            print(f"Early stopping at epoch {epoch}")
            break

print(f"\nBest Val AUROC: {best_auc:.4f}")
model.load_state_dict(torch.load(MODEL_DIR/f'ablation_homogeneous_seed{SEED}.pt'))
test_auc, test_ap = evaluate(test_loader)
full_auc = full_aurocs.get(SEED, 0.8449)
print(f"Test AUROC:  {test_auc:.4f}")
print(f"Test AUPRC:  {test_ap:.4f}")
print(f"Full CHIRP-Net seed {SEED}: {full_auc:.4f}")
print(f"Delta (heterogeneous edge types): +{full_auc-test_auc:.4f}")

pd.DataFrame([{
    'ablation': 'homogeneous_gat',
    'seed': SEED,
    'test_auroc': test_auc,
    'test_auprc': test_ap,
    'full_auroc': full_auc,
    'delta': full_auc - test_auc,
}]).to_csv(f'/root/results/ablation_homogeneous_seed{SEED}.csv', index=False)
print(f"Saved: /root/results/ablation_homogeneous_seed{SEED}.csv")
print(f"\nThis is your CHIRP homogeneous-GAT ablation result!")

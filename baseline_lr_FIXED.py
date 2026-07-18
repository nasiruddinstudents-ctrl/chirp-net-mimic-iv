import torch, pandas as pd, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

GRAPHS_DIR = Path("/root/graphs")
idx    = pd.read_parquet("/root/graphs_index_clean.parquet")
label_map = dict(zip(idx['stay_id'], idx['mortality_label']))

print("Extracting features...")
features, labels = [], []
for sid in idx['stay_id']:
    try:
        g = torch.load(GRAPHS_DIR/f"{sid}.pt", map_location='cpu', weights_only=False)
        feats = []
        if 'visit' in g.node_types and g['visit'].x is not None:
            feats.extend(g['visit'].x[0].tolist())
        else:
            feats.extend([0.0]*3)
        for nt, dim in [('vital',6),('lab_event',6)]:
            if nt in g.node_types and g[nt].x is not None and g[nt].x.shape[0]>0:
                v = g[nt].x
                feats.extend(v.mean(0).tolist()+v.std(0).tolist()+
                             v.min(0).values.tolist()+v.max(0).values.tolist())
                feats.append(float(v.shape[0]))
            else:
                feats.extend([0.0]*(dim*4+1))
        features.append(feats)
        labels.append(int(label_map[sid]))
    except:
        continue

X = np.nan_to_num(np.array(features))
y = np.array(labels)
print(f"Features: {X.shape}")

pos_idx = np.where(y==1)[0]
neg_idx = np.where(y==0)[0]
def split3(a):
    tr,tmp = train_test_split(a, test_size=0.30, random_state=42)
    va,te  = train_test_split(tmp, test_size=0.50, random_state=42)
    return tr,va,te
ptr,pva,pte = split3(pos_idx)
ntr,nva,nte = split3(neg_idx)
tr = np.concatenate([ptr,ntr])
va = np.concatenate([pva,nva])
te = np.concatenate([pte,nte])

sc = StandardScaler()
Xtr = sc.fit_transform(X[tr])
Xva = sc.transform(X[va])
Xte = sc.transform(X[te])

print("Training LR...")
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(Xtr, y[tr])
va_auc = roc_auc_score(y[va], lr.predict_proba(Xva)[:,1])
te_auc = roc_auc_score(y[te], lr.predict_proba(Xte)[:,1])
va_ap  = average_precision_score(y[va], lr.predict_proba(Xva)[:,1])
te_ap  = average_precision_score(y[te], lr.predict_proba(Xte)[:,1])
print(f"Val  AUROC: {va_auc:.4f} | AUPRC: {va_ap:.4f}")
print(f"Test AUROC: {te_auc:.4f} | AUPRC: {te_ap:.4f}")

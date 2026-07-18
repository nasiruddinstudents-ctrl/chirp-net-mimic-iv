import torch, torch.nn as nn, numpy as np, pandas as pd, os, math
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import warnings; warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = int(os.environ.get('SEED', 42))
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed(SEED)
print(f"Device: {device} | Seed: {SEED} | mTAND")

EVENTS_DIR=Path("/root/events"); CACHE_DIR=Path("/root/cache"); CACHE_DIR.mkdir(exist_ok=True)
CACHE_PATH=CACHE_DIR/"mtand_tensors.npz"; MODEL_DIR=Path("/root/models/baselines"); MODEL_DIR.mkdir(exist_ok=True)
idx=pd.read_parquet("/root/graphs_index_clean.parquet"); cohort=pd.read_parquet("/root/cohort_clean.parquet")
label_map=dict(zip(idx['stay_id'],idx['mortality_label']))
pos=idx[idx['mortality_label']==1]; neg=idx[idx['mortality_label']==0]
def split3(df):
    tr,tmp=train_test_split(df,test_size=0.30,random_state=42)
    va,te=train_test_split(tmp,test_size=0.50,random_state=42)
    return tr,va,te
ptr,pva,pte=split3(pos); ntr,nva,nte=split3(neg)
train_ids=set(pd.concat([ptr,ntr])['stay_id'].tolist())
val_ids=set(pd.concat([pva,nva])['stay_id'].tolist())
test_ids=set(pd.concat([pte,nte])['stay_id'].tolist())
all_ids=list(idx['stay_id']); id2row={sid:i for i,sid in enumerate(all_ids)}
MAX_SEQ=256

if CACHE_PATH.exists():
    print("Loading cache...")
    d=np.load(CACHE_PATH,allow_pickle=True)
    T_all=d['T']; X_all=d['X']; M_all=d['M']; Y_all=d['Y']; IDS_all=d['IDS']
    N_VARS=X_all.shape[2]; print(f"Loaded: {T_all.shape}")
else:
    print("Building mTAND tensors...")
    vitals=pd.read_parquet(EVENTS_DIR/"vitals.parquet")
    labs=pd.read_parquet(EVENTS_DIR/"labs.parquet")
    labs=labs.merge(cohort[['hadm_id','stay_id']],on='hadm_id',how='inner')
    vitals['t_norm']=vitals['t_hours'].clip(0,48)/48.0; labs['t_norm']=labs['t_hours'].clip(0,48)/48.0
    vitals['vartype']='v_'+vitals['name'].astype(str); vitals['value']=pd.to_numeric(vitals['value'],errors='coerce')
    labs['vartype']='l_'+labs['name'].astype(str); labs['value']=pd.to_numeric(labs['value'],errors='coerce')
    events=pd.concat([vitals[['stay_id','t_norm','vartype','value']],labs[['stay_id','t_norm','vartype','value']]],ignore_index=True)
    events=events[events['stay_id'].isin(set(all_ids))].dropna(subset=['value'])
    events['value']=events['value'].clip(-10,10)
    var_names=sorted(events['vartype'].unique()); N_VARS=len(var_names); var2idx={v:i for i,v in enumerate(var_names)}
    events['vi']=events['vartype'].map(var2idx).astype(int); events=events.sort_values(['stay_id','t_norm'])
    N=len(all_ids)
    T_all=np.zeros((N,MAX_SEQ,1),dtype=np.float32); X_all=np.zeros((N,MAX_SEQ,N_VARS),dtype=np.float32); M_all=np.zeros((N,MAX_SEQ,N_VARS),dtype=np.float32)
    grp=events.groupby('stay_id')
    for sid,i in id2row.items():
        if sid not in grp.groups: continue
        g=grp.get_group(sid); time_steps=sorted(g['t_norm'].unique())[:MAX_SEQ]
        for si,t in enumerate(time_steps):
            T_all[i,si,0]=t
            for _,row in g[g['t_norm']==t].iterrows():
                vi=int(row['vi'])
                if vi<N_VARS: X_all[i,si,vi]=float(row['value']); M_all[i,si,vi]=1.0
    Y_all=np.array([float(label_map.get(sid,0)) for sid in all_ids],dtype=np.float32); IDS_all=np.array(all_ids)
    tr_mask=np.array([sid in train_ids for sid in all_ids])
    x_mean=np.nan_to_num(np.nanmean(np.where(M_all[tr_mask]>0,X_all[tr_mask],np.nan),axis=(0,1)),nan=0.0)
    x_std=np.nanstd(np.where(M_all[tr_mask]>0,X_all[tr_mask],np.nan),axis=(0,1)).clip(min=1e-6)
    X_all=(X_all-x_mean[np.newaxis,np.newaxis,:])/x_std[np.newaxis,np.newaxis,:]
    X_all=np.where(M_all>0,X_all,0.0)
    np.savez(CACHE_PATH,T=T_all,X=X_all,M=M_all,Y=Y_all,IDS=IDS_all,var_names=var_names)
    print(f"Cached: {N:,} x {MAX_SEQ} x {N_VARS}")

N_VARS=X_all.shape[2]
def get_split(id_set):
    mask=np.isin(IDS_all,list(id_set)); return T_all[mask],X_all[mask],M_all[mask],Y_all[mask]
T_tr,X_tr,M_tr,Y_tr=get_split(train_ids); T_va,X_va,M_va,Y_va=get_split(val_ids); T_te,X_te,M_te,Y_te=get_split(test_ids)
print(f"Train:{len(Y_tr):,} Val:{len(Y_va):,} Test:{len(Y_te):,} Vars:{N_VARS}")

class EHRDataset(Dataset):
    def __init__(self,T,X,M,Y):
        self.T=torch.tensor(T,dtype=torch.float32); self.X=torch.tensor(X,dtype=torch.float32)
        self.M=torch.tensor(M,dtype=torch.float32); self.Y=torch.tensor(Y,dtype=torch.float32)
    def __len__(self): return len(self.Y)
    def __getitem__(self,i): return self.T[i],self.X[i],self.M[i],self.Y[i]

w_pos=1.0/Y_tr.mean(); w_neg=1.0/(1-Y_tr.mean())
weights=np.where(Y_tr==1,w_pos,w_neg)
sampler=torch.utils.data.WeightedRandomSampler(torch.tensor(weights,dtype=torch.float),len(Y_tr),replacement=True)
train_loader=DataLoader(EHRDataset(T_tr,X_tr,M_tr,Y_tr),batch_size=32,sampler=sampler)
val_loader=DataLoader(EHRDataset(T_va,X_va,M_va,Y_va),batch_size=128,shuffle=False)
test_loader=DataLoader(EHRDataset(T_te,X_te,M_te,Y_te),batch_size=128,shuffle=False)

class TimeAttention(nn.Module):
    def __init__(self,d_model,n_ref=64):
        super().__init__()
        self.ref_times=nn.Parameter(torch.linspace(0,1,n_ref).unsqueeze(1))
        self.time_emb=nn.Linear(1,d_model); self.W_q=nn.Linear(d_model,d_model)
        self.W_k=nn.Linear(d_model,d_model); self.W_v=nn.Linear(d_model,d_model); self.out=nn.Linear(d_model,d_model)
    def forward(self,t,x,m):
        B,S,_=t.shape; d=self.W_q.out_features
        t_ref=self.ref_times.unsqueeze(0).expand(B,-1,-1)
        q=self.W_q(torch.sin(self.time_emb(t_ref)))
        k=self.W_k(torch.sin(self.time_emb(t)))
        v=self.W_v(torch.sin(self.time_emb(t))+x.mean(-1,keepdim=True).expand(-1,-1,d))
        attn=torch.bmm(q,k.transpose(1,2))/math.sqrt(d)
        obs_mask=(m.sum(-1)==0).unsqueeze(1).expand(-1,self.ref_times.shape[0],-1)
        attn=torch.softmax(attn.masked_fill(obs_mask,-1e9),dim=-1)
        return self.out(torch.bmm(attn,v))

class mTAND(nn.Module):
    def __init__(self,n_vars,d_model=64,n_ref=64,n_layers=2,dropout=0.1):
        super().__init__()
        self.proj=nn.Linear(n_vars,d_model); self.ta=TimeAttention(d_model,n_ref)
        enc=nn.TransformerEncoderLayer(d_model=d_model,nhead=4,dim_feedforward=d_model*2,dropout=dropout,batch_first=True)
        self.enc=nn.TransformerEncoder(enc,num_layers=n_layers)
        self.head=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,d_model//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(d_model//2,1))
    def forward(self,t,x,m):
        ref=self.ta(t,self.proj(x*m),m); out=self.enc(ref); return self.head(out.mean(1)).squeeze(-1)

model=mTAND(N_VARS).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=30)
criterion=nn.BCEWithLogitsLoss()

def evaluate(loader):
    model.eval(); probs,labels=[],[]
    with torch.no_grad():
        for T,X,M,Y in loader:
            T,X,M=T.to(device),X.to(device),M.to(device)
            out=model(T,X,M); p=torch.sigmoid(out).cpu().numpy().flatten()
            for pi,yi in zip(p,Y.numpy().flatten()):
                if np.isfinite(pi): probs.append(float(pi)); labels.append(int(yi))
    if len(set(labels))<2: return float('nan'),float('nan')
    return roc_auc_score(labels,probs),average_precision_score(labels,probs)

print(f"\n{'='*55}\nmTAND | Seed {SEED}\n{'='*55}")
best_auc,patience=0,0
for epoch in range(1,51):
    model.train(); total_loss,count=0,0
    for T,X,M,Y in train_loader:
        T,X,M,Y=T.to(device),X.to(device),M.to(device),Y.to(device)
        optimizer.zero_grad(); out=model(T,X,M); loss=criterion(out.flatten(),Y.flatten())
        if not torch.isfinite(loss): continue
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step(); total_loss+=loss.item(); count+=1
    scheduler.step(); val_auc,val_ap=evaluate(val_loader)
    print(f"Epoch {epoch:3d} | Loss: {total_loss/max(count,1):.4f} | Val AUROC: {val_auc:.4f} | AUPRC: {val_ap:.4f}")
    if np.isfinite(val_auc) and val_auc>best_auc:
        best_auc=val_auc; patience=0; torch.save(model.state_dict(),MODEL_DIR/f'mtand_seed{SEED}.pt')
        print(f"           *** Best: {best_auc:.4f} ***")
    else:
        patience+=1
        if patience>=10: print("Early stopping"); break

model.load_state_dict(torch.load(MODEL_DIR/f'mtand_seed{SEED}.pt'))
test_auc,test_ap=evaluate(test_loader)
print(f"\nTest AUROC: {test_auc:.4f} | AUPRC: {test_ap:.4f}")
pd.DataFrame([{'model':'mTAND','seed':SEED,'test_auroc':test_auc,'test_auprc':test_ap}]).to_csv(f'/root/results/baseline_mtand_seed{SEED}.csv',index=False)
print(f"Saved: /root/results/baseline_mtand_seed{SEED}.csv")

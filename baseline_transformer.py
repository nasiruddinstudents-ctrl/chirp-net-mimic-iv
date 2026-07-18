"""
Transformer Baseline for CHIRP-Net comparison
4-layer Transformer encoder on tokenized vital/lab events
Author: Mohammad Nasir Uddin
"""
import torch, torch.nn as nn, numpy as np, pandas as pd, os
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = int(os.environ.get('SEED', 42))
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed(SEED)
print(f"Device: {device} | Seed: {SEED} | Transformer Baseline")

EVENTS_DIR = Path("/root/events")
CACHE_DIR  = Path("/root/cache"); CACHE_DIR.mkdir(exist_ok=True)
CACHE_PATH = CACHE_DIR/"transformer_tensors.npz"
MODEL_DIR  = Path("/root/models/baselines"); MODEL_DIR.mkdir(exist_ok=True)

idx    = pd.read_parquet("/root/graphs_index_clean.parquet")
cohort = pd.read_parquet("/root/cohort_clean.parquet")
label_map = dict(zip(idx['stay_id'], idx['mortality_label']))
los_map   = dict(zip(cohort['stay_id'], cohort['los_hours']))

pos=idx[idx['mortality_label']==1]; neg=idx[idx['mortality_label']==0]
def split3(df):
    tr,tmp=train_test_split(df,test_size=0.30,random_state=42)
    va,te=train_test_split(tmp,test_size=0.50,random_state=42)
    return tr,va,te
ptr,pva,pte=split3(pos); ntr,nva,nte=split3(neg)
train_ids=set(pd.concat([ptr,ntr])['stay_id'].tolist())
val_ids  =set(pd.concat([pva,nva])['stay_id'].tolist())
test_ids =set(pd.concat([pte,nte])['stay_id'].tolist())
all_ids  =list(idx['stay_id'])
id2row   ={sid:i for i,sid in enumerate(all_ids)}

N_BINS=48; BIN_SIZE=1.0

if CACHE_PATH.exists():
    print("Loading cached tensors...")
    d=np.load(CACHE_PATH,allow_pickle=True)
    X_all=d['X']; M_all=d['M']; T_all=d['T']
    Y_all=d['Y']; IDS_all=d['IDS']
    N_VARS=X_all.shape[2]
    print(f"Loaded: {X_all.shape}")
else:
    print("Building Transformer tensors...")
    vitals=pd.read_parquet(EVENTS_DIR/"vitals.parquet")
    labs  =pd.read_parquet(EVENTS_DIR/"labs.parquet")
    cohort_map=pd.read_parquet("/root/cohort_clean.parquet")[['hadm_id','stay_id']]
    labs=labs.merge(cohort_map,on='hadm_id',how='inner')

    for df in [vitals,labs]:
        df['t_hours']=df['t_hours'].clip(0,47.99)
        df['bin']=(df['t_hours']).astype(int)

    vitals['vartype']='v_'+vitals['name'].astype(str)
    vitals['value']=pd.to_numeric(vitals['value'],errors='coerce')
    labs['vartype']='l_'+labs['name'].astype(str)
    labs['value']=pd.to_numeric(labs['value'],errors='coerce')

    dfs=[vitals[['stay_id','bin','t_hours','vartype','value']],
         labs[['stay_id','bin','t_hours','vartype','value']]]
    if (EVENTS_DIR/"meds.parquet").exists():
        meds=pd.read_parquet(EVENTS_DIR/"meds.parquet")
        meds['t_hours']=meds['t_hours'].clip(0,47.99)
        meds['bin']=meds['t_hours'].astype(int)
        meds['vartype']='m_'+meds['name'].astype(str)
        meds['value']=meds['amount'].fillna(0)
        dfs.append(meds[['stay_id','bin','t_hours','vartype','value']])

    events=pd.concat(dfs,ignore_index=True)
    events=events[events['stay_id'].isin(set(all_ids))].dropna(subset=['value'])
    events['value']=events['value'].clip(-10,10)

    var_names=sorted(events['vartype'].unique())
    N_VARS=len(var_names); var2idx={v:i for i,v in enumerate(var_names)}
    events['vi']=events['vartype'].map(var2idx).astype(int)
    events['si']=events['stay_id'].map(id2row)
    events=events.dropna(subset=['si']); events['si']=events['si'].astype(int)
    print(f"Variables: {N_VARS} | Events: {len(events):,}")

    agg=events.groupby(['si','bin','vi'])['value'].mean().reset_index()
    N=len(all_ids)
    X_all=np.zeros((N,N_BINS,N_VARS),dtype=np.float32)
    M_all=np.zeros((N,N_BINS,N_VARS),dtype=np.float32)
    T_all=np.zeros((N,N_BINS,1),dtype=np.float32)  # time encoding

    si=agg['si'].values.astype(int); bi=agg['bin'].values.astype(int)
    vi=agg['vi'].values.astype(int); val=agg['value'].values.astype(np.float32)
    msk=(si>=0)&(si<N)&(bi>=0)&(bi<N_BINS)&(vi>=0)&(vi<N_VARS)
    X_all[si[msk],bi[msk],vi[msk]]=val[msk]
    M_all[si[msk],bi[msk],vi[msk]]=1.0

    # Time encoding: normalized hour
    for t in range(N_BINS):
        T_all[:,t,0]=t/48.0

    Y_all=np.array([float(label_map.get(sid,0)) for sid in all_ids],dtype=np.float32)
    IDS_all=np.array(all_ids)

    # Normalize X on train only
    tr_mask=np.array([sid in train_ids for sid in all_ids])
    x_mean=np.nanmean(np.where(M_all[tr_mask]>0,X_all[tr_mask],np.nan),axis=(0,1))
    x_std =np.nanstd( np.where(M_all[tr_mask]>0,X_all[tr_mask],np.nan),axis=(0,1)).clip(min=1e-6)
    x_mean=np.nan_to_num(x_mean,nan=0.0)
    X_all=(X_all-x_mean[np.newaxis,np.newaxis,:])/x_std[np.newaxis,np.newaxis,:]
    X_all=np.where(M_all>0,X_all,0.0)  # zero out missing

    np.savez(CACHE_PATH,X=X_all,M=M_all,T=T_all,Y=Y_all,IDS=IDS_all,var_names=var_names)
    print(f"Cached: {N:,} x {N_BINS} x {N_VARS}")

def get_split(id_set):
    mask=np.isin(IDS_all,list(id_set))
    return X_all[mask],M_all[mask],T_all[mask],Y_all[mask]

X_tr,M_tr,T_tr,Y_tr=get_split(train_ids)
X_va,M_va,T_va,Y_va=get_split(val_ids)
X_te,M_te,T_te,Y_te=get_split(test_ids)
N_VARS=X_all.shape[2]
print(f"Train:{len(Y_tr):,} Val:{len(Y_va):,} Test:{len(Y_te):,} Vars:{N_VARS}")

class EHRDataset(Dataset):
    def __init__(self,X,M,T,Y):
        self.X=torch.tensor(X,dtype=torch.float32)
        self.M=torch.tensor(M,dtype=torch.float32)
        self.T=torch.tensor(T,dtype=torch.float32)
        self.Y=torch.tensor(Y,dtype=torch.float32)
    def __len__(self): return len(self.Y)
    def __getitem__(self,i): return self.X[i],self.M[i],self.T[i],self.Y[i]

w_pos=1.0/Y_tr.mean(); w_neg=1.0/(1-Y_tr.mean())
weights=np.where(Y_tr==1,w_pos,w_neg)
sampler=torch.utils.data.WeightedRandomSampler(torch.tensor(weights,dtype=torch.float),len(Y_tr),replacement=True)
train_loader=DataLoader(EHRDataset(X_tr,M_tr,T_tr,Y_tr),batch_size=64,sampler=sampler)
val_loader  =DataLoader(EHRDataset(X_va,M_va,T_va,Y_va),batch_size=256,shuffle=False)
test_loader =DataLoader(EHRDataset(X_te,M_te,T_te,Y_te),batch_size=256,shuffle=False)

class TransformerEHR(nn.Module):
    def __init__(self,n_vars,d_model=128,n_heads=4,n_layers=4,dropout=0.1,max_len=48):
        super().__init__()
        # Input projection: vars + mask + time → d_model
        self.input_proj=nn.Linear(n_vars*2+1, d_model)
        # Positional encoding
        pe=torch.zeros(max_len,d_model)
        pos=torch.arange(max_len).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,d_model,2).float()*(-np.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe',pe.unsqueeze(0))
        # Transformer
        enc_layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=n_heads,
                                              dim_feedforward=d_model*4,
                                              dropout=dropout,batch_first=True)
        self.transformer=nn.TransformerEncoder(enc_layer,num_layers=n_layers)
        self.norm=nn.LayerNorm(d_model)
        self.head=nn.Sequential(
            nn.Linear(d_model,d_model//2),nn.GELU(),
            nn.Dropout(dropout),nn.Linear(d_model//2,1))

    def forward(self,x,m,t):
        # x: (B,T,V) m: (B,T,V) t: (B,T,1)
        # Concatenate value, mask, time
        inp=torch.cat([x,m,t],dim=-1)  # (B,T,2V+1)
        h=self.input_proj(inp)          # (B,T,d_model)
        h=h+self.pe[:,:h.shape[1],:]
        # Create padding mask: positions where all vars are missing
        pad_mask=(m.sum(dim=-1)==0)     # (B,T) True=ignore
        h=self.transformer(h,src_key_padding_mask=pad_mask)
        h=self.norm(h)
        # CLS token = mean of non-padding positions
        valid=(~pad_mask).float().unsqueeze(-1)
        cls=( h*valid).sum(dim=1)/(valid.sum(dim=1)+1e-8)
        return self.head(cls).squeeze(-1)

model=TransformerEHR(n_vars=N_VARS,d_model=128,n_heads=4,n_layers=4).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=30)
criterion=nn.BCEWithLogitsLoss()

def evaluate(loader):
    model.eval(); probs,labels=[],[]
    with torch.no_grad():
        for X,M,T,Y in loader:
            X,M,T=X.to(device),M.to(device),T.to(device)
            out=model(X,M,T); p=torch.sigmoid(out).cpu().numpy().flatten()
            for pi,yi in zip(p,Y.numpy().flatten()):
                if np.isfinite(pi): probs.append(float(pi)); labels.append(int(yi))
    if len(set(labels))<2: return float('nan'),float('nan')
    return roc_auc_score(labels,probs),average_precision_score(labels,probs)

print(f"\n{'='*55}\nTransformer Baseline | Seed {SEED}\n{'='*55}")
best_auc,patience=0,0
for epoch in range(1,51):
    model.train(); total_loss,count=0,0
    for X,M,T,Y in train_loader:
        X,M,T,Y=X.to(device),M.to(device),T.to(device),Y.to(device)
        optimizer.zero_grad()
        out=model(X,M,T); loss=criterion(out.flatten(),Y.flatten())
        if not torch.isfinite(loss): continue
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step(); total_loss+=loss.item(); count+=1
    scheduler.step()
    val_auc,val_ap=evaluate(val_loader)
    print(f"Epoch {epoch:3d} | Loss: {total_loss/max(count,1):.4f} | Val AUROC: {val_auc:.4f} | AUPRC: {val_ap:.4f}")
    if np.isfinite(val_auc) and val_auc>best_auc:
        best_auc=val_auc; patience=0
        torch.save(model.state_dict(),MODEL_DIR/f'transformer_seed{SEED}.pt')
        print(f"           *** Best: {best_auc:.4f} ***")
    else:
        patience+=1
        if patience>=10: print(f"Early stopping"); break

model.load_state_dict(torch.load(MODEL_DIR/f'transformer_seed{SEED}.pt'))
test_auc,test_ap=evaluate(test_loader)
print(f"\nTest AUROC: {test_auc:.4f} | AUPRC: {test_ap:.4f}")
pd.DataFrame([{'model':'Transformer','seed':SEED,'test_auroc':test_auc,'test_auprc':test_ap,'n_vars':N_VARS}]).to_csv(f'/root/results/baseline_transformer_seed{SEED}.csv',index=False)
print(f"Saved: /root/results/baseline_transformer_seed{SEED}.csv")

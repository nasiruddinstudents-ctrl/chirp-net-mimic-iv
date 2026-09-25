import torch, torch.nn as nn, numpy as np, pandas as pd, os
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = int(os.environ.get('SEED', 42))
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

print(f"Device: {device} | Seed: {SEED} | GRU-D v2")

EVENTS_DIR = Path("/Users/sauban/chirp-project/data/chirp_processed")
CACHE_DIR  = Path("/Users/sauban/chirp-project/data/chirp_processed/cache"); CACHE_DIR.mkdir(exist_ok=True)
CACHE_PATH = CACHE_DIR/"grud_tensors.npz"
MODEL_DIR  = Path("/Users/sauban/chirp-project/data/chirp_processed/models/baselines"); MODEL_DIR.mkdir(parents=True, exist_ok=True)

idx    = pd.read_parquet("/Users/sauban/chirp-project/data/chirp_processed/graphs_index_clean.parquet")
c_clean = pd.read_parquet("/Users/sauban/chirp-project/data/chirp_processed/cohort_clean.parquet")
label_map = dict(zip(c_clean['stay_id'], c_clean['mortality_label']))

pos = idx[idx['mortality_label']==1]; neg = idx[idx['mortality_label']==0]
def split3(df):
    tr,tmp=train_test_split(df,test_size=0.30,random_state=42)
    va,te =train_test_split(tmp,test_size=0.50,random_state=42)
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
    X_all=d['X']; M_all=d['M']; D_all=d['D']
    Y_all=d['Y']; IDS_all=d['IDS']
    N_VARS=X_all.shape[2]
    print(f"Loaded: {len(X_all):,} x {N_BINS} x {N_VARS}")
else:
    print("Building tensors (vectorized)...")
    vitals=pd.read_parquet(EVENTS_DIR/"vitals.parquet")
    labs  =pd.read_parquet(EVENTS_DIR/"labs.parquet")
    cohort=pd.read_parquet("/Users/sauban/chirp-project/data/chirp_processed/cohort_clean.parquet")[['hadm_id','stay_id']]
    labs  =labs.merge(cohort, on='hadm_id', how='inner')
    t_col='t_hours'
    v_var='name'
    l_var='name'
    v_val='value'
    l_val='value'
    vitals['t_hours']=vitals[t_col].clip(0,47.99); vitals['bin']=vitals['t_hours'].astype(int)
    labs['t_hours']  =labs[t_col].clip(0,47.99);   labs['bin']  =labs['t_hours'].astype(int)
    vitals['vartype']='v_'+vitals[v_var].astype(str)
    vitals['value']=pd.to_numeric(vitals[v_val],errors='coerce')
    labs['vartype']='l_'+labs[l_var].astype(str)
    labs['value']=pd.to_numeric(labs[l_val],errors='coerce')
    dfs=[vitals[['stay_id','bin','vartype','value']],
         labs[['stay_id','bin','vartype','value']]]
    if (EVENTS_DIR/"meds.parquet").exists():
        meds=pd.read_parquet(EVENTS_DIR/"meds.parquet")
        t_med='t_hours'
        m_var='name'
        meds['t_hours']=meds[t_med].clip(0,47.99); meds['bin']=meds['t_hours'].astype(int)
        meds['vartype']='m_'+meds[m_var].astype(str); meds['value']=meds['amount'].fillna(0).clip(0,1000)
        dfs.append(meds[['stay_id','bin','vartype','value']]); print("vitals+labs+meds")
    else: print("vitals+labs only")
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
    si=agg['si'].values.astype(int); bi=agg['bin'].values.astype(int)
    vi=agg['vi'].values.astype(int); val=agg['value'].values.astype(np.float32)
    msk=(si>=0)&(si<N)&(bi>=0)&(bi<N_BINS)&(vi>=0)&(vi<N_VARS)
    X_all[si[msk],bi[msk],vi[msk]]=val[msk]
    M_all[si[msk],bi[msk],vi[msk]]=1.0
    print("Computing deltas...")
    D_all=np.zeros((N,N_BINS,N_VARS),dtype=np.float32)
    for t in range(1,N_BINS):
        D_all[:,t,:]=np.where(M_all[:,t-1,:]>0,BIN_SIZE,D_all[:,t-1,:]+BIN_SIZE)
    Y_all=np.array([float(label_map.get(sid,0)) for sid in all_ids],dtype=np.float32)
    IDS_all=np.array(all_ids)
    tr_mask=np.array([sid in train_ids for sid in all_ids])
    d_max=D_all[tr_mask].max(axis=(0,1)).clip(min=1.0)
    D_all=D_all/d_max[np.newaxis,np.newaxis,:]
    np.savez(CACHE_PATH,X=X_all,M=M_all,D=D_all,Y=Y_all,IDS=IDS_all,var_names=var_names)
    print(f"Cached to {CACHE_PATH}")

tr_mask=np.isin(IDS_all,list(train_ids))
x_mean=np.nanmean(np.where(M_all[tr_mask]>0,X_all[tr_mask],np.nan),axis=(0,1))
x_mean=np.nan_to_num(x_mean,nan=0.0)
N_VARS=X_all.shape[2]

def get_split(id_set):
    mask=np.isin(IDS_all,list(id_set))
    return X_all[mask],M_all[mask],D_all[mask],Y_all[mask]

X_tr,M_tr,D_tr,Y_tr=get_split(train_ids)
X_va,M_va,D_va,Y_va=get_split(val_ids)
X_te,M_te,D_te,Y_te=get_split(test_ids)
print(f"Train:{len(Y_tr):,} Val:{len(Y_va):,} Test:{len(Y_te):,} Vars:{N_VARS}")

class GRUDDataset(Dataset):
    def __init__(self,X,M,D,Y):
        self.X=torch.tensor(X,dtype=torch.float32)
        self.M=torch.tensor(M,dtype=torch.float32)
        self.D=torch.tensor(D,dtype=torch.float32)
        self.Y=torch.tensor(Y,dtype=torch.float32)
    def __len__(self): return len(self.Y)
    def __getitem__(self,i): return self.X[i],self.M[i],self.D[i],self.Y[i]

w_pos=1.0/Y_tr.mean(); w_neg=1.0/(1-Y_tr.mean())
weights=np.where(Y_tr==1,w_pos,w_neg)
sampler=torch.utils.data.WeightedRandomSampler(
    torch.tensor(weights,dtype=torch.float),len(Y_tr),replacement=True)
train_loader=DataLoader(GRUDDataset(X_tr,M_tr,D_tr,Y_tr),batch_size=64,sampler=sampler)
val_loader  =DataLoader(GRUDDataset(X_va,M_va,D_va,Y_va),batch_size=256,shuffle=False)
test_loader =DataLoader(GRUDDataset(X_te,M_te,D_te,Y_te),batch_size=256,shuffle=False)

class GRUDCell(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.W_gamma_x=nn.Linear(input_size,input_size)
        self.W_gamma_h=nn.Linear(input_size,hidden_size)
        self.gru_cell=nn.GRUCell(input_size*3,hidden_size)
    def forward(self,x,m,d,h,x_mean):
        gamma_x=torch.exp(-torch.relu(self.W_gamma_x(d)))
        gamma_h=torch.exp(-torch.relu(self.W_gamma_h(d)))
        h=gamma_h*h
        x_imp=m*x+(1-m)*(gamma_x*x+(1-gamma_x)*x_mean.unsqueeze(0))
        return self.gru_cell(torch.cat([x_imp,m,gamma_x],dim=-1),h)

class GRUD(nn.Module):
    def __init__(self,input_size,hidden_size=256,x_mean=None):
        super().__init__()
        self.hidden_size=hidden_size
        self.cell=GRUDCell(input_size,hidden_size)
        self.classifier=nn.Sequential(
            nn.Linear(hidden_size,hidden_size//2),nn.ReLU(),
            nn.Dropout(0.3),nn.Linear(hidden_size//2,1))
        xm=torch.tensor(x_mean,dtype=torch.float32) if x_mean is not None \
           else torch.zeros(input_size)
        self.register_buffer('x_mean',xm)
    def forward(self,x,m,d):
        B,T,F=x.shape; h=torch.zeros(B,self.hidden_size,device=x.device)
        for t in range(T): h=self.cell(x[:,t,:],m[:,t,:],d[:,t,:],h,self.x_mean)
        return self.classifier(h).squeeze(-1)

model=GRUD(N_VARS,hidden_size=256,x_mean=x_mean).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=30)
criterion=nn.BCEWithLogitsLoss()

def evaluate(loader):
    model.eval(); probs,labels=[],[]
    with torch.no_grad():
        for X,M,D,Y in loader:
            X,M,D=X.to(device),M.to(device),D.to(device)
            out=model(X,M,D); p=torch.sigmoid(out).cpu().numpy().flatten()
            for pi,yi in zip(p,Y.numpy().flatten()):
                if np.isfinite(pi): probs.append(float(pi)); labels.append(int(yi))
    if len(set(labels))<2: return float('nan'),float('nan')
    return roc_auc_score(labels,probs),average_precision_score(labels,probs)

print(f"\n{'='*55}\nGRU-D v2 | Seed {SEED}\n{'='*55}")
best_auc,patience=0,0
for epoch in range(1,51):
    model.train(); total_loss,count=0,0
    for X,M,D,Y in train_loader:
        X,M,D,Y=X.to(device),M.to(device),D.to(device),Y.to(device)
        optimizer.zero_grad()
        out=model(X,M,D); loss=criterion(out.flatten(),Y.flatten())
        if not torch.isfinite(loss): continue
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step(); total_loss+=loss.item(); count+=1
    scheduler.step()
    val_auc,val_ap=evaluate(val_loader)
    print(f"Epoch {epoch:3d} | Loss: {total_loss/max(count,1):.4f} | Val AUROC: {val_auc:.4f} | AUPRC: {val_ap:.4f}")
    if np.isfinite(val_auc) and val_auc>best_auc:
        best_auc=val_auc; patience=0
        torch.save(model.state_dict(),MODEL_DIR/f'grud_seed{SEED}.pt')
        print(f"           *** Best: {best_auc:.4f} ***")
    else:
        patience+=1
        if patience>=10: print(f"Early stopping"); break

model.load_state_dict(torch.load(MODEL_DIR/f'grud_seed{SEED}.pt'))
test_auc,test_ap=evaluate(test_loader)
print(f"\nTest AUROC: {test_auc:.4f} | AUPRC: {test_ap:.4f}")
print(f"\nGRU-D v2 Seed {SEED} complete!")
pd.DataFrame([{'model':'GRU-D','seed':SEED,'test_auroc':test_auc,
               'test_auprc':test_ap,'n_vars':N_VARS,'n_bins':N_BINS}]).to_csv(
    f'/Users/sauban/chirp-project/data/chirp_processed/results/baseline_grud_seed{SEED}.csv',index=False)
print(f"Saved: /Users/sauban/chirp-project/data/chirp_processed/results/baseline_grud_seed{SEED}.csv")

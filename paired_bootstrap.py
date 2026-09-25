"""
Paired patient-level bootstrap for CT-HEG / CHIRP-Net (reproduces Tables 5-6 CIs).

Input : predictions_unified/<condition>_seed<N>_test_predictions.csv  (stay_id,label,prob)
Method: for each contrast (A - B), 2,000 resamples of test stays with replacement;
        in each resample the metric difference is computed per seed on the same stays
        and averaged over seeds; 95% CI = 2.5th/97.5th percentiles. RNG seed 12345 per contrast.
Usage : python3 paired_bootstrap.py --pred_dir predictions_unified [--out bootstrap_results.csv]
"""
import numpy as np, pandas as pd, glob, re, os, sys, argparse
from scipy.stats import rankdata
PRED_DIR='predictions_unified'
def load(c):
    out={}
    for f in sorted(glob.glob(os.path.join(PRED_DIR, f'{c}_seed*_test_predictions.csv'))):
        s=int(re.search(r'_seed(\d+)',os.path.basename(f)).group(1))
        out[s]=pd.read_csv(f).set_index('stay_id').sort_index()
    return out
def auroc(y,p):
    r=rankdata(p); n1=y.sum(); n0=len(y)-n1
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def ap(y,p):
    o=np.argsort(-p,kind='mergesort'); y=y[o]; tp=np.cumsum(y); prec=tp/np.arange(1,len(y)+1)
    return (prec*y).sum()/y.sum()
C=[("heterogeneous_small","homogeneous_small","Typing, small budget (aux)"),
   ("full","homogeneous_large","Typing, large budget (aux)"),
   ("heterogeneous_small_noaux","homogeneous_small_noaux","Typing, small budget (no aux)"),
   ("no_aux","homogeneous_large_noaux","Typing, large budget (no aux)"),
   ("heterogeneous_small","full","Capacity, heterogeneous 330K vs 1.85M"),
   ("homogeneous_small","homogeneous_large","Capacity, relation-collapsed 345K vs 1.84M"),
   ("heterogeneous_small","heterogeneous_small_noaux","Aux loss, 330K"),
   ("full","no_aux","Aux loss, 1.85M"),
   ("heterogeneous_small","vitlab_only_s80","Medication, 330K"),
   ("full","vitlab_only","Medication, 1.85M"),
   ("heterogeneous_small","edge_value_zeroed_s80","Edge value zeroed, 330K"),
   ("heterogeneous_small","edge_time_zeroed_s80","Edge time zeroed, 330K"),
   ("heterogeneous_small","edge_time_permuted_s80","Edge time permuted, 330K"),
   ("heterogeneous_small","no_reverse_s80","No reverse, 330K"),
   ("full","edge_value_zeroed","Edge value zeroed, 1.85M"),
   ("full","edge_time_zeroed","Edge time zeroed, 1.85M"),
   ("full","edge_time_permuted","Edge time permuted, 1.85M"),
   ("full","no_reverse","No reverse, 1.85M")]
B=2000
def run(c):
    A,Bn,lab=c; pa,pb=load(A),load(Bn); seeds=sorted(set(pa)&set(pb))
    ids=pa[seeds[0]].index
    for s in seeds: assert pa[s].index.equals(ids) and pb[s].index.equals(ids)
    y=pa[seeds[0]].label.values.astype(int)
    PA=np.stack([pa[s].prob.values for s in seeds]); PB=np.stack([pb[s].prob.values for s in seeds])
    rng=np.random.default_rng(12345); n=len(y); res=[]
    for name,fn in [('AUROC',auroc),('AUPRC',ap)]:
        sd=np.array([fn(y,PA[i])-fn(y,PB[i]) for i in range(len(seeds))])
        bs=np.empty(B)
        for b in range(B):
            ix=rng.integers(0,n,n); yb=y[ix]
            bs[b]=np.mean([fn(yb,PA[i,ix])-fn(yb,PB[i,ix]) for i in range(len(seeds))])
        lo,hi=np.percentile(bs,[2.5,97.5]); p=2*min((bs<=0).mean(),(bs>=0).mean())
        res.append(dict(contrast=lab,A=A,B=Bn,metric=name,n_seeds=len(seeds),n=n,diff=sd.mean(),lo=lo,hi=hi,p_boot=p,
                        seed_diffs=' '.join(f'{d:+.4f}' for d in sd)))
    return res
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--pred_dir',default='predictions_unified')
    ap.add_argument('--out',default='bootstrap_results.csv'); a=ap.parse_args(); PRED_DIR=a.pred_dir
    rows=[]
    for c in C:
        rows+=run(c); print('done:',c[2],flush=True)
    df=pd.DataFrame(rows); df.to_csv(a.out,index=False)
    pd.set_option('display.width',250)
    print(df[['contrast','metric','diff','lo','hi','seed_diffs']].round(4).to_string(index=False))

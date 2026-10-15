import argparse, glob, os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
AGE_BANDS = [(18, 44, '18–44'), (45, 64, '45–64'), (65, 79, '65–79'), (80, 200, '80+')]
SMALL = 11; N_BOOT = 2000; RNG = np.random.default_rng(42)

def read_any(p):
    p = str(p)
    if p.endswith('.parquet'): return pd.read_parquet(p)
    return pd.read_csv(p)

def mimic_table(root, sub, name):
    for ext in ('.csv.gz', '.csv', '.parquet'):
        p = os.path.join(root, sub, name + ext)
        if os.path.exists(p): return read_any(p)
    sys.exit(f'Cannot find {sub}/{name} under {root}')

def pick(cols, pref, user=None, what=''):
    if user:
        if user not in cols: sys.exit(f'Column {user!r} not in {list(cols)}')
        return user
    low = {c.lower(): c for c in cols}
    for p in pref:
        if p in low: return low[p]
    sys.exit(f'Could not detect the {what} column in {list(cols)}; pass it explicitly.')

def load_predictions(pattern, ic, pc, lc):
    files = sorted(glob.glob(os.path.expanduser(pattern)))
    if not files: sys.exit(f'No prediction files match {pattern!r}')
    seeds = []
    for f in files:
        d = read_any(f)
        i = pick(d.columns, ['stay_id','stay','icustay_id','id'], ic, 'stay id')
        p = pick(d.columns, ['y_prob','prob','proba','y_pred_prob','pred','y_pred','score','p'], pc, 'probability')
        y = pick(d.columns, ['y_true','label','y','target','mortality','true'], lc, 'label')
        print(f'  {os.path.basename(f)}: columns {list(d.columns)} -> id={i}, prob={p}, label={y}')
        d = d[[i, p, y]].rename(columns={i:'stay_id', p:'prob', y:'label'})
        if d['prob'].min() < 0 or d['prob'].max() > 1: d['prob'] = 1/(1+np.exp(-d['prob']))
        seeds.append(d.set_index('stay_id').sort_index())
    ids = seeds[0].index
    for s in seeds[1:]:
        if not s.index.equals(ids): sys.exit('Seed files do not cover the same stays.')
    return ids, seeds[0]['label'].astype(int).values, np.column_stack([s['prob'].values for s in seeds])

def race_group(r):
    r = str(r).upper()
    if r.startswith('WHITE') or r == 'PORTUGUESE': return 'White'
    if r.startswith('BLACK'): return 'Black'
    if r.startswith('HISPANIC') or r.startswith('SOUTH AMERICAN'): return 'Hispanic/Latino'
    if r.startswith('ASIAN'): return 'Asian'
    if r in ('UNKNOWN','UNABLE TO OBTAIN','PATIENT DECLINED TO ANSWER','NAN',''): return 'Unknown/declined'
    return 'Other'

def insurance_group(s):
    s = str(s).lower()
    for k in ('medicare','medicaid','private'):
        if k in s: return k.capitalize()
    return 'Unknown' if s in ('nan','') else 'Other'

def age_band(a):
    for lo, hi, lab in AGE_BANDS:
        if lo <= a <= hi: return lab
    return 'Unknown'

def fmt_n(n, t):
    if n == 0: return '0'
    if n < SMALL: return f'<{SMALL}'
    return f'{n:,} ({100*n/t:.1f})'

def fmt_med(x):
    q = np.nanpercentile(x, [50, 25, 75]); return f'{q[0]:.1f} [{q[1]:.1f}–{q[2]:.1f}]'

def characteristics(df, groups):
    G = list(groups.values()); rows = []
    rows.append(('Stays, n', [f'{len(g):,}' for g in G]))
    rows.append(('Age, years, median [IQR]', [fmt_med(g.age) for g in G]))
    rows.append(('Female sex, n (%)', [fmt_n(int((g.gender=='F').sum()), len(g)) for g in G]))
    for var, lab in [('age_band','Age band'),('race_group','Race/ethnicity'),('insurance_group','Insurance'),
                     ('first_careunit','First ICU care unit'),('admission_type','Admission type')]:
        levels = [b[2] for b in AGE_BANDS] if var == 'age_band' else df[var].value_counts().index.tolist()
        rows.append((f'{lab}, n (%)', ['' for _ in G]))
        for lv in levels: rows.append((f'    {lv}', [fmt_n(int((g[var]==lv).sum()), len(g)) for g in G]))
    rows.append(('ICU length of stay, days, median [IQR]', [fmt_med(g.los) for g in G]))
    rows.append(('In-hospital death after 48 h, n (%)', [fmt_n(int(g.label.sum()), len(g)) for g in G]))
    out = pd.DataFrame([r[1] for r in rows], columns=list(groups.keys()))
    out.insert(0, 'Characteristic', [r[0] for r in rows]); return out

def fast_auroc(y, p):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y==1].sum() - n1*(n1+1)/2) / (n1*n0))

def fast_ap(y, p):
    o = np.argsort(-p, kind='mergesort'); ys, ps = y[o], p[o]
    last = np.r_[np.where(np.diff(ps) != 0)[0], len(ps)-1]
    tp = np.cumsum(ys)[last]; prec = tp/(last+1); rec = tp/ys.sum()
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))

def smean(y, P, fn): return float(np.mean([fn(y, P[:, k]) for k in range(P.shape[1])]))

def boot_ci(y, P, fn):
    n = len(y); v = []
    for _ in range(N_BOOT):
        i = RNG.integers(0, n, n)
        if y[i].min() != y[i].max(): v.append(smean(y[i], P[i], fn))
    return np.percentile(v, [2.5, 97.5])

def subgroup_table(test, y, P):
    rows = []
    for var, lab in [('gender','Sex'),('age_band','Age band'),('race_group','Race/ethnicity'),('insurance_group','Insurance')]:
        for lv in sorted(test[var].dropna().unique()):
            m = (test[var]==lv).values; n, d = int(m.sum()), int(y[m].sum())
            row = {'Variable': lab, 'Group': {'F':'Female','M':'Male'}.get(lv, lv),
                   'Stays': n if n >= SMALL else f'<{SMALL}', 'Deaths': d if d >= SMALL else f'<{SMALL}'}
            if d < 10 or n-d < 10: row.update({'AUROC [95% CI]': 'not estimated (<10 events)', 'AUPRC [95% CI]': ''})
            else:
                for nm, fn in [('AUROC', fast_auroc), ('AUPRC', fast_ap)]:
                    lo, hi = boot_ci(y[m], P[m], fn); row[f'{nm} [95% CI]'] = f'{smean(y[m], P[m], fn):.4f} [{lo:.4f}, {hi:.4f}]'
            rows.append(row)
    return pd.DataFrame(rows)

def _logit(p): p = np.clip(p, 1e-6, 1-1e-6); return np.log(p/(1-p))

def _fit(X, y, off=None):
    X = np.column_stack([np.ones(len(y))] + ([X] if X is not None else []))
    off = np.zeros(len(y)) if off is None else off; b = np.zeros(X.shape[1])
    for _ in range(100):
        mu = 1/(1+np.exp(-(X@b+off))); H = X.T@(X*(mu*(1-mu))[:, None]) + 1e-9*np.eye(X.shape[1])
        s = np.linalg.solve(H, X.T@(y-mu)); b += s
        if np.abs(s).max() < 1e-10: break
    return b

def _bins(p, k=10):
    e = np.quantile(p, np.linspace(0, 1, k+1)); e[-1] += 1e-12
    return np.clip(np.searchsorted(e, p, side='right')-1, 0, k-1)

def calib(y, p):
    lp = _logit(p); w = _bins(p)
    ece = sum(abs(p[w==b].mean()-y[w==b].mean())*(w==b).mean() for b in range(10) if (w==b).any())
    return dict(brier=float(np.mean((p-y)**2)), citl=_fit(None, y, lp)[0], slope=_fit(lp, y)[1], ece=float(ece), mean_pred=float(p.mean()))

def reliability_plot(y, p, path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    w = _bins(p); mp = [p[w==b].mean() for b in range(10)]; mo = [y[w==b].mean() for b in range(10)]
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=300)
    ax.plot([0,1],[0,1], ls='--', lw=1, color='#888888', label='Perfect calibration')
    ax.plot(mp, mo, marker='o', lw=1.5, color='#1f4e79', label='CHIRP-Net (seed-averaged)')
    ax.set_xlabel('Mean predicted probability', fontsize=10); ax.set_ylabel('Observed mortality', fontsize=10)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.tick_params(labelsize=9); ax.legend(fontsize=8, frameon=False, loc='upper left')
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    for k in ('--mimic','--cohort','--pred-glob'): ap.add_argument(k, required=True)
    for k in ('--id-col','--prob-col','--label-col'): ap.add_argument(k)
    ap.add_argument('--out', default='gap_outputs'); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    print('Loading MIMIC-IV tables...')
    icu = mimic_table(a.mimic, 'icu', 'icustays'); pat = mimic_table(a.mimic, 'hosp', 'patients'); adm = mimic_table(a.mimic, 'hosp', 'admissions')
    coh = read_any(a.cohort); sid = pick(coh.columns, ['stay_id','icustay_id'], None, 'cohort stay id')
    coh = coh.rename(columns={sid:'stay_id'})[['stay_id']].drop_duplicates()
    df = (coh.merge(icu[['subject_id','hadm_id','stay_id','los','first_careunit']], on='stay_id', how='left')
             .merge(pat[['subject_id','gender','anchor_age','anchor_year']], on='subject_id', how='left')
             .merge(adm[['hadm_id','admittime','admission_type','insurance','race','hospital_expire_flag']], on='hadm_id', how='left'))
    df['age'] = df.anchor_age + (pd.to_datetime(df.admittime).dt.year - df.anchor_year)
    df['age_band'] = df.age.apply(age_band); df['race_group'] = df.race.apply(race_group)
    df['insurance_group'] = df.insurance.apply(insurance_group); df['label'] = df.hospital_expire_flag.fillna(0).astype(int)
    print(f'Cohort: {len(df):,} stays, {df.subject_id.nunique():,} patients, {int(df.label.sum()):,} deaths (manuscript: 31,142 stays, 4,160 deaths)')
    print('Loading test predictions...')
    ids, y, P = load_predictions(a.pred_glob, a.id_col, a.prob_col, a.label_col)
    test = df.set_index('stay_id').loc[ids].reset_index()
    if (test.label.values != y).any(): print(f'WARNING: labels differ for {(test.label.values != y).sum()} test stays; saved labels used.')
    test['label'] = y
    print(f'Test set: {len(y):,} stays, {int(y.sum()):,} deaths, {P.shape[1]} seeds (manuscript: 4,672 / 624 / 5)')
    print(f'Check: 5-seed mean AUROC {smean(y, P, roc_auc_score):.4f} (manuscript 0.8547)')
    t1 = characteristics(df, {'All stays': df, 'Survivors': df[df.label==0], 'Non-survivors': df[df.label==1], 'Test set': test})
    print('Bootstrapping subgroup metrics (a minute or two)...'); t2 = subgroup_table(test, y, P)
    ps = pd.DataFrame([calib(y, P[:, k]) for k in range(P.shape[1])]); av = calib(y, P.mean(1)); C = ['brier','citl','slope','ece','mean_pred']
    t3 = pd.DataFrame({'Metric': ['Brier score','Calibration-in-the-large (intercept)','Calibration slope','Expected calibration error (10 quantile bins)','Mean predicted probability','Observed mortality'],
        'Per-seed mean ± SD (population SD)': [f'{ps[c].mean():.4f} ± {ps[c].std(ddof=0):.4f}' for c in C] + [f'{y.mean():.4f}'],
        'Seed-averaged prediction': [f'{av[c]:.4f}' for c in C] + [f'{y.mean():.4f}']})
    reliability_plot(y, P.mean(1), os.path.join(a.out, '3_reliability_diagram.png'))
    with open(os.path.join(a.out, 'ALL_RESULTS.txt'), 'w') as f:
        for nm, t in [('1. CHARACTERISTICS', t1), ('2. SUBGROUPS', t2), ('3. CALIBRATION', t3)]: f.write(f'{nm}\n{t.to_string(index=False)}\n\n')
        f.write(f'Cohort {len(df)} / {int(df.label.sum())} deaths; test {len(y)} / {int(y.sum())}; {P.shape[1]} seeds; AUROC {smean(y, P, roc_auc_score):.4f}\n')
    print(open(os.path.join(a.out, 'ALL_RESULTS.txt')).read())
    print(f'Done. Send me the text above and {a.out}/3_reliability_diagram.png')

if __name__ == '__main__': main()

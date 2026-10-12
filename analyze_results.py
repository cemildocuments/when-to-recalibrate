"""Paired seed-cluster summaries and numeric tables for the frozen experiments."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
import json,argparse
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
RESULTS=Path(os.environ.get('RECALIBRATE_RESULTS',ROOT/'results'))

def load(name):
    p=RESULTS/f'{name}.jsonl';df=pd.read_json(p,lines=True)
    config=json.loads((ROOT/f'configs/{name}.json').read_text());assert len(df)==len(config['jobs']),(name,len(df),len(config['jobs']))
    assert df.job_id.nunique()==len(df)
    return df

def ci(x,seed=915,B=10000):
    x=np.array(x,dtype=float);rng=np.random.default_rng(seed);boot=np.mean(x[rng.integers(len(x),size=(B,len(x)))],axis=1)
    return [float(np.mean(x)),*map(float,np.quantile(boot,[.025,.975]))]

def main():
    dest=RESULTS/'summary';dest.mkdir(exist_ok=True)
    df=load('main');dr=df[df.kind!='none'];summary={}
    for label,data in [('drift',dr),('all',df),('no_drift',df[df.kind=='none'])]:
        means=data.groupby(['seed','policy']).regret.mean().unstack()
        summary[label]={'regret':{p:ci(means[p]) for p in means},'paired_improvement':{p:ci(means[p]-means.adaptive) for p in means if p!='adaptive'},'relative_reduction':{p:float(1-means.adaptive.mean()/means[p].mean()) for p in means if p!='adaptive'}}
    rows=[]
    for keys,g in dr.groupby(['dataset','policy']):
        a=ci(g.groupby('seed').regret.mean());rows.append(dict(dataset=keys[0],policy=keys[1],mean=a[0],lower=a[1],upper=a[2],refs=g.ref_count.mean(),near_optimal=g.near_optimal.mean()))
    pd.DataFrame(rows).to_csv(dest/'per_dataset.csv',index=False)
    rows=[]
    for (d,k,c),g in df.groupby(['dataset','kind','ref_cost']):
        w=g.pivot(index='seed',columns='policy',values='regret');a=ci(w.periodic-w.adaptive)
        rows.append(dict(dataset=d,kind=k,ref_cost=c,adaptive=w.adaptive.mean(),periodic=w.periodic.mean(),improvement=a[0],lower=a[1],upper=a[2]))
    pd.DataFrame(rows).to_csv(dest/'condition_contrasts.csv',index=False)
    rows=[]
    for (c,p),g in dr.groupby(['ref_cost','policy']):
        a=ci(g.groupby('seed').regret.mean());rows.append(dict(ref_cost=c,policy=p,mean=a[0],lower=a[1],upper=a[2],refs=g.ref_count.mean(),samples=g.sample_count.mean(),near_optimal=g.near_optimal.mean()))
    pd.DataFrame(rows).to_csv(dest/'cost_frontier.csv',index=False)
    summary['counts']={'main':len(df),'independent_test_seed_clusters':df.seed.nunique(),'drift_conditions':64,'no_drift_conditions':16}
    summary['runtime']={p:{'median_seconds':float(g.seconds.median()),'p95_seconds':float(g.seconds.quantile(.95))} for p,g in df.groupby('policy')}
    for name in ['stress','budgets','matched','paired']:
        if not (RESULTS/f'{name}.jsonl').exists():continue
        a=load(name);summary['counts'][name]=len(a)
        if name=='stress':
            rows=[]
            for key,g in a.groupby(['stress_factor','stress_value','policy']):
                vals=ci(g.groupby('seed').regret.mean());cert=g.risk_bound<=.05
                rows.append(dict(factor=key[0],value=key[1],policy=key[2],mean=vals[0],lower=vals[1],upper=vals[2],refs=g.ref_count.mean(),certified=cert.mean(),false_certified=g.false_certification.mean(),error_among_certified=float(g.loc[cert,'false_certification'].mean()) if cert.any() else None))
            pd.DataFrame(rows).to_csv(dest/'stress.csv',index=False)
        elif name=='budgets':
            a=pd.concat([a,df[(df.kind=='brownian')&(df.ref_cost.isin([.25,.5]))&(df.seed<1100)&(df.policy.isin(a.policy.unique()))]])
            rows=[]
            for key,g in a.groupby(['dataset','budget','policy']):
                vals=ci(g.groupby('seed').regret.mean());rows.append(dict(dataset=key[0],budget=key[1],policy=key[2],mean=vals[0],lower=vals[1],upper=vals[2]))
            pd.DataFrame(rows).to_csv(dest/'budgets.csv',index=False)
        elif name=='matched':
            rows=[]
            for key,g in a.groupby(['ref_cost','policy']):
                cert=g.risk_bound<=.05;rows.append(dict(ref_cost=key[0],policy=key[1],certified=int(cert.sum()),n=len(g),wrong=int(g.false_certification.sum()),risk_bound=float(g.loc[cert,'risk_bound'].mean()),regret=float(g.regret.mean())))
            pd.DataFrame(rows).to_csv(dest/'matched.csv',index=False)
        elif name=='paired':
            merged=a.merge(df[df.policy=='adaptive'],on=['dataset','kind','ref_cost','seed'],suffixes=('_paired','_adaptive'))
            vals=merged.groupby('seed')[['regret_paired','regret_adaptive','ref_count_paired','ref_count_adaptive']].mean();summary['paired_ablation']={'regret_difference':ci(vals.regret_paired-vals.regret_adaptive),'paired_regret':float(vals.regret_paired.mean()),'adaptive_regret':float(vals.regret_adaptive.mean()),'paired_refs':float(vals.ref_count_paired.mean()),'adaptive_refs':float(vals.ref_count_adaptive.mean())}
    (dest/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

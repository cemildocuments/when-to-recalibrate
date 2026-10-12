import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import json
from pathlib import Path
import pandas as pd
from analyze_results import load,ci,ROOT,RESULTS
x=load('budget_confirmation');rows=[];contr=[]
for (d,b,p),g in x.groupby(['dataset','budget','policy']):
 a=ci(g.regret);rows.append(dict(dataset=d,budget=b,policy=p,mean=a[0],lower=a[1],upper=a[2],refs=float(g.ref_count.mean())))
for (d,b),g in x.groupby(['dataset','budget']):
 w=g.pivot(index='seed',columns='policy',values='regret')
 for p in ['joint','periodic']:
  a=ci(w[p]-w.adaptive);contr.append(dict(dataset=d,budget=b,comparator=p,improvement=a[0],lower=a[1],upper=a[2],relative_reduction=float(1-w.adaptive.mean()/w[p].mean())))
S=RESULTS/'summary';S.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(S/'budget_confirmation.csv',index=False);pd.DataFrame(contr).to_csv(S/'budget_confirmation_contrasts.csv',index=False)
main=load('main');r=[]
for (d,p),g in main[main.kind!='none'].groupby(['dataset','policy']):
 cert=g.risk_bound<=.05;r.append(dict(dataset=d,policy=p,n=len(g),flag_count=int(cert.sum()),wrong_count=int(g.false_certification.sum()),near_optimal=float(g.near_optimal.mean())))
pd.DataFrame(r).to_csv(S/'reliability.csv',index=False)
print(pd.DataFrame(contr).round(6).to_string(index=False))

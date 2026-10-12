"""Independent follow-up: budget interaction with horizon-tuned schedules."""
from pathlib import Path
import json,itertools,subprocess,sys,hashlib,datetime
import pandas as pd
P=Path(__file__).resolve().parent
D=['hplc','photo_pce10','branin','gp'];K=['none','brownian','linear','jumps','periodic'];B=[15,30,60,90]
jobs=[dict(dataset=d,kind=k,ref_cost=.5,seed=s,policy='periodic',budget=b,scale=.06,q_assumed=.0036,period=p) for d,k,b,s,p in itertools.product(D,K,B,range(40,60),[2,4,8,16,32,1000])]
(P/'configs/budget_validation.json').write_text(json.dumps({'output':'results/budget_validation.jsonl','jobs':jobs},separators=(',',':')))
with (P/'results/budget_validation_progress.log').open('w') as out:subprocess.run([sys.executable,'run_experiments.py','--config','configs/budget_validation.json','--workers','10'],stdout=out,stderr=subprocess.STDOUT,check=True)
v=pd.read_json(P/'results/budget_validation.jsonl',lines=True);sel=[]
for (d,b),g in v.groupby(['dataset','budget']):
 a=g.groupby('period').regret.mean();sel.append(dict(dataset=d,budget=int(b),period=int(a.idxmin()),validation_regret=float(a.min())))
lookup={(r['dataset'],r['budget']):r['period'] for r in sel}
(P/'configs/budget_selected.json').write_text(json.dumps(sel,indent=2))
jobs=[]
for d,b,s,p in itertools.product(D,B,range(5000,6000),['adaptive','joint','periodic']):
 j=dict(dataset=d,kind='brownian',ref_cost=.5,seed=s,policy=p,budget=b,scale=.06,q_assumed=.0036)
 if p=='periodic':j['period']=lookup[d,b]
 jobs.append(j)
f=P/'configs/budget_confirmation.json';f.write_text(json.dumps({'output':'results/budget_confirmation.jsonl','jobs':jobs},separators=(',',':')))
(P/'configs/budget_confirmation_protocol.json').write_text(json.dumps({'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'Follow-up of the first budget sensitivity analysis: independently estimate reference value versus remaining measurement opportunity, including all four response families and all four budgets. Schedules are reselected for each horizon on validation data.','outcomes':'All 16 dataset/budget conditions are reported, with paired seed-cluster intervals.','test_seeds':[5000,5999],'sha256':hashlib.sha256(f.read_bytes()).hexdigest()},indent=2))
with (P/'results/budget_confirmation_progress.log').open('w') as out:subprocess.run([sys.executable,'run_experiments.py','--config','configs/budget_confirmation.json','--workers','10'],stdout=out,stderr=subprocess.STDOUT,check=True)
print('Budget confirmation complete',len(jobs),flush=True)

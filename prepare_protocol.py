"""Select baseline settings on validation seeds, then freeze held-out designs."""
import json,hashlib,itertools,datetime
from pathlib import Path
import pandas as pd
P=Path(__file__).resolve().parent
D=['hplc','photo_pce10','branin','gp'];K=['none','brownian','linear','jumps','periodic'];C=[.25,.5,1.,2.]
params={'periodic':'period','variance':'threshold','residual':'threshold','timegp':'rho','duel_ids':'beta'}
v=pd.read_json(P/'results/validation_final.jsonl',lines=True)
selected=[]
for (d,c,p),g in v.groupby(['dataset','ref_cost','policy']):
    par=params[p];a=g.groupby(par).regret.mean().sort_index();best=a.idxmin()
    selected.append({'dataset':d,'ref_cost':c,'policy':p,par:float(best) if par!='period' else int(best),'validation_regret':float(a.loc[best])})
(P/'configs/selected_parameters.json').write_text(json.dumps(selected,indent=2))
lookup={(r['dataset'],r['ref_cost'],r['policy']):{params[r['policy']]:r[params[r['policy']]]} for r in selected}
methods=['adaptive','joint','periodic','variance','residual','timegp','duel_ids','duel_kg','stationary','oracle']
def job(d,k,c,s,p,budget=60,**extra):
    return dict(dataset=d,kind=k,ref_cost=c,seed=s,policy=p,budget=budget,scale=.06,q_assumed=.0036,**lookup.get((d,c,p),{}),**extra)
configs={}
configs['main']=[job(d,k,c,s,p) for d,k,c,s,p in itertools.product(D,K,C,range(1000,1200),methods)]
configs['budgets']=[job(d,'brownian',c,s,p,budget=b) for d,c,s,p,b in itertools.product(D,[.25,.5],range(1000,1100),['adaptive','joint','periodic','duel_kg','stationary','oracle'],[15,30,90])]
configs['paired']=[job(d,k,c,s,'paired') for d,k,c,s in itertools.product(D,K,C,range(1000,1050))]
stress=[]
for d,s,p in itertools.product(D,range(2000,2100),['adaptive','joint','periodic','stationary']):
    for field,values in [('q_assumed',[.0004,.0036,.0144]),('reference_bias',[0.,.05,.15]),('gain_amplitude',[0.,.2,.5]),('length',[.2,.35,.6])]:
        for value in values:
            j=job(d,'brownian',.5,s,p);j[field]=value;j.update(stress_factor=field,stress_value=value);stress.append(j)
configs['stress']=stress
configs['matched']=[job('gp','brownian',c,s,p,drift_prior=0.) for c,s,p in itertools.product(C,range(3000,4000),['adaptive','periodic','joint'])]
for name,jobs in configs.items():
    (P/f'configs/{name}.json').write_text(json.dumps({'output':f'results/{name}.jsonl','jobs':jobs},separators=(',',':')))
protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'primary_outcome':'Terminal simple regret; equal-weight mean over 64 nonzero-drift dataset/regime/reference-cost conditions. Zero-drift controls reported separately.','inference':'Paired percentile bootstrap of the 200 independent seed clusters, 10000 replicates; preserve all conditions and policies within each seed.','comparison':'Adaptive versus separately validation-tuned periodic, threshold and temporal baselines, joint model without references, dueling baselines, stationary model; oracle is an information-advantaged diagnostic.','validation_seeds':[40,59],'test_seeds':[1000,1199],'stress_seeds':[2000,2099],'matched_model_seeds':[3000,3999],'adaptation':'No parameter changes after observing held-out results.','config_sha256':{n:hashlib.sha256((P/f'configs/{n}.json').read_bytes()).hexdigest() for n in configs},'code_sha256':{str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((P/'src').rglob('*.py'))}}
(P/'configs/protocol.json').write_text(json.dumps(protocol,indent=2))
print('Counts:',{k:len(v) for k,v in configs.items()});print(pd.DataFrame(selected).to_string(index=False))

#!/usr/bin/env python3
"""Re-run frozen campaigns and compare their numerical outputs with the reference run."""
import argparse,json,math,subprocess,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SUITES=['validation_final','budget_validation','main','budgets','paired','stress','matched','budget_confirmation']

def equivalent(a,b,path=''):
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        if not math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-10):raise AssertionError(f'{path}: {a} != {b}')
    elif a!=b:raise AssertionError(f'{path}: {a!r} != {b!r}')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--suite',choices=SUITES+['all'],default='main');ap.add_argument('--workers',type=int,default=10);ap.add_argument('--quick',action='store_true');args=ap.parse_args()
    dest=ROOT/'reproduced';dest.mkdir(exist_ok=True)
    chosen=['main'] if args.quick else SUITES if args.suite=='all' else [args.suite]
    report=[]
    for name in chosen:
        cfg=json.loads((ROOT/f'configs/{name}.json').read_text());jobs=[]
        for i,j in enumerate(cfg['jobs']):
            if args.quick and not(j['seed'] in [1000,1001] and j['kind']=='brownian' and j['ref_cost']==.5):continue
            jobs.append({**j,'source_job_id':i})
        suffix=name+('_quick' if args.quick else '');out=dest/(suffix+'.jsonl');newcfg=dest/(suffix+'_config.json')
        newcfg.write_text(json.dumps({'jobs':jobs,'output':str(out)},separators=(',',':')))
        subprocess.run([sys.executable,str(ROOT/'run_experiments.py'),'--config',str(newcfg),'--workers',str(args.workers)],cwd=ROOT,check=True)
        expected={r['job_id']:r for r in map(json.loads,(ROOT/f'results/{name}.jsonl').open())}
        count=0
        for line in out.open():
            r=json.loads(line);e=expected[r['source_job_id']]
            for k,v in e.items():
                if k not in ['seconds','job_id']:equivalent(v,r[k],f'{name}:{r["source_job_id"]}:{k}')
            count+=1
        assert count==len(jobs);report.append({'suite':name,'matched_campaigns':count,'tolerance_relative':1e-9,'tolerance_absolute':1e-10});print(f'{name}: {count} campaigns match the reference outputs.',flush=True)
    (dest/('verification_quick.json' if args.quick else 'verification.json')).write_text(json.dumps(report,indent=2))
    if args.suite=='all' and not args.quick:
        env={**os.environ,'RECALIBRATE_RESULTS':str(dest)}
        subprocess.run([sys.executable,str(ROOT/'analyze_results.py')],cwd=ROOT,env=env,check=True)
        subprocess.run([sys.executable,str(ROOT/'analyze_confirmation.py')],cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()

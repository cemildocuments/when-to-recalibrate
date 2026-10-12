#!/usr/bin/env python3
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMBA_NUM_THREADS']:os.environ[key]='1'
import sys,json,time,argparse,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from concurrent.futures import ProcessPoolExecutor,as_completed
from recalibrate.core import campaign
from recalibrate.duel import duel_campaign
from recalibrate.data import load_pool

def worker(job):
    X,f=load_pool(job['dataset'],job['seed'],job.get('n_candidates',96))
    keys=['seed','policy','kind','budget','ref_cost','scale','q_assumed','period','threshold','noise','ref_noise','gain_amplitude','reference_bias','length','drift_prior','rho','beta']
    t=time.perf_counter();fn=duel_campaign if job['policy'] in ['duel_ids','duel_kg'] else campaign;result=fn(X,f,**{k:job[k] for k in keys if k in job});return {**job,**result,'seconds':time.perf_counter()-t}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--workers',type=int,default=10);args=ap.parse_args()
    config=json.loads(Path(args.config).read_text());jobs=config['jobs'];out=Path(config['output']);out.parent.mkdir(parents=True,exist_ok=True)
    done=set()
    if out.exists():
        for line in out.read_text().splitlines():
            record=json.loads(line);idx=record['job_id']
            if idx>=len(jobs) or any(record.get(k)!=v for k,v in jobs[idx].items()):raise ValueError('Existing output does not match this configuration; use a new output path.')
            done.add(idx)
    pending=[{**j,'job_id':i} for i,j in enumerate(jobs) if i not in done]
    print(f'{len(pending)} pending of {len(jobs)}',flush=True);start=time.time()
    with out.open('a') as stream,ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(worker,j) for j in pending]
        for k,fut in enumerate(as_completed(futures),1):
            result=fut.result();stream.write(json.dumps(result)+'\n');stream.flush()
            if k%500==0 or k==len(pending):print(f'{k}/{len(pending)}; {time.time()-start:.1f} s',flush=True)

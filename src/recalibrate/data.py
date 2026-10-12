import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]

def load_pool(name,seed,n=96):
    rng=np.random.default_rng(np.random.SeedSequence([seed,99]))
    if name=='gp':
        from .core import kernel
        X=rng.random((n,3));f=.5+np.linalg.cholesky(kernel(X)+1e-12*np.eye(n))@rng.normal(size=n)
        return X,f
    if name=='branin':
        X=rng.random((n,2));x=15*X[:,0]-5;y=15*X[:,1]
        f=-(y-5.1*x*x/(4*np.pi**2)+5*x/np.pi-6)**2-10*(1-1/(8*np.pi))*np.cos(x)-10
        f=(f-f.min())/(f.max()-f.min())
        return X,f
    d=ROOT/'data/upstream'/name
    config=json.loads((d/'config.json').read_text())
    arr=np.loadtxt(d/'data.csv',delimiter=',');dim=len(config['parameters'])
    frame=pd.DataFrame(arr);arr=frame.groupby(list(range(dim)),sort=True,as_index=False).mean().to_numpy()
    X=arr[:,:dim];f=arr[:,dim]
    for j,param in enumerate(config['parameters']):X[:,j]=(X[:,j]-param['low'])/(param['high']-param['low'])
    if config['default_goal']=='minimize': f=-f
    # Full archive extrema define public benchmark units, never an acquisition feature.
    f=(f-f.min())/(f.max()-f.min())
    ids=rng.choice(len(f),min(n,len(f)),replace=False)
    return X[ids],f[ids]

"""Random-order paired-difference baselines, using a common candidate kernel."""
import numpy as np
from .core import kernel,drift_path,kg,normal_cdf

def duel_campaign(X,f,seed,policy,kind='brownian',budget=60,ref_cost=.5,scale=.06,q_assumed=None,period=8,threshold=.15,noise=.025,ref_noise=.025,trace=False,gain_amplitude=0.,reference_bias=0.,length=.35,drift_prior=.02,rho=.98,beta=2.):
    n=len(f);q=scale**2 if q_assumed is None else q_assumed
    mu=np.full(n,.5);P=kernel(X,length)
    rng=np.random.default_rng(np.random.SeedSequence([seed,183]))
    initial=np.random.default_rng(np.random.SeedSequence([seed,451])).choice(n,6,replace=False)
    eps=np.random.default_rng(np.random.SeedSequence([seed,891])).normal(size=(n+1,1024));counts=np.zeros(n,dtype=int)
    times,b=drift_path(kind,seed,budget,scale);cost=0;h=[]
    while cost+2<=budget:
        if cost<6:i,j=int(initial[int(cost)]),int(initial[int(cost)+1])
        else:
            i=int(np.argmax(mu));v=np.maximum(P[i,i]+np.diag(P)-2*P[i],0.)
            if policy=='duel_kg':
                vals=np.array([kg(mu,(P[:,i]-P[:,a])/np.sqrt(max(v[a]+2*noise**2+q,1e-14))) for a in range(n)]);j=int(np.argmax(vals))
            else:
                delta=float(np.max(mu-mu[i]+beta*np.sqrt(v)))
                gaps=np.maximum(mu[i]-mu,0.);p=np.minimum(1.,delta/np.maximum(gaps,1e-12))
                info=np.log1p(v/(2*noise**2+q))
                ratio=(delta+p*gaps)**2/np.maximum(p*info,1e-14);ratio[i]=np.inf
                j=int(np.argmin(ratio))
                if rng.random()>p[j]:j=i
        order=[i,j] if rng.random()<.5 else [j,i]
        obs=[]
        for a in order:
            cost+=1;gain=1+gain_amplitude*np.sin(2*np.pi*cost/30.)
            obs.append(gain*f[a]+np.interp(cost,times,b)+noise*eps[a,counts[a]]);counts[a]+=1
        sign=1 if order[0]==i else -1
        diff=sign*(obs[0]-obs[1]);c=P[:,i]-P[:,j]
        den=P[i,i]+P[j,j]-2*P[i,j]+2*noise**2+q
        mu+=c*(diff-(mu[i]-mu[j]))/den;P-=np.outer(c,c)/den
        rec=int(np.argmax(mu));regret=float(f.max()-f[rec]);v=np.maximum(P[rec,rec]+np.diag(P)-2*P[rec],1e-14)
        risk=min(1.,sum(normal_cdf(-(mu[rec]-mu[a]+.05)/np.sqrt(v[a])) for a in range(n) if a!=rec))
        h.append([cost,regret,0,rec,risk])
    h=np.array(h);best=int(np.argmax(f));v=max(P[best,best]+P[rec,rec]-2*P[best,rec],1e-14)
    out={'regret':regret,'near_optimal':float(regret<=.05),'ref_count':0,'sample_count':int(cost),'cost':cost,'risk_bound':risk,'false_certification':int(risk<=.05 and regret>.05),'auc_regret':float(np.mean(h[:,1])),'contrast_covered':int(abs(regret-(mu[best]-mu[rec]))<=1.95996398454*np.sqrt(v)),'bias_rmse':None}
    if trace:out['trace']=h.tolist();out['mu']=mu.tolist();out['P']=P.tolist()
    return out

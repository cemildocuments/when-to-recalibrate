"""Sequential finite-pool optimization with a latent instrument state."""
import math
import numpy as np
from numba import njit

@njit(cache=True)
def normal_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))

@njit(cache=True)
def normal_pdf(x):
    return math.exp(-0.5*x*x) / math.sqrt(2.0*math.pi)

@njit(cache=True)
def kg(mu, slope):
    """Exact E[max_i(mu_i+slope_i Z)]-max(mu); upper envelope of lines."""
    n=len(mu)
    order=np.argsort(slope)
    indices=np.empty(n,np.int64)
    cuts=np.empty(n,np.float64)
    k=0
    for pos in range(n):
        j=order[pos]
        if k>0 and abs(slope[j]-slope[indices[k-1]])<1e-14:
            if mu[j]<=mu[indices[k-1]]:
                continue
            k-=1
        cross=-np.inf
        while k>0:
            old=indices[k-1]
            cross=(mu[old]-mu[j])/(slope[j]-slope[old])
            if cross>cuts[k-1]:
                break
            k-=1
        if k==0:
            cross=-np.inf
        indices[k]=j
        cuts[k]=cross
        k+=1
    value=0.0
    for h in range(k):
        lo=cuts[h]
        hi=cuts[h+1] if h+1<k else np.inf
        j=indices[h]
        value+=mu[j]*(normal_cdf(hi)-normal_cdf(lo))+slope[j]*(normal_pdf(lo)-normal_pdf(hi))
    return max(0.,value-np.max(mu))

@njit(cache=True)
def scores(mu, P, q, noise, ref_noise, ref_cost):
    n=len(mu)-1
    score=np.empty(n+1)
    for a in range(n+1):
        v=P[-1,-1]+q*(ref_cost if a==n else 1.)
        c=P[:n,-1].copy()
        if a<n:
            c+=P[:n,a]
            v+=P[a,a]+2*P[a,-1]+noise**2
        else:
            v+=ref_noise**2
        score[a]=kg(mu[:n],c/math.sqrt(max(v,1e-14)))/(ref_cost if a==n else 1.)
    return score

@njit(cache=True)
def paired_scores(mu,P,q,noise,ref_noise,ref_cost,nodes,weights):
    n=len(mu)-1
    vr=P[-1,-1]+q*ref_cost+ref_noise**2
    cr=P[:,-1].copy();cr[-1]+=q*ref_cost
    beta1=cr[:n]/math.sqrt(vr)
    Pc=P.copy();Pc[-1,-1]+=q*ref_cost;Pc-=np.outer(cr,cr)/vr
    out=np.empty(n)
    for a in range(n):
        vs=Pc[a,a]+Pc[-1,-1]+2*Pc[a,-1]+q+noise**2
        beta2=(Pc[:n,a]+Pc[:n,-1])/math.sqrt(max(vs,1e-14))
        value=0.
        for k in range(len(nodes)):
            m=mu[:n]+beta1*nodes[k]
            value+=weights[k]*(np.max(m)+kg(m,beta2))
        out[a]=max(0.,value-np.max(mu[:n]))/(1+ref_cost)
    return out

@njit(cache=True)
def update(mu,P,a,y,q,noise,cost):
    n=len(mu)-1
    P[-1,-1]+=q*cost
    c=P[:,-1].copy()
    pred=mu[-1]
    if a<n:
        c+=P[:,a]
        pred+=mu[a]
        v=P[a,a]+P[-1,-1]+2*P[a,-1]+noise*noise
    else:
        v=P[-1,-1]+noise*noise
    mu+=c*((y-pred)/v)
    P-=np.outer(c,c)/v
    return (y-pred)/math.sqrt(v)

def kernel(X,length=0.35,nugget=0.03):
    d=np.sqrt(np.sum(((X[:,None,:]-X[None,:,:])/length)**2,axis=2))
    z=np.sqrt(5.)*d
    return (0.35**2)*((1+z+z*z/3)*np.exp(-z)+nugget*np.eye(len(X)))

def drift_path(kind,seed,budget,scale,dt=0.25):
    rng=np.random.default_rng(np.random.SeedSequence([seed,727]))
    t=np.arange(int(budget/dt)+5)*dt
    if kind=='none': b=np.zeros(len(t))
    elif kind=='brownian': b=np.r_[0,np.cumsum(rng.normal(0,scale*np.sqrt(dt),len(t)-1))]
    elif kind=='linear': b=(rng.choice([-1.,1.])*scale*0.35)*t
    elif kind=='periodic': b=scale*4*np.sin(2*np.pi*t/30+rng.uniform(0,2*np.pi)); b-=b[0]
    elif kind=='jumps':
        jump=(rng.random(len(t)-1)<dt/18)*rng.normal(0,scale*5,len(t)-1)
        b=np.r_[0,np.cumsum(jump+rng.normal(0,scale*.25*np.sqrt(dt),len(t)-1))]
    else: raise ValueError(kind)
    return t,b

def campaign(X,f,seed,policy,kind='brownian',budget=60,ref_cost=.5,scale=.06,q_assumed=None,period=8,threshold=.15,noise=.025,ref_noise=.025,trace=False,gain_amplitude=0.,reference_bias=0.,length=.35,drift_prior=.02,rho=.98,beta=2.):
    n=len(f);q=scale**2 if q_assumed is None else q_assumed
    if policy in ['stationary','oracle','timegp']: q=0.
    mu=np.zeros(n+1);mu[:n]=.5
    P=np.zeros((n+1,n+1));P[:n,:n]=kernel(X,length);P[-1,-1]=drift_prior**2
    times,drifts=drift_path(kind,seed,budget,scale)
    init_rng=np.random.default_rng(np.random.SeedSequence([seed,451]))
    initial=init_rng.choice(n,min(6,n),replace=False)
    # Action-indexed noise permits paired comparisons without policy-specific streams.
    noise_rng=np.random.default_rng(np.random.SeedSequence([seed,891]))
    eps=noise_rng.normal(size=(n+1,1024))
    counts=np.zeros(n+1,dtype=int);cost=0.;steps=0;since=0;history=[];last_resid=0.; pending=-1
    nodes,weights=np.polynomial.hermite.hermgauss(9);nodes*=np.sqrt(2);weights/=np.sqrt(np.pi)
    while cost+min(1,ref_cost)<=budget+1e-9:
        if policy=='timegp':
            P[:n,-1]*=rho;P[-1,:n]*=rho
            mu[:n]=.5+rho*(mu[:n]-.5)
            P[:n,:n]=rho**2*P[:n,:n]+(1-rho**2)*kernel(X,length)
        if pending>=0:
            a=pending;pending=-1
        elif steps<len(initial):a=int(initial[steps])
        else:
            sc=scores(mu,P,q,noise,ref_noise,ref_cost)
            if policy in ['adaptive','paired']:
                a=int(np.argmax(sc))
                if policy=='paired' and cost+1+ref_cost<=budget+1e-9:
                    pair=paired_scores(mu,P,q,noise,ref_noise,ref_cost,nodes,weights)
                    if np.max(pair)>np.max(sc):
                        a=n;pending=int(np.argmax(pair))
            elif policy in ['stationary','oracle','joint','timegp']: a=int(np.argmax(sc[:n]))
            elif policy=='periodic': a=n if since>=period else int(np.argmax(sc[:n]))
            elif policy=='variance': a=n if np.sqrt(P[-1,-1]+q*ref_cost)>threshold else int(np.argmax(sc[:n]))
            elif policy=='residual': a=n if abs(last_resid)>threshold and since>0 else int(np.argmax(sc[:n]))
            elif policy=='information':
                # Target-f information per unit cost, still using KG for candidate selection.
                a0=int(np.argmax(sc[:n]));inds=[a0,n];vals=[]
                Pf=P[:n,:n]+1e-9*np.eye(n)
                for j in inds:
                    c=P[:n,-1].copy();v=P[-1,-1]+q*(ref_cost if j==n else 1)
                    if j<n:c+=P[:n,j];v+=P[j,j]+2*P[j,-1]+noise**2
                    else:v+=ref_noise**2
                    ratio=c@np.linalg.solve(Pf,c)/v
                    vals.append(-.5*np.log(max(1-ratio,1e-12))/(ref_cost if j==n else 1))
                a=inds[int(np.argmax(vals))]
            else:raise ValueError(policy)
        cst=ref_cost if a==n else 1.
        if cost+cst>budget+1e-9:
            if cost+1.<=budget+1e-9:
                a=int(np.argmax(scores(mu,P,q,noise,ref_noise,ref_cost)[:n]));cst=1.;pending=-1
            elif cost+ref_cost<=budget+1e-9:
                a=n;cst=ref_cost;pending=-1
            else:break
        cost+=cst;since=0 if a==n else since+1
        b=float(np.interp(cost,times,drifts))
        sigma=ref_noise if a==n else noise
        gain=1.+gain_amplitude*np.sin(2*np.pi*cost/30.)
        y=(reference_bias if a==n else gain*f[a])+b+sigma*eps[a,counts[a]]
        if policy=='oracle': y-=b
        last_resid=update(mu,P,a,y,q,sigma,cst)
        counts[a]+=1;steps+=1
        rec=int(np.argmax(mu[:n]));regret=float(np.max(f)-f[rec])
        delta=mu[rec]-mu[:n]+.05;var=np.maximum(P[rec,rec]+np.diag(P)[:n]-2*P[rec,:n],1e-14)
        # Union bound for posterior probability of any superior candidate.
        prob=np.array([normal_cdf(-delta[j]/np.sqrt(var[j])) if j!=rec else 0 for j in range(n)])
        risk=min(1.,float(np.sum(prob)))
        history.append([cost,regret,int(a==n),rec,risk,b,mu[-1],np.sqrt(max(P[-1,-1],0)),a, y])
    h=np.array(history)
    best=int(np.argmax(f));diffvar=max(P[best,best]+P[rec,rec]-2*P[best,rec],1e-14)
    contrast_error=(f[best]-f[rec])-(mu[best]-mu[rec])
    result={'contrast_covered':int(abs(contrast_error)<=1.95996398454*np.sqrt(diffvar)), 'bias_rmse':float(np.sqrt(np.mean((h[:,5]-h[:,6])**2))), 'regret':regret,'near_optimal':float(regret<=.05),'ref_count':int(counts[n]),'sample_count':int(counts[:n].sum()),'cost':cost,'risk_bound':risk,'false_certification':int(risk<=.05 and regret>.05),'auc_regret':float(np.sum(h[:,1]*np.diff(np.r_[0,h[:,0]]))/cost)}
    if trace: result['trace']=h.tolist();result['mu']=mu.tolist();result['P']=P.tolist()
    return result

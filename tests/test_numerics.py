import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from recalibrate.core import kg,kernel,update,campaign,paired_scores
from recalibrate.data import load_pool

class Numerics(unittest.TestCase):
 def test_kg_monte_carlo(self):
  rng=np.random.default_rng(63)
  for n in [2,8,31]:
   mu=rng.normal(size=n);s=rng.normal(size=n);z=rng.normal(size=400000)
   vals=np.max(mu[:,None]+s[:,None]*z,axis=0)
   self.assertLess(abs(kg(mu,s)-(vals.mean()-mu.max())),6*vals.std()/np.sqrt(len(z)))
   self.assertAlmostEqual(kg(mu,s),kg(mu,s+7),places=10)
   self.assertAlmostEqual(kg(mu,np.ones(n)),0.,places=10)
   self.assertLessEqual(kg(mu,s),(s.max()-s.min())/np.sqrt(2*np.pi)+1e-10)
 def test_batch_gp_agreement(self):
  rng=np.random.default_rng(39);X=rng.random((9,2));K=kernel(X);q=.004;R=.001;prior=.0004
  mu=np.r_[np.full(9,.5),0.];P=np.zeros((10,10));P[:9,:9]=K;P[-1,-1]=prior
  ids=[2,9,5,2,9,7];costs=[1,.5,1,1,.5,1];y=rng.normal(size=6);T=np.cumsum(costs)
  for a,c,z in zip(ids,costs,y):update(mu,P,a,z,q,np.sqrt(R),c)
  A=np.zeros((6,9))
  for i,a in enumerate(ids):
   if a<9:A[i,a]=1
  C=A@K@A.T+prior+q*np.minimum(T[:,None],T[None,:])+R*np.eye(6)
  cross=np.vstack([K@A.T,prior+q*T]);initial=np.r_[np.full(9,.5),0.]
  m=initial+cross@np.linalg.solve(C,y-A@np.full(9,.5));V=np.zeros((10,10));V[:9,:9]=K;V[-1,-1]=prior+q*T[-1];V-=cross@np.linalg.solve(C,cross.T)
  np.testing.assert_allclose(mu,m,atol=1e-10);np.testing.assert_allclose(P,V,atol=1e-10)
  self.assertGreater(np.linalg.eigvalsh(P).min(),-1e-10)
 def test_reference_value_decreases_with_cost(self):
  rng=np.random.default_rng(5);A=rng.normal(size=(12,12));P=A@A.T;mu=rng.normal(size=11);c=P[:-1,-1]
  costs=np.logspace(-2,2,30);v=[kg(mu,c/np.sqrt(P[-1,-1]+.1*t+.001))/t for t in costs]
  self.assertTrue(np.all(np.diff(v)<=1e-12))
 def test_reproducibility_and_budget(self):
  X,f=load_pool('gp',521,n=48)
  for policy in ['adaptive','paired','periodic','joint','timegp']:
   a=campaign(X,f,521,policy,budget=20,ref_cost=.5,trace=True)
   b=campaign(X,f,521,policy,budget=20,ref_cost=.5,trace=True)
   self.assertEqual(a,b);self.assertLessEqual(a['cost'],20)
   P=np.array(a['P']);self.assertGreater(np.linalg.eigvalsh(P).min(),-1e-8)
 def test_no_drift_oracle(self):
  X,f=load_pool('gp',320,n=48)
  a=campaign(X,f,320,'stationary',kind='none',budget=20)
  b=campaign(X,f,320,'oracle',kind='none',budget=20)
  self.assertEqual(a,b)
if __name__=='__main__':unittest.main()

import numpy as np
import scipy.linalg as la
from scipy.stats import norm

class Engine:
    def __init__(self,S,K,r,T,sig,corr,w):
        self.S=np.array(S,dtype=float)
        self.K=float(K)
        self.r=float(r)
        self.T=float(T)
        self.sig=np.array(sig,dtype=float)
        self.corr=np.array(corr,dtype=float)
        self.w=np.array(w,dtype=float)
        self.dim=len(self.S)
        self.cov=self.sig[:,None]*self.sig[None,:]*self.corr
        self.L=la.cholesky(self.cov,lower=True)

    def _mc(self,n):
        z=np.random.normal(size=(n,self.dim))
        y=z@self.L.T
        drift=(self.r-0.5*np.diag(self.cov))*self.T
        st=self.S*np.exp(drift+y*np.sqrt(self.T))
        return st

    def price_call(self,n=200000):
        st=self._mc(n)
        payoff=np.maximum(st@self.w-self.K,0.0)
        return np.exp(-self.r*self.T)*payoff.mean()

    def price_put(self,n=200000):
        st=self._mc(n)
        payoff=np.maximum(self.K-st@self.w,0.0)
        return np.exp(-self.r*self.T)*payoff.mean()

    def delta(self,eps=1e-4,n=150000):
        base=self.price_call(n)
        d=np.zeros(self.dim)
        for i in range(self.dim):
            s0=self.S[i]
            self.S[i]=s0*(1+eps)
            up=self.price_call(n)
            self.S[i]=s0*(1-eps)
            dn=self.price_call(n)
            self.S[i]=s0
            d[i]=(up-dn)/(2*s0*eps)
        return d

    def vega(self,eps=1e-4,n=150000):
        v=np.zeros(self.dim)
        for i in range(self.dim):
            s0=self.sig[i]
            self.sig[i]=s0*(1+eps)
            self.cov=self.sig[:,None]*self.sig[None,:]*self.corr
            self.L=la.cholesky(self.cov,lower=True)
            up=self.price_call(n)
            self.sig[i]=s0*(1-eps)
            self.cov=self.sig[:,None]*self.sig[None,:]*self.corr
            self.L=la.cholesky(self.cov,lower=True)
            dn=self.price_call(n)
            self.sig[i]=s0
            self.cov=self.sig[:,None]*self.sig[None,:]*self.corr
            self.L=la.cholesky(self.cov,lower=True)
            v[i]=(up-dn)/(2*s0*eps)
        return v

def basket_analytic(S,K,r,T,sig,corr,w):
    S=np.array(S,dtype=float)
    sig=np.array(sig,dtype=float)
    w=np.array(w,dtype=float)
    cov=sig[:,None]*sig[None,:]*corr
    mu=np.log(S)+(r-0.5*np.diag(cov))*T
    var=w@cov@w*T
    m=w@np.exp(mu+0.5*np.diag(cov)*T)
    s=np.sqrt(var)
    d1=(np.log(m/K)+0.5*s*s)/s
    d2=d1-s
    return np.exp(-r*T)*(m*norm.cdf(d1)-K*norm.cdf(d2))

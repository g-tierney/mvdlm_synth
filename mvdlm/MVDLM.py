import scipy, math
import numpy as np
import copy
from scipy.stats import multivariate_t 


# MVDLM without covariates
class MVDLM:
    
    def __init__(self, beta, delta, q, m0=None, c0=100, n0=20, D0=None):
        """
        :param beta: Stochastic volitility discount factor
        :param delta: State space discount factor
        :param q: Dimension of series
        :param m0: prior mean
        :param c0: prior 
        
        theta ~ NIW(m,c,n,D) such that: 
            1) theta|Sigma ~ MVN(m,c Sigma)
            2) Sigma ~ IW(n,D) such that E[Sigma] = D/(n-2)
            3) theta ~ T_n(m,c S) where S = D/n
        """
        if m0 is None: m0=np.zeros(q)
        if D0 is None: D0=n0*np.eye(q)
        self.beta = beta
        self.delta = delta
        self.q = q
        self.m = m0
        self.m_store = []
        self.D = D0
        self.D_store = []
        self.c = c0
        self.c_store = []
        
        self.n = n0 #self.h-self.q+1
        self.n_store = []
        self.h = self.n + self.q - 1
        self.h_store = []
        self.t = 0
        self.y = None
        
        self.G = np.eye(q)
        
    def add_data(self, y):
        e = y-self.m
        r = self.c/self.delta
        q_t = r + 1
        A = r/q_t
        self.m = (self.m + A*e)
        self.m_store.append(self.m) 
        self.c = r - (A**2)*q_t
        self.c_store.append(self.c) 
        self.D = self.beta*self.D + np.reshape(e, (self.q,1)) @ np.reshape(e, (1,self.q))/q_t
        self.D_store.append(self.D) 
        
        self.h = self.beta*self.h + 1
        self.h_store.append(self.h) 
        
        self.n = self.h-self.q + 1
        self.n_store.append(self.n) 
        
        self.t = self.t +1
        if self.y is None:
            self.y = y
        else:
            self.y = np.row_stack([self.y, y])
        
        #if set_w: self.set_w()
        
    def forecast_marginal(self,k=1,nsamps=1,params_only=False,mean_only=False): 
        #Gk = np.linalg.matrix_power(self.G,k)
        a = self.m
        r = self.c/self.delta
        q_t = r + 1
        
        df = self.beta*self.h - self.q + 1
        S = self.D/self.n
        S = (S + S.T)/2
        
        if mean_only:
            return a
        elif params_only: 
            return a,q_t,df,S
        else:
            samples = np.zeros((nsamps,self.q))
            
            rv = multivariate_t(a,q_t*S,df)
            return rv.rvs(size=nsamps)
        
    def forecast_path(self,k=1,nsamps=1): 
        
        samps = np.zeros((k,self.q,nsamps))
        for i in range(nsamps):
            mvdlm_copy = copy.deepcopy(self)
            for t in range(k):
                ysim = mvdlm_copy.forecast_marginal(k=1)
                mvdlm_copy.add_data(ysim)
                
                samps[t,:,i] = ysim
        
        return samps
                
# MVDLM with covariates (X)
class MVDLM_X:
    
    def __init__(self, beta, delta, q, p, m0=None, c0=None, n0 = 20, D0 = None):
        
        self.beta = beta
        self.delta = delta
        self.q = q #y dimension
        self.p = p #X dimension
        self.n = n0
        self.h = self.n + self.q - 1
        
        if m0 is None: m0 = np.zeros((self.p,self.q))
        if c0 is None: c0 = np.eye(self.p)
        if D0 is None: D0 = np.eye(self.q)*self.n
            
        self.M = m0
        self.C = c0
        self.D = D0
        
        self.F = np.empty((self.p,1))
        
        self.t = 0
        
        self.M_store = [self.M]
        self.C_store = [self.C]
        self.D_store = [self.D]
        self.h_store = [self.h]
        self.n_store = [self.n]
        
    
    def add_data(self,y,x):
        
        #collect values 
        self.F = np.reshape(x,(self.p,1))
        self.y = np.reshape(y,(self.q,1))
        
        #evolve posterior to prior
        self.R = self.C/self.delta
        
        #collect errors
        self.f = self.M.T @ self.F
        e = self.y - self.f
        self.q_t = (self.F.T @ self.R @ self.F) + 1
        A = self.R @ self.F / self.q_t
        
        #update 
        self.M = self.M + A @ e.T
        self.C = self.R - A @ A.T * self.q_t
        self.h = self.beta*self.h + 1
        self.n = self.h - self.q + 1
        self.D = self.beta * self.D + e @ e.T / self.q_t
        
        self.t = self.t + 1
        
        #store results
        self.M_store.append(self.M)
        self.C_store.append(self.C)
        self.D_store.append(self.D)
        self.h_store.append(self.h)
        self.n_store.append(self.n)
        
    def forecast_marginal(self,x,k=1,nsamps=1,Y=None,params_only=False,mean_only=False,log_likelihood=False): 
        
        #collect values 
        F = np.reshape(x,(self.p,1))
        R = self.C/self.delta
        
        f = self.M.T @ F
        q_t = (F.T @ R @ F) + 1
        
        df = self.beta*self.h - self.q + 1
        S = self.D/self.n
        S = (S + S.T)/2
                
        if mean_only: 
            return f
        elif params_only: 
            return f, q_t, df, S
        elif log_likelihood:
          rv = multivariate_t(f.reshape((self.q,)),q_t*S,df)
          return rv.logpdf(Y)
        else:
            
            rv = multivariate_t(f.reshape((self.q,)),q_t*S,df)
            return rv.rvs(size=nsamps)
    
    def forecast_path(self,x,k=1,nsamps=1): 
        x = np.reshape(x,(k,self.p))
        
        samps = np.zeros((k,self.q,nsamps))
        for i in range(nsamps):
            mvdlm_x_copy = copy.deepcopy(self)
            for t in range(k):
                ysim = mvdlm_x_copy.forecast_marginal(x=x[t,:],k=1,nsamps=1)
                mvdlm_x_copy.add_data(ysim,x[t,:])
                
                samps[t,:,i] = ysim
        
        return samps
        
        
        

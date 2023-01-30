import numpy as np
import pandas as pd
import time

from MVDLM import *

def get_prior(Y,X,t=None):
  
  if t is None: t = X.shape[1]+1
  if len(Y.shape) == 1: Y = Y.reshape((-1,1))

  X = X[0:t,:]
  Y = Y[0:t,:]
  
  vcov = np.linalg.inv(X.T @ X)
  betahat = vcov @ X.T @ Y
  
  sigma2 = (Y - X @ betahat).var(axis=0)
  
  return [betahat,vcov,sigma2]

def mv_synth(Y,X,prior_length=20,T=52,show_times=True,n_mc=200):
  
  start = time.perf_counter()
  
  #set prior from first t days
  prior = get_prior(Y,X,t=prior_length)
  p = X.shape[1]
  q = Y.shape[1]
  
  #initialize model
  synth_model = MVDLM_X(beta=0.99,delta=0.99,q=q,p=p,
                        m0=prior[0],c0=prior[1]*100,
                        n0=2,D0=2*np.diag(prior[2]))
                        
  #train the model, track log-likelihood
  ll_seq = np.zeros((T))
  for t in range(T):
    ll_seq[t] = synth_model.forecast_marginal(Y=Y[t,:],x=X[t,:],log_likelihood=True)
    synth_model.add_data(y=Y[t,:],x=X[t,:])
    
  end_train = time.perf_counter()
  
  #forecast from the model
  kmax = Y.shape[0]+1 - (T+1)
  mv_samples = synth_model.forecast_path(x=X[T:,:],k=kmax,nsamps=n_mc)
  end = time.perf_counter()
  
  if show_times:
    print(f'Training took {round((end_train-start),2)} seconds')
    print(f'Forecasting {n_mc} samples took {round((end-end_train),2)} seconds')
  return([mv_samples,synth_model,ll_seq])

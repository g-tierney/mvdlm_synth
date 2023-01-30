MVDLM Causal Forecasting
================
2023-01-29

# Introduction

This document will run the Causal Forecasting MVDLM model on simulated
data similar to the proprietary data in Tierney et al. (2023) paper
“Multivariate {B}ayesian Dynamic Modeling for Causal Prediction.” The
MVDLM is implemented in python and run via the `reticulate` R package.
This code extends that used in Emily Tallman’s implementation
[here](https://github.com/emtall/MVDLM). Python 3.9 is required to
simulate from multivariate $t$ distributions.

# Simulate Data

The code below shows the simulated data process. We simulate 3 treated
series with positive and negative correlations. We also simulate 6
control series, only the first four of which are relevant to the treated
series. We fit two models, one which excludes the irrelevant series and
one that includes them to verify that the BMA procedure will correctly
identify the optimal model.

``` r
n_pre <- 52
n_post <- 16
n_total <- n_pre + n_post

n_factors <- 2
n_control <- 6
n_treat <- 3

tau <- 2

set.seed(1)

### Simulate Data ###

#control stores via AR(1) with AR coefficient of 0.97
control_outcomes <- rnorm(n_control*n_total) %>% 
  matrix(nrow=n_total,ncol=n_control) %>% 
  apply(2,function(x){
    for(t in 2:length(x)){
      x[t] <- .97*x[t-1] + x[t]
    }
    x
  }) %>% 
  as_tibble(.name_repair = "unique") %>% 
  rename_with(~str_c("C",1:6))

#constant Theta
Theta <- rnorm(n_control*n_treat)/sqrt(3) %>% 
  matrix(nrow=n_control,ncol = n_treat) 
Theta[5:6,] <- 0 #final two have no impact
Theta[3,1:2] <- 0

#constant Sigma
Sigma <- diag(1,nrow=n_treat,ncol=n_treat)
Sigma[1,2] = Sigma[2,1] <- .4
Sigma[1,3] = Sigma[3,1] <- .3
Sigma[2,3] = Sigma[3,2] <- -.2

#treatment stores 
Y0 <- matrix(NA,nrow = n_total,ncol = n_treat)
Y1 <- matrix(NA,nrow = n_total,ncol = n_treat)

for(t in 1:n_total){
  Y0[t,] = Y1[t,] <- (as.matrix(control_outcomes[t,],nrow=1) %*% Theta) + 
    c(99:101) + 
    mvtnorm::rmvnorm(1,rep(0,n_treat),Sigma)
  if(t>n_pre){
    Y1[t,] <- Y1[t,] + tau
  }
}

#observed Y
Y <- rbind(Y0[1:n_pre,],Y1[(n_pre+1):n_total,])

if(!dir.exists("data")) dir.create("data")
write_csv(control_outcomes,"data/sim_control_outcomes.csv")
write_csv(Y %>% as_tibble() %>% rename_with(~str_c("T",1:3)),
          "data/sim_treatment_outcomes.csv")
```

# Causal Forecasting

Next, we show how to call the main workhorse function `mv_synth` to run
the analysis.

``` r
'
import sys
sys.path.append("mvdlm")

import pandas as pd
import numpy as np

from synth_functions import get_prior, mv_synth

X_df = pd.read_csv("data/sim_control_outcomes.csv")
X = X_df.to_numpy()
X = np.hstack([np.ones((X.shape[0],1)),X])
X_correct = X[:,:5]

Y_df = pd.read_csv("data/sim_treatment_outcomes.csv")
Y = Y_df.to_numpy()

#run on full data
mv_samples_full = mv_synth(Y,X,T=52,n_mc=2000)
#run on correctly specified data
mv_samples = mv_synth(Y,X_correct,T=52,n_mc=2000)
' -> py_code

py_run_string(py_code)
```

The resulting object `mv_samples` is a list of three items. The first is
an array of samples from the posterior with dimensions (number of
post-intervention time points) x (number of treated units) x (number of
MC samples). The second is the mvdlm object, which contains the final
and sequence of parameter estimates. The third is the sequence of
one-step predictive log-likelihoods for each pre-intervention time
point. The next section shows how to use each of these features.

# Analyze results

First, we do model comparison based on BMA weights for the correctly
specified model (excluding the last two predictors) and the full model
(including the two irrelevant predictors). The BMA weights quickly
identify the correct model. All subsequent analysis will focus only on
this model.

``` r
bma_probs <- tibble(t=1:n_pre,mfull_onestep = py$mv_samples_full[[3]],mtrue_onestep = py$mv_samples[[3]]) %>% 
  mutate(mfull_lcum = cumsum(mfull_onestep),
         mtrue_lcum = cumsum(mtrue_onestep),
         mfull_prob = exp(mfull_lcum)/(exp(mfull_lcum) + exp(mtrue_lcum)),
         mtrue_prob = exp(mtrue_lcum)/(exp(mfull_lcum) + exp(mtrue_lcum)),
         t_plot = t-53) %>% 
  pivot_longer(cols = -starts_with("t"),names_sep = "_",names_to = c("model","quantity")) %>%
  filter(quantity != "lcum") %>% 
  mutate(model = ifelse(model=="mfull","Full Model","Correct Model"),
         quantity = ifelse(quantity == "onestep","One-Step Log Likelihood","Cumulative Model Probability")) 

bma_probs %>% 
  ggplot(aes(x=t_plot,y=value,color = model)) + 
  geom_line() + 
  facet_wrap(~ quantity,scales = "free_y",nrow = 2) + 
  labs(x="Time until Treatment",y = NULL)
```

![](example_application_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

Next, we show the posterior mean for $\Theta_T$ and harmonic mean for
$\Sigma_T$ at the end of the pre-intervention period (when the model is
frozen) and compare them to the true values.

``` r
#Compare estimated and true Theta
py$mv_samples[[2]]$M
```

    ##             [,1]        [,2]         [,3]
    ## [1,] 99.10706797 99.76799847 100.87275701
    ## [2,]  0.88232152 -0.28157453   1.09528665
    ## [3,] -0.54680874  0.38828616  -0.59332995
    ## [4,]  0.02282678  0.01249907  -0.01499091
    ## [5,]  0.41657447 -0.13917360  -0.99933094

``` r
rbind(matrix(99:101,nrow=1),Theta[1:4,])
```

    ##            [,1]        [,2]         [,3]
    ## [1,] 99.0000000 100.0000000 101.00000000
    ## [2,]  0.8324354  -0.2933437   0.82853780
    ## [3,] -0.4027239   0.3023125  -0.41013297
    ## [4,]  0.0000000   0.0000000  -0.03756678
    ## [5,]  0.3767421  -0.1450099  -1.01582975

``` r
#Compare estimated and true Sigma
Sigma
```

    ##      [,1] [,2] [,3]
    ## [1,]  1.0  0.4  0.3
    ## [2,]  0.4  1.0 -0.2
    ## [3,]  0.3 -0.2  1.0

``` r
py$mv_samples[[2]]$D/py$mv_samples[[2]]$n
```

    ##           [,1]       [,2]       [,3]
    ## [1,] 0.8834864  0.5560247  0.1133924
    ## [2,] 0.5560247  1.2495542 -0.3268920
    ## [3,] 0.1133924 -0.3268920  1.1772993

Now we analyze the actual posterior samples. First we plot the observed
(dashed) and counterfactual forecast (solid) for each treatment store.

``` r
samples <- py$mv_samples[[1]] 
#examine time-varying effects
weekly_results <- tibble()
for(s in 1:3){
  output <- samples[,s,] %>% apply(1,quantile,probs=c(0.025,0.50,0.975)) %>% 
    t %>% 
    as_tibble() %>% 
    rename_with(~c("lb","med","ub")) %>% 
    mutate(t=1:16,store = str_c("T",s)) %>% 
    select(store,t,everything())
  weekly_results <- weekly_results %>% 
    bind_rows(output)
}

Y_long <- py$Y_df %>% 
  mutate(t=1:n_total - n_pre + 1) %>% 
  pivot_longer(cols = -t,names_to = "store",values_to ="observed")

weekly_results <- bind_rows(weekly_results,Y_long) %>% 
  mutate(method = ifelse(is.na(observed),"Forecast","Observed"),
         med = ifelse(is.na(observed),med,observed)) 

weekly_results %>% 
  filter(t>0) %>% 
  ggplot(aes(x=t,y=med,ymin=lb,ymax=ub,color=store,linetype=method)) + 
  geom_point() + geom_line() + #geom_ribbon(alpha=.2) + 
  labs(x="Weeks Post Intervention",y = "Weekly Outcomes") + 
  facet_wrap(~store)
```

![](example_application_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

Next, we compute the store-level percent lift over the post-intervention
period.

``` r
store_results <- samples %>% 
  apply(c(2,3),sum) %>% 
  apply(2, function(s) (colSums(Y[(n_pre+1):n_total,]) - s)/s) %>% 
  apply(1, quantile,probs = c(0.025,0.50,0.975)) %>% 
  t %>% 
  as_tibble() %>% 
  rename_with(~c("lb","med","ub")) %>% 
  mutate(store = str_c("T",1:3))

store_results %>% 
  ggplot(aes(x=store,y=med,ymax=ub,ymin=lb)) + 
  geom_point() + 
  geom_errorbar(width=.2) +
  geom_hline(yintercept = 0) + 
  scale_y_continuous(labels = scales::label_percent()) + 
  labs(x = "Treatment Store",y = "Percent Lift")
```

![](example_application_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

And finally, we show the average percent lift across stores using either
our model of a Multivariate DLM or the results of fitting independent
DLMS by randomly shuffling the store-specific effects across Monte Carlo
draws before averaging. As expected, the multivariate methods give wider
credible intervals, reflecting the correlated uncertainty regarding
counter-factual outcomes.

``` r
agg_results <- samples %>% 
  apply(c(2,3),sum) %>% 
  apply(2, function(s) (colSums(Y[(n_pre+1):n_total,]) - s)/s) %>% 
  apply(2, mean) %>% 
  quantile(probs = c(0.025,0.50,0.975)) %>% 
  t %>% 
  as_tibble() %>% 
  rename_with(~c("lb","med","ub")) %>% 
  mutate(type = "Multivariate DLM")

agg_results_indep <- samples %>% 
  apply(c(2,3),sum) %>% 
  apply(2, function(s) (colSums(Y[(n_pre+1):n_total,]) - s)/s) %>% 
  apply(1, sample) %>% 
  apply(2, mean) %>% 
  quantile(probs = c(0.025,0.50,0.975)) %>% 
  t %>% 
  as_tibble() %>% 
  rename_with(~c("lb","med","ub")) %>% 
  mutate(type = "Independent DLMs")

bind_rows(agg_results,agg_results_indep) %>% 
  ggplot(aes(x=type,y=med,ymax=ub,ymin=lb)) + 
  geom_point() + 
  geom_errorbar(width=.2) +
  geom_hline(yintercept = 0) + 
  scale_y_continuous(labels = scales::label_percent()) + 
  labs(x = "Model Type",y = "Average Percent Lift")
```

![](example_application_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

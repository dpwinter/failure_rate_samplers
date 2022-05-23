import scipy.stats as stats
from scipy.special import factorial
import numpy as np

def comb(n,k):
    return factorial(n) / (factorial(k) * factorial(n-k))

def binom(k,n,p):
    return comb(n,k) * p**k * (1-p)**(n-k)

def Wilson_std(p, N, z=1.96):
    # Estimator of standard deviation of binomial distribution according to Wilson
    wilson_max = (p + z**2/(2*N) + z*np.sqrt(p*(1-p)/N+z**2/(4*N**2)))/(1+z**2/N)
    wilson_min = (p + z**2/(2*N) - z*np.sqrt(p*(1-p)/N+z**2/(4*N**2)))/(1+z**2/N)
    return wilson_max - wilson_min

def Wilson_var(p, N):
    return Wilson_std(p, N)**2

def Wald_var(p, N):
    # Wald variance estimator (known issues, better use Wilson)
    return p * (1-p) / N

def Wald_std(p, N):
    return np.sqrt(Wald_var(p, N))

def std_sum(Aws, pws, n_samples, var=Wilson_var):
    # Returns standard deviation according to Gaussian error propagation 
    # applied to p_L: V_L = sum_i (dp_L/dp_i)^2 * V_i
    return np.sqrt( np.sum( Aws[1:]**2 * var(pws[1:],n_samples), axis=0 ) )

def balanced_SS_selector(counts, *args, **kwargs):
    # Return index of least sampled SS
    return np.argmin(counts)

def ERV_SS_selector(counts, fail_counts, var=Wilson_var, *args, **kwargs):
    # Return index of SS which yields maximum ERV

    p = fail_counts / counts # list of SS failure rates
    v = var(p, counts) # list of variances

    # prospective failure rates
    p_p = (fail_counts+1) / (counts+1) # next msmt yields +1 (i.e. 1)
    p_m = fail_counts / (counts+1) # next msmt yields -1 (i.e. 0)

    # prospective variances
    v_p = var(p_p, counts+1) 
    v_m = var(p_m, counts+1)

    # Calculate prospective variances
    v_prop = p*v_p + (1-p)*v_m 

    # Differences to current variance
    # The larger delta, the smaller v_prop.
    delta = v - v_prop

    # Maximize the difference
    return np.argmax(delta)

def Metropolis_SS_selector(counts, fail_counts, curr_idx, a=5, b=5, *args, **kwargs):
    # Not fully understood yet!
    # Also this implementation needs
    # a) fail counts and counts to be non-zero
    # b) virtual samples must be removed after MC loop
    # c) must pass an initial index that is fed back in next iteration
    # Overall, current implementation rather ugly and not ideal
    # But: the binning of samples looks more promising than ERV
    # i.e. samples a lot more in relevant subsets than ERV

    cnts = counts + 2
    fcnts = fail_counts + 1

    # Return index of SS sampled from posterior over SS config space

    # Propose new index
    prop_idx = np.random.choice(len(counts))

    k_curr, k_prop = fcnts[curr_idx], fcnts[prop_idx]
    n_curr, n_prop = cnts[curr_idx], cnts[prop_idx]

    # Calculate SS failure rates
    # pw_curr = k_curr / n_curr
    # pw_prop = k_prop / n_prop
    p_L = np.sum(fcnts) / np.sum(cnts)

    # Compute Bayes numerators
    # posterior_prop = binom.pmf(k_prop, n_prop, p_L) * beta.pdf(pw_prop, a, b)
    # posterior_curr = binom.pmf(k_curr, n_curr, p_L) * beta.pdf(pw_curr, a, b)
    likelihood_prop = stats.binom.logpmf(k_prop, n_prop, p_L)
    likelihood_curr = stats.binom.logpmf(k_curr, n_curr, p_L) # avoid pmf for large n,k
    p_accept_prop = min(likelihood_prop / likelihood_curr, 1.0)

    # Probabilistic acceptance rule
    # If ratio is > 1.0 always accept the proposed subset, else probabilistically
    # p_accept_prop = min(posterior_prop / posterior_curr, 1.0)

    if p_accept_prop > np.random.uniform():
        return prop_idx
    else:
        return curr_idx

def weight_cutoff(p_max, delta_max, n_circ_elems):
    # Return weight cutoff at p_max for delta_max.
    delta = 1
    for w_max in range(n_circ_elems+1):
        # delta -= binom.pmf(w_max, n_circ_elems, p_max)
        delta -= binom(w_max, n_circ_elems, p_max)
        if delta < delta_max: break
    return w_max

from scipy.stats import binom, beta
import numpy as np

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
    # Return index of SS sampled from posterior over SS config space

    # Propose new index
    prop_idx = np.random.choice(len(counts))

    k_curr, k_prop = fail_counts[curr_idx], fail_counts[prop_idx]
    n_curr, n_prop = counts[curr_idx], counts[prop_idx]

    # Calculate SS failure rates
    pw_curr = k_curr / n_curr
    pw_prop = k_prop / n_prop

    # Compute Bayes numerators
    posterior_prop = binom.pmf(k_prop, n_prop, pw_prop) * beta.pdf(pw_prop, a, b)
    posterior_curr = binom.pmf(k_curr, n_curr, pw_curr) * beta.pdf(pw_curr, a, b)

    # Probabilistic acceptance rule
    # If ratio is > 1.0 always accept the proposed subset, else probabilistically
    p_accept_prop = min(posterior_prop / posterior_curr, 1.0)
    if p_accept_prop > np.random.uniform():
        return prop_idx
    else:
        return curr_idx

def weight_cutoff(p_max, delta_max, n_circ_elems):
    # Return weight cutoff at p_max for delta_max.
    delta = 1
    for w_max in range(n_circ_elems+1):
        delta -= binom.pmf(w_max, n_circ_elems, p_max)
        if delta < delta_max: break
    return w_max

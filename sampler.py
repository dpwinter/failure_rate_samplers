import itertools as it
from simulator import StabSim
import numpy as np
from error_gen import DepolarGen
import circuit_utils
import calc

class Sampler():
    '''Generic Sampler base class'''
    
    def __init__(self, circuit, err_params):
        self.circuit = circuit
        self.n_qubits = circuit_utils.n_qubits(circuit)
        self.partitions = circuit_utils.partition_circuit(circuit, err_params.keys())
        self.p_phys_mat = np.vstack(list(err_params.values())).T # p_phy_range x partitions

    def _sample(self, err_gen=None, p_phys=None):
        sim = StabSim(self.n_qubits)
        err_circuit = None if err_gen==None else err_gen.generate(self.partitions, p_phys)
        msmt = circuit_utils.run_circuit(sim, self.circuit, err_circuit)
        return msmt

    def _check_logical_failure(self, msmt):
        return 1 if msmt else 0

class DirectSampler(Sampler):
    '''Direct Monte Carlo sampler'''

    def __init__(self, circuit, err_params):
        super().__init__(circuit, err_params)
        choose_fn = lambda l,p: set(i for i in l if np.random.random() < p)
        self.err_gen = DepolarGen(choose_fn=choose_fn, n_ticks=len(circuit))

    def run(self, n_samples=100, var=calc.Wilson_var):
        fail_cnts = np.zeros((self.p_phys_mat.shape[0])) # one fail counter per p_phys

        for i, p_phys in enumerate(self.p_phys_mat):
            for _ in range(n_samples):
                msmt = self._sample(self.err_gen, p_phys)
                fail_cnts[i] += self._check_logical_failure(msmt)

        p_L = fail_cnts / n_samples
        std = np.sqrt( var(p_L, n_samples) )
        return p_L, std

class SubsetSampler(Sampler):
    '''Monte Carlo subset sampler'''
    
    def __init__(self, circuit, err_params):
        super().__init__(circuit, err_params)
        choose_fn = lambda l,k: [l[i] for i in np.random.choice(len(l),k,replace=False)]
        self.err_gen = DepolarGen(choose_fn=choose_fn, n_ticks=len(circuit))
            
    def run(self, p_max, delta_max, n_samples=20, SS_sel_fn=calc.balanced_SS_selector):
        
        # Find w_max vector for delta_max at p_max
        assert len(p_max) == len(delta_max) == len(self.partitions)
        w_cutoff = np.zeros_like(p_max, dtype=int)
        for i in range(len(w_cutoff)):
            w_cutoff[i] = calc.weight_cutoff(p_max[i], delta_max[i], len(self.partitions[i]))
        
        # Generate weight vectors (combis of all ws up to each w_max per partition)
        w_vecs = list(it.product( *[list(range(w_cutoff_i+1)) for w_cutoff_i in w_cutoff] ))

        ### Numerical part ###
        # Generate 1D list of subset failure rates per weight vector combi
        cnts      = np.zeros((len(w_vecs))) + 1 # one virtual sample to avoid div0
        fail_cnts = np.zeros((len(w_vecs)))
        for i in range(n_samples):
            idx = SS_sel_fn(cnts, fail_cnts)
            choice_hist.append(idx) # debug
            w_vec = w_vecs[idx]
            msmt = self._sample(self.err_gen, w_vec)
            fail_cnts[idx] += self._check_logical_failure(msmt)
            cnts[idx] += 1
        pws = (fail_cnts / cnts)[:,None]
        ###

        ### Analytical part ###
        # Calculate 3D tensor of binom. coefs for each partition
        partition_lens = np.array([len(p) for p in self.partitions])
        Aws = np.array([calc.binom.pmf(w_vec, partition_lens, self.p_phys_mat) for w_vec in w_vecs])
        Aws = np.product(Aws, axis=-1) # get list of Aw for each partition SS combination

        # Calculate failure rates and std
        p_L_low = np.sum(Aws * pws, axis=0)
        p_L_up = p_L_low + 1 - np.sum(Aws, axis=0)
        std = calc.std_sum(Aws, pws, n_samples)
        ###

        return p_L_low, p_L_up, std
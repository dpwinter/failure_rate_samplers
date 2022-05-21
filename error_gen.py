import numpy as np
import itertools as it
from circuit import Circuit

class DepolarGen():
    '''Depolar error generation for circuit level noise.'''
    one_qb_errs = ['X','Y','Z'] # Pauli errors
    two_qb_errs  = list(it.product(one_qb_errs + ["I"], repeat=2)) # variation with repetition
    two_qb_errs.remove(('I','I')) # remove II as it is not a fault
    
    def __init__(self, choose_fn, n_ticks):
        self.choose_fn = choose_fn # function used to generate errors in circuit partitions
        self.n_ticks = n_ticks # length of entire circuit
    
    def generate(self, partitions, values):

        err_circ = Circuit([None] * self.n_ticks)
        for errs in [self.choose_fn(p,v) for p, v in zip(partitions, values)]: # generate errors
            for (tick, qbs) in errs:
                if type(qbs) == int:
                    err_circ[tick] = (np.random.choice(self.one_qb_errs), qbs)
                elif type(qbs) == tuple:
                    tick_faults = []
                    for (fault, qb) in zip(self.two_qb_errs[np.random.choice(len(self.two_qb_errs))], qbs):
                        if fault != "I":
                            tick_faults.append((fault,qb))
                    err_circ[tick] = tick_faults


        return err_circ # circuit with sum(values) errors

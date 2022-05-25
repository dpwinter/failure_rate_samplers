import random
import numpy as np

class ChpSimulator:
    """ Implementation of Aaronson/Gottesman CHP simulator.
    Updates stabilizer generators acted upon by gates of
    the Clifford group: CNOT, Hadamard and Phase (S).
    Ref: https://arxiv.org/pdf/quant-ph/0406196.pdf.
    """

    def __init__(self, n_qbs):
        self.n = n_qbs
        # tableau: X|Z, generators in rows, qubits in columns
        # Track destabilizers and stabilizers in 2*n rows
        # Destabilizers makes measurement more efficient
        self._table = np.eye(2 * n_qbs + 1, dtype=bool)
        self._x = self._table[:, :self.n]
        self._z = self._table[:, self.n:-1]
        self._r = self._table[:, -1] # track +/-1 global phase

    def cnot(self, c, t):
        self._r ^= self._x[:, c] & self._z[:, t] & (self._x[:, t] ^ self._z[:, c] ^ True)
        # 2 cases for sign flips: Y_c & Y_t and X_c & Z_t, the above is most efficient
        self._x[:, t] ^= self._x[:, c] # X propagates forward: c->t
        self._z[:, c] ^= self._z[:, t] # Z propagates backward: t->c

    def hadamard(self, i): # swap X and Z bases
        self._r ^= self._x[:, i] & self._z[:, i] # 0<->+, 1<->- no phase change
        # but +i<->-i. Thus phase change for qb on Y-axis
        # swap X-col with Z-col for a qb: 0110|1010 (column vecs)
        self._x[:, i] ^= self._z[:, i]  # 1100|1010
        self._z[:, i] ^= self._x[:, i]  # 1100|0110
        self._x[:, i] ^= self._z[:, i]  # 1010|0110

    def phase(self, i):
        # Phase gate: [[1,0],[0,i]], i.e. rotate 90 arount Z-axis
        self._r ^= self._x[:, i] & self._z[:, i]  # phase change after leaving Y-axis
        self._z[:, i] ^= self._x[:, i]  # stabilized by X or Y, not by Z.

    def measure(self, i):
        # If one generator has X in i-th column (i-th qubit) the stabilizer state
        # includes a |+> (or |+i>) for the i-th qubit. If this is the case, the state
        # must be a X or Y +1 eigenstate on this qubit, thus not in Z basis.
        # A msmt will give random outcome and thus be not deterministic.
        for p in range(self.n):
            if self._x[p+self.n, i]: # all X-ops for qubit i in stab gens (excl. destabs)
                return self._measure_random(i, p) # p: destab corresp. to 1st gen w X in i-th row
        return self._measure_determined(i) # no X's found in i-th row of stab gens

    def _measure_random(self, i, p, bias = 0.5): # p: index of first destab
        # If i-th qubit is eigenstate of X (or Y) basis,
        # After msmt the state is projected onto Z, need to update state (gens)
        self._table[p, :] = self._table[p+self.n, :] # set destab equal to stab (proj. onto Z)
        self._table[p+self.n, :] = 0 # delete stab
        self._z[p+self.n, i] = 1 # project to Z: eigenstate of Z basis
        self._r[p+self.n] = random.random() < bias # decide randomly if + or -1 eigenstate

        # Update the other stabs and destabs that have x in i-th row
        # those entries must become either I or Z (not X), since the stab state is in
        # Z-basis for this qubit. This we accomplish by multiplying all those rows
        # with the destabilizer p. This also changes the signs correctly.
        for j in range(2*self.n): 
            if self._x[j, i] and j != p and j != p+self.n:
                self._row_mult(i, p) 
        return MeasureResult(value=self._r[p+self.n], determined=False)

    def _measure_determined(self, i):
        self._table[-1, :] = 0 # set scratch space to 0
        for j in range(self.n): # go through destabilizer rows
            if self._x[j, i]:   # for each destab that is X at i-th position
                self._row_mult(-1, j + self.n) # multiply scratch row with corresponding stab
        return MeasureResult(value=self._r[-1], determined=True) # r[-1] gives msmt result

    def _row_product_sign(self, i, k):
        # 2*r_h + 2*r_i + sum(g(x1,x2,z1,z2)) == 0 or 2 (mod 4)
        # Find +1 = 0 (mod 4) or -1 = 2 (mod 4) sign of row multiplication
        # 3 parts: sign of stab 1 (r_h), sign of stab 2 (r_i) and 
        # sign of individual pauli-products in tensor product. This can only
        # be +/-1 because any imaginary unit from Pauli product will vanish, as
        # all stabilizers commute: we always have an even number of i's
        pauli_phases = sum( pauli_product_phase(self._x[i,j], self._z[i,j], 
            self._x[k,j], self._z[k,j]) for j in range(self.n) )
        p = (pauli_phases >> 1) & 1
        return bool(self._r[i] ^ self._r[k] ^ p)

    def _row_mult(self, i, k):
        # 1. Get sign of multiplication by only looking at r
        self._r[i] = self._row_product_sign(i,k)
        # 2. Do XOR operation between X and Z separately, ignoring signs
        self._x[i, :self.n] ^= self._x[k, :self.n]
        self._z[i, :self.n] ^= self._z[k, :self.n]

    def __str__(self):
        _cell = lambda row, col: ['.','X','Z','Y'][int(self._x[row,col])+2*int(self._z[row,col])]
        _row = lambda row: ('-' if self._r[row] else '+') +''.join([str(_cell(row, col)) for col in range(self.n)])
        z_obs = [_row(row) for row in range(self.n)]
        sep = ['-' * (self.n + 1)]
        x_obs = [_row(row) for row in range(self.n, 2*self.n)]
        return '\n'.join(z_obs + sep + x_obs)

def pauli_product_phase(x1, z1, x2, z2):
    """Determine power of i in product of two Paulis (X|Z)
    Returns: (0,-1,+1) for (1,-i,+i)
    Cases:  0: PP=I, IP=P, PI=P (P=any Pauli)
           +1: XY=iZ, YZ=iX, ZY=iX
           -1: XZ=-iY, YX=-iZ, ZX=-iY
    """
    
    if x1 and z1: return int(z2) - int(x2) # YZ, YX, YI
    elif x1: return z2 and 2 * int(x2) - 1 # XZ, XY, XI
    elif z1: return x2 and 1 - 2 * int(z2) # ZX, ZY, ZI
    else: return 0 # PP, IP

class MeasureResult:
    def __init__(self, value, determined):
        self.value = bool(value)
        self.determined = bool(determined)

    def __bool__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, (bool, int)):
            return self.value == other
        if isinstance(other, MeasureResult):
            return self.value == other.value and self.determined == other.determined
        return NotImplemented

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '{} ({})'.format(int(self.value), ['random', 'determined'][self.determined])

class StabSim(ChpSimulator):
    '''Wrapper class for ChpSim, implements useful gates.'''

    def h(self, i):
        self.hadamard(i)

    def z(self, i):
        self.phase(i)
        self.phase(i)

    def x(self, i):
        self.h(i)
        self.z(i)
        self.h(i)

    # def y(self, i):
    #     self.h(i)
    #     self.phase(i)
    #     self.h(i)

    def phase_adjoint(self, i):
        self.phase(i)
        self.phase(i)
        self.phase(i)

    def y(self, i):
        self.phase_adjoint(i)
        self.x(i)
        self.phase(i)

    def _apply_gate(self, sym, qbs):
        gate = getattr(self, sym.lower())
        args = (qbs,) if type(qbs)==int else qbs
        return gate(*args)

    def run(self, circuit, err_circuit=None):
        msmts = []
        for tick_idx, tick in enumerate(circuit):
            if type(tick) == list:
                for sub_tick in tick:
                    res = self._apply_gate(*sub_tick)
                    if res: 
                        msmts.append((tick_idx,res))
            elif type(tick) == tuple:
                res = self._apply_gate(*tick)
                if res: 
                    msmts.append((tick_idx,res))
            
            if err_circuit:
                err_tick = err_circuit[tick_idx]
                if err_tick:
                    if type(err_tick) == list:
                        for sub_err_tick in err_tick:
                            self._apply_gate(*sub_err_tick)
                    elif type(err_tick) == tuple:
                        self._apply_gate(*err_tick)
        return msmts

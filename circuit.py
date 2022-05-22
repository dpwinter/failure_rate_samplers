from collections.abc import MutableSequence

class Circuit(MutableSequence):
    '''Representation of a quantum circuit

    Consists of a list of ticks. One tick can be
    either a (gate, qubit) tuple or a list of those
    representing parallel gate application per tick.
    Note: For n-qb gates `qubit` is a n-tuple.
    '''

    def __init__(self, ticks=[]):
        self._ticks = ticks

    def __getitem__(self, tick_idx):
        return self._ticks[tick_idx]

    def __setitem__(self, tick_idx, tick):
        self._ticks[tick_idx] = tick

    def __delitem__(self, tick_idx):
        del self._ticks[tick_idx]

    def __len__(self):
        return len(self._ticks)

    def __str__(self):
        str_list = []
        for i, tick in enumerate(self._ticks):
            str_list.append("%i: " % i + str(tick))
        return "\n".join(str_list)

    def __repr__(self):
        return self.__str__()

    def _iter_qubits(self):
        for tick_idx,tick in enumerate(self._ticks):
            if type(tick) == list:
                for sym,qbs in tick:
                    yield tick_idx, sym.upper(), (qbs,) if type(qbs) == int else qbs
            elif type(tick) == tuple:
                sym,qbs = tick
                yield tick_idx, sym.upper(), (qbs,) if type(qbs) == int else qbs

    def insert(self, tick_idx, tick):
        self._ticks.insert(tick_idx, tick)

    @property
    def n_qubits(self):
        qubits = []
        for _,_,qbs in self._iter_qubits():
            qubits += qbs
        return max(qubits)+1

    def partition(self, gategroup):
        return [(tick_idx,qbs) for tick_idx,sym,qbs in self._iter_qubits() if sym in gategroup]


one_qb_gates = {'H','X','Z'} # (incomplete) list of elements in p1 partition
two_qb_gates = {'CNOT'} # (incomplete) list of elements in p2 partition

def partition_elements(circuit: list, partition: set) -> list: 
    # Return list of (tick,qb) tuples for all found elments
    # in circuit belonging to the given partition
    locs = []
    for i, (sym, qbs) in enumerate(circuit):
        if sym.upper() in partition:
            if sym.upper in two_qb_gates:
                locs += [(i,qb) for qb in qbs]
            else:
                locs += [(i,qbs)]
    return locs

def partition_circuit(circuit: list, keys: list) -> list:
    # Return list of lists of partition elements corresponding
    # the the given partition keys
    partitions = {
        "p":  partition_elements(circuit, one_qb_gates.union(two_qb_gates)),
        "p1": partition_elements(circuit, one_qb_gates),
        "p2": partition_elements(circuit, two_qb_gates)
        }
    return [partitions[k] for k in keys]

def run_circuit(sim, circuit: list, err_circuit=None) -> list:
    # Run a circuit, return list of results or empty list
    out = []
    for tick,gate in enumerate(circuit):
        res = sim.run(gate)
        if res: 
            out.append(res)
        if err_circuit:
            sim.run(err_circuit[tick])
    return out

def n_qubits(circuit: list) -> int:
    # Returns number of qubits in a given circuit
    locs = []
    for sym,qbs in circuit:
        if sym.upper() in two_qb_gates:
            locs += qbs
        else:
            locs += [qbs]
    return max(locs) + 1

import pickle

from qiskit import QuantumCircuit
import qiskit.ignis.verification.randomized_benchmarking as rb


def perform_some_computations():
    rb_opts = {}
    # Number of Cliffords in the sequence
    rb_opts['length_vector'] = [1]
    # Number of seeds (random sequences)
    rb_opts['nseeds'] = 1
    rb_opts['rb_pattern'] = calculate_pattern()

    rb_circs, _ = rb.randomized_benchmarking_seq(**rb_opts)
    qregs = rb_circs[0][0].qregs
    # cregs = rb_circs[0][0].cregs
    qc = QuantumCircuit(*qregs)
    for i in rb_circs[0][0][0:-NUMBER_OF_CODE_QUBITS]:
        qc.data.append(i)
    return qc


NUMBER_OF_CODE_QUBITS = 7
RANDOMIZED_BENCHMARKING_FILE = "rb"


def calculate_pattern():
    if NUMBER_OF_CODE_QUBITS == 7:
        return [[0, 1], [2, 3], [4, 5], [6]]
    if NUMBER_OF_CODE_QUBITS == 9:
        return [[0, 1], [2, 3], [4, 5], [6, 7], [8]]


if __name__ == "__main__":
    qc = perform_some_computations()
    filehandler = open(RANDOMIZED_BENCHMARKING_FILE, 'wb')
    pickle.dump(qc, filehandler)

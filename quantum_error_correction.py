from qiskit import QuantumRegister, ClassicalRegister, transpile
from qiskit import execute, Aer
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
import matplotlib.pyplot as plt
from qiskit.providers.aer.noise import NoiseModel, thermal_relaxation_error, ReadoutError

from randomized_benchmarking import *
from surface.benchmarking_tools import SurfaceCodeBenchmarkingTool
from surface.circuits import SurfaceCodeLogicalQubit
from surface.fitters import GraphDecoder

""" 
Some notes:
- first qubit is always initialized to 0. At the end of the circuit, the first qubit is measured. The accuracy is
the % of measurements in which the readout is 0.

- with the params in this commit, the accuracy is:
without correction: 0.65
steane code: 0.77
rotated surface code: 0.9

- the surface code is rather slow. TBD

- in this commit, TWO_QUBIT_GATE_ERROR isn't applied. TBD


"""

class Codes:
    NO_CORRECTION = 1
    STEANE = 7
    SURFACE1 = 9  # rotated surface code with 9 physical qubits representing one logical qubit. d=3
    # SURFACE2 =  # to be implemented, d=5


ONE_QUBIT_GATE_ERROR = 0.1
TWO_QUBIT_GATE_ERROR = 0.0304
READOUT_ERROR = 0.005
RANDOMIZED_BENCHMARKING = True
NOISE = True

CODE = Codes.STEANE
NUMBER_OF_CODE_QUBITS = CODE
NUMBER_ANCILLAS_X = int(NUMBER_OF_CODE_QUBITS / 2)


def main():
    aer_sim = Aer.get_backend('aer_simulator')
    quantum_circuit = generate_circuit()
    noise_model = create_noise_model()
    run(quantum_circuit, noise_model, aer_sim)


def run(quantum_circuit, noise_model, aer_sim):
    if CODE in [Codes.STEANE, Codes.NO_CORRECTION]:
        counts = run_simulations_for_circuit(aer_sim, quantum_circuit, noise_model)
        report_results(counts)
        return
    run_surface_code(aer_sim, noise_model, quantum_circuit)


def run_surface_code(aer_sim, noise_model, quantum_circuit):
    quantum_circuit = transpile(quantum_circuit, aer_sim, optimization_level=0)
    draw_circuit(quantum_circuit)
    counts = execute(quantum_circuit, backend=aer_sim, noise_model=noise_model, shots=200).result().get_counts()
    # print(counts)
    code_distance = 3 if CODE == Codes.SURFACE1 else 5
    benchmarking_tool = SurfaceCodeBenchmarkingTool(
        decoder=GraphDecoder(d=code_distance, T=1),
        readout_circuit=quantum_circuit
    )
    accuracy = 1 - benchmarking_tool.logical_error_rate(counts, correct_logical_value=0)
    print("accuracy: " + str(accuracy))


def create_noise_model():
    noise_model = NoiseModel()
    if NOISE:
        add_one_gate_error(noise_model)
        # add_two_gate_error(noise_model)
        add_readout_error(noise_model)
        noise_model.add_basis_gates(['u1', 'u2', 'u3', 'cx'])
    return noise_model


def add_readout_error(noise_model):
    readout_error_prob = READOUT_ERROR
    readout_error = ReadoutError(
        [[1 - readout_error_prob, readout_error_prob], [readout_error_prob, 1 - readout_error_prob]])
    noise_model.add_readout_error(readout_error, [0])  # add readout error to 1st qubit


def add_two_gate_error(noise_model):
    depolarizing_error_2 = depolarizing_error(TWO_QUBIT_GATE_ERROR, 2)
    t1 = 10e10
    t2 = 2e8
    two_qubit_gate_time = 2.1e5
    thermal_relaxation_error_2 = thermal_relaxation_error(t1, t2, two_qubit_gate_time)
    thermal_relaxation_error_2 = thermal_relaxation_error_2.tensor(thermal_relaxation_error_2)
    noise_model.add_all_qubit_quantum_error(depolarizing_error_2.compose(thermal_relaxation_error_2), 'cx')


def add_one_gate_error(noise_model):
    depolarizing_error_1 = depolarizing_error(ONE_QUBIT_GATE_ERROR, 1)
    # all times are in microseconds
    t1 = 10e10
    t2 = 2e8
    one_qubit_gate_time = 1e4
    thermal_relaxation_error_1 = thermal_relaxation_error(t1, t2, one_qubit_gate_time)
    noise_model.add_all_qubit_quantum_error(depolarizing_error_1.compose(thermal_relaxation_error_1),
                                            ['u1', 'u2', 'u3'])


def append_randomized_benchmarking_subcircuit(quantum_circuit):
    if RANDOMIZED_BENCHMARKING:
        quantum_circuit.barrier()
        with open(RANDOMIZED_BENCHMARKING_FILE, 'rb') as f:
            qc = pickle.load(f)
        quantum_circuit.compose(qc, inplace=True)
        quantum_circuit.barrier()


def report_results(counts):
    accuracy = counts['0'] / (counts['0'] + counts['1'])
    print("accuracy: " + str(accuracy))


def run_simulations_for_circuit(aer_sim, quantum_circuit, noise_model):
    quantum_circuit = transpile(quantum_circuit, aer_sim, optimization_level=0)
    draw_circuit(quantum_circuit)
    counts = execute(quantum_circuit, backend=aer_sim, noise_model=noise_model, shots=10000).result().get_counts()
    count_0 = 0
    count_1 = 0
    for key, val in counts.items():
        if key[0] == '0':
            count_0 += val
        if key[0] == '1':
            count_1 += val

    return {'0': count_0, '1': count_1}


def draw_circuit(quantum_circuit):
    if CODE in [Codes.STEANE, Codes.NO_CORRECTION]:
        quantum_circuit.draw('mpl', scale=2, style={'backgroundcolor': '#EEEEEE'})
    else:
        quantum_circuit.draw(output='mpl', fold=150)
    plt.savefig(f"circuit{NUMBER_OF_CODE_QUBITS}.png")
    print("circuit has been drawn")


def generate_circuit():
    if CODE == Codes.STEANE:
        return generate_circuit_steane()
    if CODE == Codes.SURFACE1:
        return generate_circuit_surface()
    if CODE == Codes.NO_CORRECTION:
        return generate_circuit_without_correction()


def generate_circuit_without_correction():
    quantum_circuit = QuantumCircuit(7)
    initialize_first_qubit(quantum_circuit)
    entange_qubits(quantum_circuit)
    append_randomized_benchmarking_subcircuit(quantum_circuit)
    original_qubit_final_outcome = ClassicalRegister(1, 'outcome')
    quantum_circuit.add_register(original_qubit_final_outcome)
    disentangle_qubits(quantum_circuit)
    quantum_circuit.measure([0], original_qubit_final_outcome)
    return quantum_circuit


def disentangle_qubits(quantum_circuit):
    decode_logical_qubit_steane(range(7), quantum_circuit)


def entange_qubits(quantum_circuit):
    encode_to_logical_qubit_steane(range(7), quantum_circuit)


def generate_circuit_steane():
    all_qubits = range(NUMBER_OF_CODE_QUBITS)
    quantum_circuit = QuantumCircuit(NUMBER_OF_CODE_QUBITS)
    ancillas_x, ancillas_z, classical_register_x, classical_register_z, original_qubit_final_outcome = define_ancillas_and_classical_registers(
        quantum_circuit)

    initialize_first_qubit(quantum_circuit)
    encode_to_logical_qubit_steane(all_qubits, quantum_circuit)

    append_randomized_benchmarking_subcircuit(quantum_circuit)

    append_stabilizers_logic_steane(quantum_circuit)
    measure_ancillas(ancillas_x, ancillas_z, classical_register_x, classical_register_z, quantum_circuit)
    correct_errors_steane(classical_register_x, classical_register_z, quantum_circuit)

    decode_logical_qubit_steane(all_qubits, quantum_circuit)

    quantum_circuit.measure(all_qubits[0], original_qubit_final_outcome)
    return quantum_circuit


def generate_circuit_surface():
    quantum_circuit = SurfaceCodeLogicalQubit(3)
    quantum_circuit.stabilize()
    append_randomized_benchmarking_subcircuit(quantum_circuit)
    # quantum_circuit.x(0)
    quantum_circuit.stabilize()
    quantum_circuit.readout_z()
    return quantum_circuit


def append_stabilizers_logic_steane(quantum_circuit):
    ancillas = range(7, 13)
    for i in ancillas:
        quantum_circuit.h(i)
    append_stabilizers_7(quantum_circuit)
    for i in ancillas:
        quantum_circuit.h(i)


def define_ancillas_and_classical_registers(quantum_circuit):
    ancillas_x = QuantumRegister(NUMBER_ANCILLAS_X, 'ancillas_x')
    ancillas_z = QuantumRegister(NUMBER_ANCILLAS_X, 'ancillas_z')
    classical_register_x = ClassicalRegister(NUMBER_ANCILLAS_X, 'synd_X')
    classical_register_z = ClassicalRegister(NUMBER_ANCILLAS_X, 'synd_Z')
    original_qubit_final_outcome = ClassicalRegister(1, 'outcome')
    quantum_circuit.add_register(ancillas_x)
    quantum_circuit.add_register(ancillas_z)
    quantum_circuit.add_register(classical_register_x)
    quantum_circuit.add_register(classical_register_z)
    quantum_circuit.add_register(original_qubit_final_outcome)
    return ancillas_x, ancillas_z, classical_register_x, classical_register_z, original_qubit_final_outcome


def encode_to_logical_qubit_steane(all_qubits, quantum_circuit):
    encoding = Encoding7()
    quantum_circuit.append(encoding, all_qubits)


def decode_logical_qubit_steane(all_qubits, quantum_circuit):
    encoding = Encoding7()
    quantum_circuit.append(encoding.inverse(), all_qubits)


def correct_errors_steane(classical_register_x, classical_register_z, quantum_circuit):
    for i in range(NUMBER_OF_CODE_QUBITS):
        quantum_circuit.z(i).c_if(classical_register_z, i + 1)
        quantum_circuit.x(i).c_if(classical_register_x, i + 1)


def measure_ancillas(ancillas_x, ancillas_z, classical_register_x, classical_register_z, quantum_circuit):
    for i in range(NUMBER_ANCILLAS_X):
        quantum_circuit.measure(ancillas_x[i], classical_register_x[i])
    for i in range(NUMBER_ANCILLAS_X):
        quantum_circuit.measure(ancillas_z[i], classical_register_z[i])


def append_stabilizers_7(quantum_circuit):
    stab1Z = QuantumCircuit(5, name='M1')  # IIIZZZZ
    for i in range(0, stab1Z.num_qubits - 1):
        stab1Z.cz(i, 4)
    stab2Z = QuantumCircuit(5, name='M2')  # IZZIIZZ
    for i in range(0, stab2Z.num_qubits - 1):
        stab2Z.cz(i, 4)
    stab3Z = QuantumCircuit(5, name='M3')  # ZIZIZIZ
    for i in range(0, stab3Z.num_qubits - 1):
        stab3Z.cz(i, 4)
    stab1X = QuantumCircuit(5, name='M4')  # IIIXXXX
    for i in range(0, stab1X.num_qubits - 1):
        stab1X.cx(i, 4)
    stab2X = QuantumCircuit(5, name='M5')  # IXXIIXX
    for i in range(0, stab2X.num_qubits - 1):
        stab2X.cx(i, 4)
    stab3X = QuantumCircuit(5, name='M6')  # XIXIXIX
    for i in range(0, stab3X.num_qubits - 1):
        stab3X.cx(i, 4)
    quantum_circuit.append(stab1Z, [3, 4, 5, 6, 9])
    quantum_circuit.append(stab2Z, [1, 2, 5, 6, 8])
    quantum_circuit.append(stab3Z, [0, 2, 4, 6, 7])
    quantum_circuit.append(stab3X, [0, 2, 4, 6, 10])
    quantum_circuit.append(stab2X, [1, 2, 5, 6, 11])
    quantum_circuit.append(stab1X, [3, 4, 5, 6, 12])


def initialize_first_qubit(quantum_circuit):
    initial_state = [1, 0]
    quantum_circuit.initialize(initial_state, 0)


def Encoding7():
    q_encoding = QuantumCircuit(7, name='Enc')
    q_encoding.h(6)
    q_encoding.h(5)
    q_encoding.h(4)

    q_encoding.cx(0, 1)
    q_encoding.cx(0, 2)
    q_encoding.cx(6, 3)
    q_encoding.cx(6, 1)
    q_encoding.cx(6, 0)
    q_encoding.cx(5, 3)
    q_encoding.cx(5, 2)
    q_encoding.cx(5, 0)
    q_encoding.cx(4, 3)
    q_encoding.cx(4, 2)
    q_encoding.cx(4, 1)

    return q_encoding


PRINT_MESSAGE = """ notes:
       _               _          _ _  __  __    __             _                                 _ _ _ 
      | |             | |        | (_)/ _|/ _|  / _|           (_)                               | | | |
   ___| |__   ___  ___| | __   __| |_| |_| |_  | |_ ___  _ __   _ _ __ ___   __ _  __ _  ___  ___| | | |
  / __| '_ \ / _ \/ __| |/ /  / _` | |  _|  _| |  _/ _ \| '__| | | '_ ` _ \ / _` |/ _` |/ _ \/ __| | | |
 | (__| | | |  __/ (__|   <  | (_| | | | | |   | || (_) | |    | | | | | | | (_| | (_| |  __/\__ \_|_|_|
  \___|_| |_|\___|\___|_|\_\  \__,_|_|_| |_|   |_| \___/|_|    |_|_| |_| |_|\__,_|\__, |\___||___(_|_|_)
                                                                                   __/ |                
                                                                                  |___/                 
- remember to use barrier between gates to avoid optimization combining them to one gate

"""

if __name__ == "__main__":
    print(PRINT_MESSAGE)
    print("num qubits: " + str(NUMBER_OF_CODE_QUBITS))
    print("RB: " + str(RANDOMIZED_BENCHMARKING))
    print("noise: " + str(NOISE))
    main()

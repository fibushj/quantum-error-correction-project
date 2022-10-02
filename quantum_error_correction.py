import time

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

"""

class Codes:
    NO_CORRECTION = 1
    STEANE = 7
    SURFACE1 = 9  # rotated surface code with 9 physical qubits representing one logical qubit. d=3
    # SURFACE2 =  # to be implemented, d=5


class Config:
    ONE_QUBIT_GATE_ERROR = 0.001
    TWO_QUBIT_GATE_ERROR = 0.003
    READOUT_ERROR = 0.005
    RANDOMIZED_BENCHMARKING = True
    NOISE = True

    @staticmethod
    def configure_code(code):
        Config.CODE = code
        Config.NUMBER_OF_CODE_QUBITS = code
        Config.NUMBER_ANCILLAS_X = int(Config.NUMBER_OF_CODE_QUBITS / 2)


def main():
    codes_to_run = [
        Codes.NO_CORRECTION,
        Codes.STEANE,
        Codes.SURFACE1
    ]
    for randomized_benchmarking_length in range(0, 75, 5):
        for code in codes_to_run:
            Config.configure_code(code)
            Config.RANDOMIZED_BENCHMARKING_LENGTH = randomized_benchmarking_length

            print_conf()
            start_time = time.time()

            aer_sim = Aer.get_backend('aer_simulator')
            quantum_circuit = generate_circuit()
            noise_model = create_noise_model()
            accuracy = run(quantum_circuit, noise_model, aer_sim)

            elapsed_time = time.time() - start_time
            print(elapsed_time)

            result = [Config.NUMBER_OF_CODE_QUBITS, Config.RANDOMIZED_BENCHMARKING_LENGTH, accuracy]
            with open("results.txt", 'a') as f:
                f.write(",".join(map(str, result)) + '\n')


def print_conf():
    print("num qubits: " + str(Config.NUMBER_OF_CODE_QUBITS))
    # print("RB: " + str(Config.RANDOMIZED_BENCHMARKING))
    print("RB length: " + str(Config.RANDOMIZED_BENCHMARKING_LENGTH))
    # print("noise: " + str(Config.NOISE))


def run(quantum_circuit, noise_model, aer_sim):
    if Config.CODE in [Codes.STEANE, Codes.NO_CORRECTION]:
        counts = run_simulations_for_circuit(aer_sim, quantum_circuit, noise_model)
        return calculate_accuracy(counts)
    return run_surface_code(aer_sim, noise_model, quantum_circuit)


def run_surface_code(aer_sim, noise_model, quantum_circuit):
    quantum_circuit = transpile(quantum_circuit, aer_sim, optimization_level=0)
    draw_circuit(quantum_circuit)
    counts = execute(quantum_circuit, backend=aer_sim, noise_model=noise_model).result().get_counts()
    # print(counts)
    code_distance = 3 if Config.CODE == Codes.SURFACE1 else 5
    benchmarking_tool = SurfaceCodeBenchmarkingTool(
        decoder=GraphDecoder(d=code_distance, T=1),
        readout_circuit=quantum_circuit
    )
    accuracy = 1 - benchmarking_tool.logical_error_rate(counts, correct_logical_value=0)
    return accuracy


def create_noise_model():
    noise_model = NoiseModel()
    if Config.NOISE:
        add_one_gate_error(noise_model)
        add_two_gate_error(noise_model)
        add_readout_error(noise_model)
        noise_model.add_basis_gates(['u1', 'u2', 'u3', 'cx'])
    return noise_model


def add_readout_error(noise_model):
    readout_error_prob = Config.READOUT_ERROR
    readout_error = ReadoutError(
        [[1 - readout_error_prob, readout_error_prob], [readout_error_prob, 1 - readout_error_prob]])
    noise_model.add_readout_error(readout_error, [0])  # add readout error to 1st qubit


def add_two_gate_error(noise_model):
    depolarizing_error_2 = depolarizing_error(Config.TWO_QUBIT_GATE_ERROR, 2)
    t1 = 10e10
    t2 = 2e8
    two_qubit_gate_time = 2.1e5
    thermal_relaxation_error_2 = thermal_relaxation_error(t1, t2, two_qubit_gate_time)
    thermal_relaxation_error_2 = thermal_relaxation_error_2.tensor(thermal_relaxation_error_2)
    noise_model.add_all_qubit_quantum_error(depolarizing_error_2.compose(thermal_relaxation_error_2), 'cx')


def add_one_gate_error(noise_model):
    depolarizing_error_1 = depolarizing_error(Config.ONE_QUBIT_GATE_ERROR, 1)
    # all times are in microseconds
    t1 = 10e10
    t2 = 2e8
    one_qubit_gate_time = 1e4
    thermal_relaxation_error_1 = thermal_relaxation_error(t1, t2, one_qubit_gate_time)
    noise_model.add_all_qubit_quantum_error(depolarizing_error_1.compose(thermal_relaxation_error_1),
                                            ['u1', 'u2', 'u3'])


def append_randomized_benchmarking_subcircuit(quantum_circuit):
    if Config.RANDOMIZED_BENCHMARKING:
        with open(RANDOMIZED_BENCHMARKING_FILE, 'rb') as f:
            qc = pickle.load(f)
        for _ in range(Config.RANDOMIZED_BENCHMARKING_LENGTH):
            quantum_circuit.barrier()
            quantum_circuit.compose(qc, inplace=True)
        quantum_circuit.barrier()


def calculate_accuracy(counts):
    accuracy = counts['0'] / (counts['0'] + counts['1'])
    return accuracy


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
    # if Config.CODE in [Codes.STEANE, Codes.NO_CORRECTION]:
    #     quantum_circuit.draw('mpl', scale=2, style={'backgroundcolor': '#EEEEEE'})
    # else:
    #     quantum_circuit.draw(output='mpl', fold=150)
    # plt.savefig(f"circuit{Config.NUMBER_OF_CODE_QUBITS}.png")
    print("circuit has been drawn")


def generate_circuit():
    if Config.CODE == Codes.STEANE:
        return generate_circuit_steane()
    if Config.CODE == Codes.SURFACE1:
        return generate_circuit_surface()
    if Config.CODE == Codes.NO_CORRECTION:
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
    all_qubits = range(Config.NUMBER_OF_CODE_QUBITS)
    quantum_circuit = QuantumCircuit(Config.NUMBER_OF_CODE_QUBITS)
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
    ancillas_x = QuantumRegister(Config.NUMBER_ANCILLAS_X, 'ancillas_x')
    ancillas_z = QuantumRegister(Config.NUMBER_ANCILLAS_X, 'ancillas_z')
    classical_register_x = ClassicalRegister(Config.NUMBER_ANCILLAS_X, 'synd_X')
    classical_register_z = ClassicalRegister(Config.NUMBER_ANCILLAS_X, 'synd_Z')
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
    for i in range(Config.NUMBER_OF_CODE_QUBITS):
        quantum_circuit.z(i).c_if(classical_register_z, i + 1)
        quantum_circuit.x(i).c_if(classical_register_x, i + 1)


def measure_ancillas(ancillas_x, ancillas_z, classical_register_x, classical_register_z, quantum_circuit):
    for i in range(Config.NUMBER_ANCILLAS_X):
        quantum_circuit.measure(ancillas_x[i], classical_register_x[i])
    for i in range(Config.NUMBER_ANCILLAS_X):
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


if __name__ == "__main__":
    main()

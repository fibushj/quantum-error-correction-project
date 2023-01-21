from collections import defaultdict

from matplotlib import pyplot as plt

from quantum_error_correction import Codes
NOISE_MODEL_NAME = "honeywell"
# NOISE_MODEL_NAME = "ionq"
with open(f"results_{NOISE_MODEL_NAME}.txt") as f:
    lines = f.read().splitlines()

results = defaultdict(list)
for line in lines:
    try:
        code, length, accuracy = list(map(float, line.split(',')))
        results[code].append([length, accuracy])
    except:
        pass

codes_names = {Codes.NO_CORRECTION: "No correction applied",
               Codes.STEANE: "Steane Code (7 qubit code)",
               Codes.SURFACE1: "Rotated Surface code (9 qubit code)"
               }

for code in [Codes.NO_CORRECTION, Codes.STEANE, Codes.SURFACE1]:
    print(results[code])
    x_coordinates = [result[0] for result in results[code]]
    y_coordinates = [result[1] for result in results[code]]
    plt.plot(x_coordinates, y_coordinates, label=codes_names[code])
    plt.legend()

plt.xlabel("Randomized Benchmarking sequence length")
plt.ylabel("Accuracy")
# plt.show()
plt.savefig(f'results_{NOISE_MODEL_NAME}.png')

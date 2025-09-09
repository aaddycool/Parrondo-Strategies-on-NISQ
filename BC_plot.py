import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator, noise
from qiskit.circuit.library import UnitaryGate, QFT

shots = 100000
p = 0.998489
q = 0.119545

fid_1 = []


A = UnitaryGate(np.array([[np.sqrt(p), np.sqrt(1-p)],[np.sqrt(1-p),-np.sqrt(p)]]),label="A")
B = UnitaryGate(np.array([[np.sqrt(q), np.sqrt(1-q)],[np.sqrt(1-q),-np.sqrt(q)]]),label="B")

def BC(P,Q):
    positions = sorted(set(P.keys()).union(Q.keys()))
    BC = 0 
    for p in positions:
        p_val = P.get(p,0)
        q_val = Q.get(p,0)
        BC += np.sqrt(p_val*q_val)
    return BC**2
max_t = 12
sim = AerSimulator()

prob_noise = 0.02
error = noise.depolarizing_error(prob_noise, 2)
noise_model = noise.NoiseModel()
noise_model.add_all_qubit_quantum_error(error, ['cp'])
basis_gates = noise_model.basis_gates

for t in range(max_t+1):
    qc = QuantumCircuit(3,2)
    qc.append(QFT(2, do_swaps=False).to_gate(label='QFT'), [0, 1])
    for i in range(t):
        if i % 4 == 0 or i % 4 == 1:
            qc.append(A,[2])
        else:
            qc.append(B,[2])
        qc.p(-np.pi,0)
        qc.p(-np.pi/2,1)
        qc.cp(np.pi,2,1)
    qc.append(QFT(2, do_swaps=False).inverse().to_gate(label='QFT'), [0, 1])
    qc.measure([0,1],[0,1])
    qc_t = transpile(qc,backend=sim)
    result = sim.run(qc_t,shots=shots).result()
    counts = result.get_counts()
    probs = {state: count / shots for state, count in counts.items()}
    
    result_noisy = sim.run(qc_t, shots=shots, noise_model=noise_model, basis_gates=basis_gates).result()
    counts_noisy = result_noisy.get_counts()
    probs_noisy = {state: count / shots for state, count in counts_noisy.items()}

    fid = BC(probs, probs_noisy)
    fid_1.append(fid)
plt.plot(range(max_t+1), fid_1)
plt.show()
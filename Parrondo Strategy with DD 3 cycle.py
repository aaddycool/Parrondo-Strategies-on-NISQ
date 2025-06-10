#To plot Probability at initial position and Hellinger fidelity vs time steps for 3 cycle DTQW via Parrondo strategy with DD
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import UnitaryGate
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2


service = QiskitRuntimeService()
backend_name = "ibm_sherbrooke"
sim_ibm = service.backend(backend_name)

sim = AerSimulator()

shots = 100000

def hellinger_fidelity(P, Q):
    positions = sorted(set(P.keys()).union(Q.keys()))  # Ensure all positions are considered
    H = 0
    for p in positions:
        p_val = P.get(p, 0)
        q_val = Q.get(p, 0)
        H += (np.sqrt(p_val) - np.sqrt(q_val)) ** 2
    hf = np.sqrt(H / 2)
    HF = hf ** 2
    return (1 - HF) ** 2

fidelities2 = [] #to store fidelity values at each step

#Chaotic Coins
p = 0.264734
q = 0.801571
coin_A = np.array([[np.sqrt(p), np.sqrt(1-p)], [np.sqrt(1-p), -np.sqrt(p)]])
coin_B = np.array([[np.sqrt(q), np.sqrt(1-q)], [np.sqrt(1-q), -np.sqrt(q)]])

A = UnitaryGate(coin_A, label='A')
B = UnitaryGate(coin_B, label='B')

# QFT and IQFT matrices
def qft_3cycle_matrix():
    omega = np.exp(2j * np.pi / 3)
    qft_matrix = np.zeros((4, 4), dtype=complex)
    qft_matrix[:3, :3] = np.array([
        [1, 1, 1],
        [1, omega, omega**2],
        [1, omega**2, omega]
    ]) / np.sqrt(3)
    qft_matrix[3, 3] = 1
    return qft_matrix

def iqft_3cycle_matrix():
    omega = np.exp(-2j * np.pi / 3)
    iqft_matrix = np.zeros((4, 4), dtype=complex)
    iqft_matrix[:3, :3] = np.array([
        [1, 1, 1],
        [1, omega, omega**2],
        [1, omega**2, omega]
    ]) / np.sqrt(3)
    iqft_matrix[3, 3] = 1
    return iqft_matrix

# Create QFT and IQFT gates
qft_gate = UnitaryGate(qft_3cycle_matrix(), label="QFT_3cycle")
iqft_gate = UnitaryGate(iqft_3cycle_matrix(), label="IQFT_3cycle")

probs_ibm = [] #to store probabilities obtained in real hardware
probs_ideal = [] #to store probabilities obtained via ideal simulations

# Loop over time steps
max_t = 20
for t in range(max_t+1):
    #Step 1:  Create a new quantum circuit for each time step
    qc = QuantumCircuit(3, 2)
    #Step 2: QFT to position qubits at beginning
    qc.append(qft_gate, [0, 1])
    #Step 3 : Coin and Shift via phase gates at every time step
    for i in range(t):
       if i % 4 == 0 or i % 4 == 1:
           qc.append(A, [2])
       else:
           qc.append(B, [2])
       qc.p(-4 * np.pi / 3, 0)
       qc.p(-2 * np.pi / 3, 1)
       qc.cp(8 * np.pi / 3, 2, 0)
       qc.cp(4 * np.pi / 3, 2, 1) 
    # Step 4: IQFT to position qubits at the end 
    qc.append(iqft_gate, [0, 1])
    #Step 5: Measure the position qubits
    qc.measure([0, 1], [0, 1])
   
    # Simulate the circuit
    qc_t = transpile(qc,backend=sim) #to run in AerSimulator()
    qc_ibm = transpile(qc,backend = sim_ibm, optimization_level=3) #to run in real hardware
    
    result = sim.run(qc_t, shots=shots).result()
    counts = result.get_counts()
   
    sampler = SamplerV2(mode = sim_ibm) 
    sampler.options.dynamical_decoupling.enable = True #False to disable
    sampler.options.dynamical_decoupling.sequence_type = "XY4" #DD pulses
    job_ibm = sampler.run([qc_ibm], shots=shots)
    result_ibm = job_ibm.result()
    counts_ibm = result_ibm[0].data.c.get_counts()
    
    prob_counts= {}
    for state, count in counts.items():
        position = int(state, 2) % 3  # Map binary states to 3-cycle positions
        if position in prob_counts:
            prob_counts[position] += count
        else:
            prob_counts[position] = count
    
    prob_initial= prob_counts.get(0, 0) / shots #concerning only to initial position 0
    probs_ideal.append(prob_initial)
    
    prob_counts_ibm = {}
    for state, count in counts_ibm.items():
        position = int(state, 2) % 3  # Map binary states to 3-cycle positions
        if position in prob_counts_ibm:
            prob_counts_ibm[position] += count
        else:
            prob_counts_ibm[position] = count

    prob_initial_ibm = prob_counts_ibm.get(0, 0) / shots
    probs_ibm.append(prob_initial_ibm)
   
    fidelity = hellinger_fidelity({'00': prob_initial}, {'00': prob_initial_ibm})
 
    fidelities2.append(fidelity)
#Plot the results
plt.figure(figsize=(8, 6))
plt.plot(range(max_t+1), probs_ibm, marker='o', color='brown', label="ibm_sherbrooke opt - 3 with DD")
plt.plot(range(max_t+1), probs_ideal, marker = 'o',color = 'blue', label = 'AerSimulator')
plt.xlabel("Time Steps", fontsize=18,fontweight = 'bold')
plt.ylabel("Probability", fontsize=18, fontweight='bold')
plt.xticks(fontweight='bold',fontsize=15)
plt.yticks(fontweight='bold',fontsize=15)
plt.legend(prop = {'weight':'bold','size': '14'})
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(range(max_t + 1), fidelities2, marker='o', color='brown', label="ibm_sherbrooke opt - 3 with DD")
plt.xlabel("Time Steps", fontsize=18,fontweight = 'bold')
plt.ylabel("Hellinger Fidelity", fontsize=18,fontweight = 'bold')
plt.xticks(fontweight='bold',fontsize=15)
plt.yticks(fontweight='bold',fontsize=15)
plt.legend(prop = {'weight':'bold','size': '14'})
plt.show()



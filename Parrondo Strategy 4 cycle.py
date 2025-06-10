#To plot Probability at initial position and Hellinger fidelity vs time steps for 4 cycle DTQW via Parrondo strategy
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import UnitaryGate, QFT
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
p = 0.998489
q = 0.119545
coin_A = np.array([[np.sqrt(p), np.sqrt(1-p)], [np.sqrt(1-p), -np.sqrt(p)]])
coin_B = np.array([[np.sqrt(q), np.sqrt(1-q)], [np.sqrt(1-q), -np.sqrt(q)]])

A = UnitaryGate(coin_A, label='A')
B = UnitaryGate(coin_B, label='B')


probs_ibm = [] #to store probabilities obtained in real hardware
probs_ideal = [] #to store probabilities obtained via ideal simulations

# Loop over time steps
max_t = 20
for t in range(max_t+1):
    #Step 1:  Create a new quantum circuit for each time step
    qc = QuantumCircuit(3, 2)
    #Step 2: QFT to position qubits at beginning
    qc.append(QFT(2, do_swaps=False).to_gate(label='QFT'), [0, 1])
    #Step 3 : Coin and Shift via phase gates at every time step
    for i in range(t):
       if i % 4 == 0 or i % 4 == 1:
           qc.append(A, [2])
       else:
           qc.append(B, [2])
       qc.p(-np.pi,0)
       qc.p(-np.pi/2,1)
       qc.cp(np.pi,2,1)
    # Step 4: IQFT to position qubits at the end 
    qc.append(QFT(2, do_swaps=False).inverse().to_gate(label='QFT'), [0, 1])
    #Step 5: Measure the position qubits
    qc.measure([0, 1], [0, 1])
   
    # Simulate the circuit
    qc_t = transpile(qc,backend=sim) #to run in AerSimulator()
    qc_ibm = transpile(qc,backend = sim_ibm, optimization_level=3) #to run in real hardware
    
    result = sim.run(qc_t, shots=shots).result()
    counts = result.get_counts()
   
    sampler = SamplerV2(mode = sim_ibm) 
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
plt.plot(range(max_t+1), probs_ibm, marker='o', color='brown', label="ibm_sherbrooke opt - 3")
plt.plot(range(max_t+1), probs_ideal, marker = 'o',color = 'blue', label = 'AerSimulator')
plt.xlabel("Time Steps", fontsize=18,fontweight = 'bold')
plt.ylabel("Probability", fontsize=18, fontweight='bold')
plt.xticks(fontweight='bold',fontsize=15)
plt.yticks(fontweight='bold',fontsize=15)
plt.legend(prop = {'weight':'bold','size': '14'})
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(range(max_t + 1), fidelities2, marker='o', color='brown', label="ibm_sherbrooke opt - 3")
plt.xlabel("Time Steps", fontsize=18,fontweight = 'bold')
plt.ylabel("Hellinger Fidelity", fontsize=18,fontweight = 'bold')
plt.xticks(fontweight='bold',fontsize=15)
plt.yticks(fontweight='bold',fontsize=15)
plt.legend(prop = {'weight':'bold','size': '14'})
plt.show()




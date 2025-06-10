#To plot the Depth Vs Time steps of 3 and 4 cyclic DTQW via Parrondo strategy
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import UnitaryGate,QFT
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
backend_name = 'ibm_brisbane'
sim_ibm = service.backend(backend_name)

shots = 100000

#Coin Parameters for 3 cycle
p = 0.264734
q = 0.801571

A = UnitaryGate(np.array([[np.sqrt(p), np.sqrt(1-p)], [np.sqrt(1-p), -np.sqrt(p)]]), label='A')
B = UnitaryGate(np.array([[np.sqrt(q), np.sqrt(1-q)], [np.sqrt(1-q), -np.sqrt(q)]]), label='B')

#Coin parameters for 4 cycle
w = 0.998489
g = 0.119545

C = UnitaryGate(np.array([[np.sqrt(w), np.sqrt(1-w)], [np.sqrt(1-w), -np.sqrt(w)]]), label='A')
D = UnitaryGate(np.array([[np.sqrt(g), np.sqrt(1-g)], [np.sqrt(1-g), -np.sqrt(g)]]), label='B')

# QFT and IQFT matrices 3cycle
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

# Create QFT and IQFT gates for 3cycle
qft_gate = UnitaryGate(qft_3cycle_matrix(), label="QFT_3cycle")
iqft_gate = UnitaryGate(iqft_3cycle_matrix(), label="IQFT_3cycle")


depths = [] #to store depths of 3cycle with opt_level 3
depth_opt1_3 =[] #to store depths of 3cycle with opt_level 1
depth_opt1_4 = [] #to store depths of 4cycle with opt_level 1
depths_1 = [] #to store depths of 4cycle with opt_level 3

# Loop over time steps for 3 cycle
max_t = 20
for t in range(max_t+1):
    # Create a new quantum circuit for each time step
    qc = QuantumCircuit(3, 2)
    qc.append(qft_gate, [0, 1])
    for i in range(t):
       if i % 4 == 0 or i % 4 == 1:
           qc.append(A, [2])
       else:
           qc.append(B, [2])
       qc.p(-4 * np.pi / 3, 0)
       qc.p(-2 * np.pi / 3, 1)
       qc.cp(8 * np.pi / 3, 2, 0)
       qc.cp(4 * np.pi / 3, 2, 1)
      
 
    qc.append(iqft_gate, [0, 1])
    qc.measure([0, 1], [0, 1])
   

    qc_ibm = transpile(qc,backend = sim_ibm, optimization_level=3)
    qc_ibm1 = transpile(qc,backend = sim_ibm, optimization_level=1)
    depths.append(qc_ibm.depth())
    depth_opt1_3.append(qc_ibm1.depth())
    
 #4 cycle   
for x in range(max_t+1):
    qc1 = QuantumCircuit(3,2)
    qc1.append(QFT(2, do_swaps=False).to_gate(label='QFT'), [0, 1])
    for j in range(x):
        if j % 4 == 0 or j % 4 == 1:
            qc1.append(C,[2])
        else:
            qc1.append(D,[2])
        qc1.p(-np.pi,0)
        qc1.p(-np.pi/2,1)
        qc1.cp(np.pi,2,1)
    qc1.append(QFT(2, do_swaps=False).inverse().to_gate(label='IQFT'), [0, 1])
    qc1.measure([0,1],[0,1])
    
    qc1_ibm = transpile(qc1,backend = sim_ibm,optimization_level = 3)
    qc1_ibm1 = transpile(qc1,backend = sim_ibm,optimization_level = 1)
    depths_1.append(qc1_ibm.depth())
    depth_opt1_4.append(qc1_ibm1.depth())

#Plot the results
plt.plot(range(max_t+1),depths,marker='o',color = 'blue', label = "3 cycle Opt 3")
plt.plot(range(max_t+1),depth_opt1_3,marker = 's',color = 'red', label = "3 cycle Opt 1")
plt.plot(range(max_t+1),depths_1,marker='^',color = 'green', label = "4 cycle Opt 3")
plt.plot(range(max_t+1),depth_opt1_4,marker = '<',color = 'orange', label = "4 cycle Opt 1")
plt.show()
# Lourdes Simon Codina
# In this code I simulate the BB84 protocol with Qiskit and eavesdropping using an optimized 
# Breidbart-like basis. Phaseshift errors with probability g are applied. If g = 0 and beta = 0 
# the Breidbart basis is recovered. Alice and Bob always use the same basis, so no bits
# are discarded. The simulation is repeated s times to study the probability of detecting
# Eve (d/s), the probability that Eve guesses correctly, and the error induced by her presence.

import qiskit as qk
import numpy as np
import random
from tqdm import tqdm
from qiskit_aer import AerSimulator
from qiskit_aer.noise import pauli_error
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

n = 100
s = 500
g = 0.3

# Theoretical probability of Eve measuring the right result
def f(angle):
    return 0.5*((np.cos(angle))**2 - g*(np.cos(2*angle)) + 0.5 + (g - 0.5)*(np.sin(2*angle)))

# Theoretical probability of Bob measuring the wrong result
def h(angle): 
    return  (
    0.5*((2*g*(1-g)*((np.sin(angle))**4 + (np.cos(angle))**4) + 2*((np.sin(angle))**2)*((np.cos(angle))**2)*((1-g)**2 + g**2)) 
         + (2*(1-g)*g*((0.5-np.sin(angle)*np.cos(angle))**2 + (0.5+np.sin(angle)*np.cos(angle))**2 ) + 2* ((1-g)**2 + g**2)*(0.25 - ((np.sin(angle))**2)*((np.cos(angle))**2))))
)

# Objective function to minimize
def objective(angle, alpha, beta):
    return -alpha * f(angle) + beta * h(angle)

# Weights
alpha = 0.0  # Importance of maximizing Eve's probability of measuring correctly
beta = 1.0   # Importance of minimizing Bob's error

# Minimize the objective function in the range [0, pi*2]
res = minimize_scalar(objective, args=(alpha, beta), bounds=(0, np.pi*2.0), method='bounded')
angle = res.x

# angle = 0.5 * np.arctan(2.0*g - 1.0) # optimal eavesdropping basis angle maximizing f(angle) 

# print(f"Optimal angle: {optimal_angle:.8f} rad")
# print(f"f(opt) = {f(optimal_angle):.8f}, h(opt) = {g(optimal_angle):.8f}")
# print(f"Angle: {angle:.8f} rad")
# print(f"f(opt) = {f(angle):.8f}, h(opt) = {h(angle):.8f}")



simulator = AerSimulator()
bitflip_error = pauli_error([('Y', g), ('I', 1 - g)])
error_instruction = bitflip_error.to_instruction()

P_tot_eve_correct = []
P_tot_error_caused = []
d = 0

for j in tqdm(range(s)):
    bits_Alice = np.random.randint(2, size=n)
    basis_Alice = np.random.randint(2, size=n)
    basis_Bob = basis_Alice.copy()

    total_raw_key = []
    total_eve_key = []
    circuits = []

    eve_circuits = []
    for i in range(n):
        qc = qk.QuantumCircuit(1, 1)
        if bits_Alice[i] == 1:
            qc.x(0)
        if basis_Alice[i] == 1:
            qc.h(0)
        qc.append(error_instruction, [0])
        qc.ry(2 * angle, 0)
        qc.measure(0, 0)
        eve_circuits.append(qc)

    #Run all Eve circuits in one batch
    eve_circuits = qk.transpile(eve_circuits, simulator)
    eve_result = simulator.run(eve_circuits, shots=1).result()
    total_eve_key = []

    # Save Eve measurements and prepare circuits Bob will recieve
    bob_circuits = []
    for i, qc in enumerate(eve_circuits):
        measured = int(list(eve_result.get_counts(qc).keys())[0])
        total_eve_key.append(measured)

        qc_bob = qk.QuantumCircuit(1, 1)
        if measured == 1:
            qc_bob.x(0)
        qc_bob.ry(-2 * angle, 0)
        qc_bob.append(error_instruction, [0])
        if basis_Bob[i] == 1:
            qc_bob.h(0)
        qc_bob.measure(0, 0)
        bob_circuits.append(qc_bob)

    # Run all Bob circuits in one batch
    bob_circuits = qk.transpile(bob_circuits, simulator)
    bob_result = simulator.run(bob_circuits, shots=1).result()
    total_raw_key = []

    for c in bob_circuits:
        measured = int(list(bob_result.get_counts(c).keys())[0])
        total_raw_key.append(measured)

    # bits_Alice_arr = bits_Alice[:len(total_eve_key)]
    total_raw_key = np.array(total_raw_key)
    total_eve_key = np.array(total_eve_key)

    P_eve_correct = np.count_nonzero(bits_Alice == total_eve_key) / n
    P_tot_eve_correct.append(P_eve_correct)

    alice_key = bits_Alice.copy()
    bob_key = total_raw_key

    # compare a subset of the keys to check for eavesdropping
    size = max(1, round(len(alice_key) / 3))
    subset = random.sample(range(len(alice_key)), size)

    # obtain the subsets
    alice_subset = alice_key[subset]
    bob_subset = bob_key[subset]
    P_error_caused = np.count_nonzero(alice_subset != bob_subset) / size
    P_tot_error_caused.append(P_error_caused)

    # if there is no eavesdropping the error in bob's key is p/2 since bitflip error does not affect the |+⟩/|-⟩ basis
    if P_error_caused > g/2:
        d += 1

# calculate the mean and standard deviation of the probabilities and write it in a
sigma_error_eve = np.std(P_tot_eve_correct, ddof=1) / np.sqrt(len(P_tot_eve_correct))
sigma_error_caused = np.std(P_tot_error_caused, ddof=1) / np.sqrt(len(P_tot_error_caused))


Pc = 0.5*((np.cos(angle))**2 - g*np.cos(2*angle) + 0.5 + (g - 0.5)*(np.sin(2*angle))) # theoretical probability of Eve measuring the right result
P_err_bob= (
    0.5*((2*g*(1-g)*((np.sin(angle))**4 + (np.cos(angle))**4) + 2*((np.sin(angle))**2)*((np.cos(angle))**2)*((1-g)**2 + g**2)) 
         + (2*(1-g)*g*((0.5-np.sin(angle)*np.cos(angle))**2 + (0.5+np.sin(angle)*np.cos(angle))**2 ) + 2* ((1-g)**2 + g**2)*(0.25 - ((np.sin(angle))**2)*((np.cos(angle))**2))))
)

with open("Prob_eve_correct__Breidbart_error g_1qbit.txt", "a", encoding="utf-8") as f:
    f.write(f"n = {n}\tg = {g}\talpha = {alpha}\tbeta = {beta}\tangle = {angle}\n")
    f.write(f"Prob. Eve correct: ({np.mean(P_tot_eve_correct):.8f}\u00B1{sigma_error_eve:.8f})\n")
    f.write(f"Error caused by Eve and g: ({np.mean(P_tot_error_caused):.8f}\u00B1{sigma_error_caused:.8f})\n")
    f.write(f"Theoretical prob. of Eve measuring the right results:\t{Pc:.8f}\n")
    f.write(f"Theoretical prob. of Bob's measuring the wrong results:\t{P_err_bob:.8f}\n")

with open("Detected eavesdropping_Breidbart_error g_1qbit.txt", "a") as f:
    f.write(f"Initial length: {n*2}\tg = {g}\tbits compared: {size}\t Prob detecting eve: {d / s:.8f}\n")


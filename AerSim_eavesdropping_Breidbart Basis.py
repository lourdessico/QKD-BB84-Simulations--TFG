# Lourdes Simon Codina
# In this code I simulate the BB84 protocol using qiskit with eavesdropping, were Eve uses
# the Breidbart basis attack. I skip the step of discarding the bits in which the bases do 
# not match since we dont need those bits after all. So Alice and Bob will have the same bases
# and n= the number of bits after discarding. I simulate the protocol s times with n qbits 
# each time. Then I study the probability of detecting Eve (d/s), with d the amount of times 
# Eve is detected.


import qiskit as qk
import numpy as np
import random
from tqdm import tqdm
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

n = 45 # number of qbits after discarding
s = 300 # number of simulations carried out
d = 0 # number of times Eve is detected

bits_Alice = np.random.randint(2, size=n*s)
basis_Alice = np.random.randint(2, size=n*s)
basis_Bob = basis_Alice.copy()
circuits_eve = []
circuits_bob = []

simulator = AerSimulator()  # Create an instance of Qiskit Aer's quantum simulator called AerSimulator.

for i in tqdm(range(n*s)):
        qc = qk.QuantumCircuit(1, 1)
        if bits_Alice[i] == 1: # |0⟩ --> |1⟩
            qc.x(0)
        if basis_Alice[i] == 1: # |0⟩ --> |+⟩ & |1⟩ --> |-⟩
            qc.h(0)

        # Breidbart basis: |b0⟩ = cos(-pi/8)|0⟩ - sin(-pi/8)|1⟩ ; |b1⟩ = sin(-pi/8)|0⟩ + cos(-pi/8)|1⟩
        # To measure in Breidbart basis: apply RY(2*(-π/8)) (rotation around Y axis) before measurement.
        qc.ry(-np.pi / 4, 0)
        qc.measure(0, 0) # Now Eve measures Alice's qbits
        circuits_eve.append(qc)

#Run all Eve circuits in one batch
circuits_eve = qk.transpile(circuits_eve, simulator)
eve_result = simulator.run(circuits_eve, shots=1).result()
total_eve_key = []

# Save Eve measurements and prepare circuits Bob will recieve
bob_circuits = []
for i, qc in enumerate(circuits_eve):
    measured = int(list(eve_result.get_counts(qc).keys())[0])
    total_eve_key.append(measured)

    qc = qk.QuantumCircuit(1, 1)
    if measured == 1: # |0⟩ --> |1⟩
        qc.x(0) 

    qc.ry(np.pi / 4, 0) # Encode eve's qbits in the Breidbart basis 

    if basis_Bob[i] == 1: # |0⟩ --> |+⟩ & |1⟩ --> |-⟩
        qc.h(0)
    qc.measure(0, 0) # Now Bob measures Eve's qbits
    bob_circuits.append(qc)

# Run all Bob circuits in one batch and save the measurements in total_raw_key
bob_circuits = qk.transpile(bob_circuits, simulator)
bob_result = simulator.run(bob_circuits, shots=1).result()
total_raw_key = []

for c in tqdm(bob_circuits):
    measured = int(list(bob_result.get_counts(c).keys())[0])
    total_raw_key.append(measured)

total_eve_key = np.array(total_eve_key)
total_raw_key = np.array(total_raw_key)

P_tot_eve_correct = []
P_tot_error_caused = []

# divide the key in s blocks of n bits (s simulations of n bits each)
for i in tqdm(range(s)):
    start = i*n
    end = (i + 1)*n

    Alice_block = bits_Alice[start:end]
    Bob_block = total_raw_key[start:end]
    Eve_block = total_eve_key[start:end]

    size = max(1,round(n/3)) # Compare a third of the bits in the destilled keys obtained.

    subset = np.random.choice(n, size=size, replace=False) # Select the bits to compare at random 
    alice_subset = Alice_block[subset] # generate the subsets I am going to compare
    bob_subset = Bob_block[subset]

    # # Obtain the keys without the bits that were compared
    # Alice_key = [alice_key[i] for i in range(len(alice_key)) if i not in subset]
    # Bob_key   = [bob_key[i] for i in range(len(bob_key)) if i not in subset]
    
    # calculate the error caused by evaesdropping
    P_error_caused= (np.count_nonzero(alice_subset != bob_subset))/size
    P_tot_error_caused.append(P_error_caused)
    if P_error_caused > 0: # If any of the bits compared do not match, Eve is detected (no noise in the quantum channel)
        d += 1

    P_eve_correct = np.count_nonzero(Alice_block == Eve_block)/n
    P_tot_eve_correct.append(P_eve_correct)

P_tot_error_caused = np.array(P_tot_error_caused)
P_tot_eve_correct = np.array(P_tot_eve_correct)

# Calculate the standard deviation of the probabilities
sigma_error_eve = np.std(P_tot_eve_correct, ddof=1) / np.sqrt(len(P_tot_eve_correct))
sigma_error_caused = np.std(P_tot_error_caused, ddof=1) / np.sqrt(len(P_tot_error_caused))
sigma_detect = np.sqrt((d/s)*(1.0 - (d/s))/s) # standard deviation of the probability of detecting Eve (binomial distribution)

# Write the results it in a text file
with open("Backend_Prob_eve_correct_Breidbart.txt", "a", encoding="utf-8") as f:
    f.write(f"n = {n}\t Prob. Eve correct: ({np.mean(P_tot_eve_correct):.6f}\u00B1{sigma_error_eve:.6f})\n")
    f.write(f"Error caused by Eve: ({np.mean(P_tot_error_caused):.6f}\u00B1{sigma_error_caused:.6f})\n")
    f.write(f"Theoretical prob. of Eve measuring the right results:\t{0.50:.6f}\n")
    f.write(f"Theoretical prob. of Bob's measuring the wrong results:\t{0.25:.6f}\n")

with open("Backend_Detected eavesdropping_Breidbart.txt", "a") as f:
    # f.write("n\tsize\td/s\terror\n")
    f.write(f"{n}\t{size}\t{d/s:.6f}\t{sigma_detect:.6f}\n")


#________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________

# generate a graphic: probability of detecting Eve (d/s) vs. number of initial qbits n
data = np.loadtxt("Backend_Detected eavesdropping_Breidbart.txt", skiprows=1)

# Extract 1st and 3rd column
ñ_values = data[:, 1]
probabilities = data[:, 2]
errors = data[:, 3]

x = np.linspace(0, 25, 100)  # 100 points between 0 and 50
f_x = 1 - (3/4)**x # probability of detection vs. ñ 


# Plot
plt.figure(figsize=(8, 5))
plt.errorbar(ñ_values, probabilities, yerr=errors, fmt='go', capsize=4, label='Simulated')
plt.plot(x, f_x, color='blue', label=r"$f(\widetilde{n}) = 1 - \left(\frac{3}{4}\right)^{\widetilde{n}}$")

plt.gca().xaxis.set_minor_locator(MultipleLocator(1))  # exterior ticks
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.05))  # exterior ticks
plt.tick_params(which='minor', direction='out', length=2, width=0.8, labelsize=20)
plt.tick_params(which='major', direction='out', length=5, width=1, labelsize=20)

plt.xlabel("$\widetilde{n}$", fontsize=20)
plt.ylabel("$P_d$", fontsize=20)
plt.xlim(0, 25)
plt.ylim(bottom=0)
plt.tick_params(direction='out')
plt.legend(loc="lower right", fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig("Backend_Prob_deteccion_eve_vs_ñ_Breidbart.png")
plt.show()

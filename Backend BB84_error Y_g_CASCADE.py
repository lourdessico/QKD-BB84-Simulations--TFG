# Lourdes Simon Codina
# In this code I simulate the BB84 protocol using qiskit with a nosy channel.
# I simulate the protocol 1 time with n qbits. There is a probability g of bitflip+phaseflip error.
# Then the CASCADE error correcting protocol is applyed.


import qiskit as qk
import numpy as np
import random
import math
from tqdm import tqdm
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
from qiskit_aer.noise import pauli_error


service = QiskitRuntimeService()

n = 500
g = 0.2 # probability of bitflip error

# Error correction with CASCADE protocol

# In this function the Binary protocol is applied to a block (array) with an odd error parity. Since the function
# returns its own function a condition if is used to stop the protocol when the block size is 1. The parameter start
# allows to keep track of the position of the error bit in the original block.
def binary(alice_block, bob_block, leaked_info, start=0):
    if bob_block.size == 1:  # if the block size is 1, return the position of the error bit
        return start, leaked_info
    
    half = bob_block.size//2

    # calculate the parity of the first hald of the blocks
    parity_bob = sum(bob_block[:half])%2
    parity_alice = sum(alice_block[:half])%2
    leaked_info += 1  # alice sends a public bit to inform bob about the parity

    # if the parities are different, the error is in the first half of the block, the start position stays the same
    if parity_bob != parity_alice:
        return binary(alice_block[:half], bob_block[:half], leaked_info, start)
    else:
        # if the parities are the same, the error is in the second half of the block and the start position is updated
        return binary(alice_block[half:], bob_block[half:], leaked_info, start + half)


# In this function the CASCADE protocol is applied to the keys obtained by Alice and Bob (Keys after discarding 
# the bits where their basis were not the same and the subsets they compared). The function returns the corrected keys and the 
# amount of information leaked during the protocol. In order to return the keys in the original order, the permutations 
# are stored in a list and then the inverse total permutation is applied.
def CASCADE(Alice_key, Bob_key, QBER, iterations=4):
    current_alice_key = Alice_key.copy()
    current_bob_key = Bob_key.copy()
    
    leaked_info = 0
    # initialy the permutation is the identity, so the keys are in the original order
    perm = np.arange(len(Alice_key))

    # blcok size for the first iteration (top-level block size)
    block_size = max(1, int(0.73/QBER))
    
    # we do 4 iterations
    for i in range(iterations):
        
        # the first iteration does not have shuffling, the rest do
        if i > 0:
            indices = np.arange(len(current_bob_key))
            np.random.shuffle(indices)
            # Apply the permutations
            current_bob_key = current_bob_key[indices]
            current_alice_key = current_alice_key[indices]
            # actualize the permutation list
            perm = perm[indices]

            # leaked_info += len(current_bob_key)? # we count each permutation comunicated as a bit of information leaked
        
        # number of blocks in the current iteration
        num_blocks = math.ceil(len(current_bob_key)/block_size) #math.ceil rounds up to the nearest integer
        
        # now we check the error parity and apply the Binary protocol
        for j in range(num_blocks):
            start_index = j*block_size
            end_index = min((j + 1)*block_size, len(current_bob_key)) # makes sure the end index does not exceed the length of the key
            
            # obtain the blocks
            bob_block = current_bob_key[start_index:end_index]
            alice_block = current_alice_key[start_index:end_index]
            
            # calculate the parity of the blocks
            parity_bob = np.sum(bob_block)%2
            parity_alice = np.sum(alice_block)%2
            leaked_info += 1  # alice sends a public bit to inform bob about the parity
            
            # if error parity is odd
            if parity_bob != parity_alice:
                # We use Binary protocol to find the error bit position.
                error_index, leaked_info = binary(alice_block, bob_block, leaked_info) # This returns the position relative to the blcok
                error_index = start_index + error_index # here we obtain the position relative to the key
                current_bob_key[error_index] = 1 - current_bob_key[error_index] # we correct the error bit

        # Change the block size for the next iteration
        block_size *= 2

    # calculate the inverse permutation
    inv_perm = np.zeros(len(perm), dtype=int)
    for i, p in enumerate(perm): # enumeratt() returns the index i and the value p of the element in the list
        inv_perm[p] = i # the original position i of the element now in p 

    # obtain the corrected key and recuperate alice_key in the original order
    current_bob_key = current_bob_key[inv_perm]
    current_alice_key = current_alice_key[inv_perm]
    

    return current_bob_key, current_alice_key, leaked_info

#________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________

alice_key = np.random.randint(2, size=n) #generate the bit string from which we will obtain the key
basis_Alice = np.random.randint(2, size=n) #generate de random selectrion of basis for alice
basis_Bob = basis_Alice.copy() # Bob has the same basis as alice (we skip shifting process)
total_raw_key = np.array([])

with open("Simulation results.txt", "a") as f:
    f.write(f"Theoretical Initial length: {n*2}\tn= {n}\tg = {g}\n")
    f.write(f"Basis: {''.join(map(str, basis_Alice))}\n")
    f.write(f"alice_key: {''.join(map(str, alice_key))}\n")

circuits = [] # list to store the quantum circuits for each qubit
for i in tqdm(range(n)):
    qc = qk.QuantumCircuit(1, 1)
    
    if alice_key[i] == 1:
        qc.x(0)
    if basis_Alice[i] == 1:
        qc.h(0)

    if np.random.rand() < g: # noisy channel
        qc.x(0)

    if basis_Bob[i] == 1:
        qc.h(0)

    qc.measure(0, 0)
    circuits.append(qc)

# backend = service.least_busy(simulator=False, operational=True)
# print("Usando backend:", backend.name)
backend = service.backend("ibm_brisbane")
circuits = qk.transpile(circuits, backend=backend)

with Session(backend=backend) as session:
    sampler = Sampler()
    job = sampler.run(circuits, shots=1)
    print("Job ID:", job.job_id())
    result = job.result()

bob_key = np.array([
    int(max(res.data.c.get_counts().items(), key=lambda item: item[1])[0])
    for res in result
])

with open("Simulation results.txt", "a") as f:
    f.write(f"backend  : {backend.name}\n")
    f.write(f"bob_key  : {''.join(map(str, bob_key))}\n")

    
# Now Alice and Bob compare a subset of their obtained keys to check for eavesdropping.

size = round(len(alice_key)/3) # Compare a third of the bits in the destilled keys obtained. (len(alice_key)=len(bob_key))
if size != 0: 
    subset = random.sample(range(len(alice_key)), size) # Select the bits to compare at random

    alice_subset = alice_key[subset] # generate the subsets I am going to compare
    bob_subset = bob_key[subset]

    # Obtain the keys without the bits that were compared
    Alice_key = np.array([alice_key[i] for i in range(len(alice_key)) if i not in subset])
    Bob_key   = np.array([bob_key[i] for i in range(len(bob_key)) if i not in subset])
    
    # calculate the error in the channel
    P_error= (np.count_nonzero(alice_subset != bob_subset))/size
    with open("Simulation results.txt", "a") as f:
        f.write(f"size: {size}\tQBER: {P_error:.9f}\n")


with open("Simulation results.txt", "a") as f:
    f.write(f"alice_key after comparing subsets: {''.join(map(str, Alice_key))}\n")
    f.write(f"bob_key after comparing subsets: {''.join(map(str, Bob_key))}\n")
    f.write(f"# errors (unknown by them): {np.count_nonzero(Alice_key!=Bob_key)}\n")



# Check for the errors in the obtained keys
print(np.count_nonzero(Alice_key!=Bob_key))

# Apply the CASCADE protocol to the keys obtained by Alice and Bob
final_bob_key, final_alice_key, final_info= CASCADE(Alice_key, Bob_key, g)

# Check if the errors have been corrected. final_alice_key shoud be the same as the input argument Alice_key
print(np.count_nonzero(final_alice_key!=final_bob_key))

with open("Simulation results.txt", "a") as f:
    f.write(f"final lenght: {final_bob_key.size}\n")
    f.write(f"final_alice_key: {''.join(map(str, final_alice_key))}\n")
    f.write(f"final_bob_key  : {''.join(map(str, final_bob_key))}\n")
    f.write(f"# errors final key (unknown by them): {np.count_nonzero(final_alice_key!=final_bob_key)}\n")
    f.write(f"leaked info: {final_info}\n")
    f.write(f"indices subset: {subset}\n")
    f.write(f" \n")
    f.write(f" \n")
   


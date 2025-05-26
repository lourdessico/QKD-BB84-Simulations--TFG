# QKD-BB84-Simulations--TFG
This repository contains the code used in the BB84 Quantum Key Distribution protocol simulations, developed as part of the undergraduate thesis of Lourdes Simón Codina. They are implemented using IBM's Qiskit framework.

### Codes explanation
- `AerSim_eavesdropping.py`  --> This is the code of a BB84 protocol simulation with eavesdropping classical attack using AerSimulator. Alice and Bob have the same basis, so we skip the step of discarding the bits where the basis do not match (since we do not use those for anything) and n=# bits of initial key/2 (because 50% would be discarted but we "already did that"). 
    
    - The results of this code are saved at `AerSim_Prob_eve_correct.txt` and `AerSim_Detected eavesdropping.txt`, and graphed at `AerSim_Prob_deteccion_eve_vs_ñ.png` were ñ=number of qbits compared.

- `AerSim_eavesdropping_Breidbart Basis.py`  --> This is the code of a BB84 protocol simulation with eavesdropping attack using the Breidbart Basis and AerSimulator. Again, Alice and Bob have the same basis, so we skip the step of discarding the bits where the basis do not match.

    - The results of this code are saved at `AerSim_Prob_eve_correct_Breidbart.txt` and `AerSim_Detected eavesdropping_Breidbart.txt`, and graphed at `AerSim_Prob_deteccion_eve_vs_ñ_Breidbart.png`.

- `BB84_Breidbart Basis_bitflip p_1qbit.py`  --> This is the code of a BB84 protocol simulation with eavesdropping attack using the Breidbart Basis and AerSimulator. Again, Alice and Bob have the same basis, so we skip the step of discarding the bits where the basis do not match. The quantum channel has a bitflip error probability p.
 
- `BB84_Breidbart Basis_phaseshift q_1qbit.py`  --> This is the code of a BB84 protocol simulation with eavesdropping attack using the Breidbart Basis and AerSimulator. Again, Alice and Bob have the same basis, so we skip the step of discarding the bits where the basis do not match. The quantum channel has a phaseshift error probability q.
 
- `BB84_Breidbart Basis_bitflip+phaseshift g_1qbit.py`  --> This is the code of a BB84 protocol simulation with eavesdropping attack using the Breidbart Basis and AerSimulator. Again, Alice and Bob have the same basis, so we skip the step of discarding the bits where the basis do not match. The quantum channel has a bitflip+phaseshif error probability g.
 
- `Backend BB84_error Y_g_CASCADE.py`  --> This is the code of a BB84 protocol simulation using IBM's real backend ibm_brisbane. Again, Alice and Bob have the same basis, so we skip the step of discarding the bits where the basis do not match. The quantum channel has a bitflip+phaseshif error probability g. Finally, the CASCADE error correction
protocol is applied.

  - The results of this code are saved at `Simulation results.txt`

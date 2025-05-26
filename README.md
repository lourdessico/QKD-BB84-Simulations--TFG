# QKD-BB84-Simulations--TFG
This repository contains the code used in the BB84 Quantum Key Distribution protocol simulations, developed as part of the undergraduate thesis of Lourdes Simón Codina.

### Codes explanation
- `Backend_eavesdropping.py`  --> This is the principal code of a BB84 protocol simulation with eavesdropping classical attack. Alice and Bob have the same basis, so I skip the step of discarding the bits where the basis do not match (since we do not use those for anything) and n=# bits of initial key/2 (because 50% would be discarted but we "already did that") .
    
    - The results of this code are saved at `Backend_Prob_eve_correct.txt` and `Backend_Detected eavesdropping.txt`, and graphed at `Backend_Proba_deteccion_eve_vs_ñ.png`were ñ=number of qbits compared.

- `Backend_eavesdropping_Breidbart Basis.py`  --> This is the principal code of a BB84 protocol simulation with eavesdropping attack using the Breidbart Basis. Alice and Bob have the same basis, so I skip the step of discarding the bits where the basis do not match (since we do not use those for anything) and n=# bits of initial key/2 (because 50% would be discarted but we "already did that")

    - The results of this code are saved at `Backend_Prob_eve_correct_Breidbart.txt` and `Backend_Detected eavesdropping_Breidbart.txt`, and graphed at `Backend_Prob_deteccion_eve_vs_ñ_Breidbart.png`.

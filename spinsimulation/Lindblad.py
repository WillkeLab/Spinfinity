# import python libraries
import numpy as np

# import other SpinFinity packages
from spinsimulation.SpinSys import SpinSys
import spinsimulation.SpinHamiltonian as hamil 
import spinsimulation.calcBath as bath

# import qutip code
from qutip import *


# Content

# - calculate collapse operators with off diagonal elements
# - Filter rates and kick out values below a threshhold
# - Translator into QuTIp objects 
# -   
# - Lindblad master equation solver for spin systems
# (- Current from density matrix)
# -  


def test():
    """
    To test if Lindblad package was successfully imported
    """
    print("hello!")


def collapse_operators(spinsys):
    """
    Build Lindblad relaxation operators from the system's rates.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing eigenvectors, spin operators, and populations.

    Returns
    -------
    list[Qobj]
        Off-diagonal relaxation operators in the energy eigenbasis.
    """ 

    hamil.calcEigStates(spinsys)  # QUESTION should this be recalculated every time?
    bath.calcRates(spinsys)  
    bath.calcPopulations(spinsys)
    Rates_Summed = spinsys.Rates_Summed

    N_levels = len(spinsys.E_All)    # number of energy levels (NxN: dimension of Lindblad operators)

    # calculate relaxation operators
    collapse_ops = []   # empty list to store operators

    for i_initial in range(N_levels):
        for j_final in range(N_levels):

            if i_initial == j_final:
                continue  # skip diagonal elements

            E_op = np.zeros((N_levels, N_levels))    # matrix filled with zeros of dimension NxN as base for Lindblad hopping operators
            E_op[j_final][i_initial] = 1    # hopping operator from energy i to j

            L_ij = np.sqrt(Rates_Summed[i_initial][j_final]) * E_op    # multiply E_op with rates/thermal prefactor            
            ## for testing: 
            print(f'L[{i_initial}][{j_final}] = {L_ij}')

            collapse_ops.append(Qobj(L_ij))  # append new L_ij to the list of collapse operators


    return collapse_ops 








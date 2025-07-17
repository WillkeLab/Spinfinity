import numpy as np
import scipy as sp
import scipy.constants as const
from SpinSys import SpinSys

def calcSpinOperators(spinsys: SpinSys):
        """
        Calculate and assign spin operator matrices (Sx, Sy, Sz) for each spin in the system.
        This function computes the spin operator matrices for all spins in the given `SpinSys` object.
        For each spin, the corresponding Sx, Sy, and Sz matrices are constructed using Kronecker products,
        such that each operator acts on the correct subspace of the total Hilbert space. The results are
        stored in the `spinsys.Sx`, `spinsys.Sy`, and `spinsys.Sz` attributes.

        Parameters
        ----------
        spinsys : SpinSys
            An instance of the SpinSys class containing information about the spin system,
            including the number of spins (`NSpins`), the list of spin quantum numbers (`Spins`),
            and the total Hilbert space dimension (`dimensionOfMatrix`). 

        Returns
        -------
        None
            This function modifies the `spinsys` object in place and does not return a value.

        Notes
        -----
        The function assumes that the `calcAngMomMatrices` function is available and returns
        the angular momentum matrices for a given spin quantum number. The spin operator matrices
        are constructed such that each operator acts non-trivially only on its corresponding spin
        and as the identity on all other spins.
        """
        
        spinsys.Sx = np.zeros((spinsys.NSpins, spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix), dtype=complex)
        spinsys.Sy = np.zeros((spinsys.NSpins, spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix), dtype=complex)
        spinsys.Sz = np.zeros((spinsys.NSpins, spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix), dtype=complex)

        # Precompute identity matrices for Kronecker products
        identity_matrices = [np.eye(int(2 * spin + 1)) for spin in spinsys.Spins]

        # Loop through each spin
        for i_spin in range(spinsys.NSpins):
            # Calculate angular momentum matrices for the given spin type
            sigma_x, sigma_y, sigma_z = calcAngMomMatrices(s=spinsys.Spins[i_spin])

            # Initialize temporary variables for spin operators
            SxTemp = sigma_x
            SyTemp = sigma_y
            SzTemp = sigma_z

            # Apply Kronecker product for previous spins
            for j in range(i_spin):
                SxTemp = np.kron(identity_matrices[j], SxTemp)
                SyTemp = np.kron(identity_matrices[j], SyTemp)
                SzTemp = np.kron(identity_matrices[j], SzTemp)

            # Apply Kronecker product for following spins
            for j in range(i_spin + 1, spinsys.NSpins):
                SxTemp = np.kron(SxTemp, identity_matrices[j])
                SyTemp = np.kron(SyTemp, identity_matrices[j])
                SzTemp = np.kron(SzTemp, identity_matrices[j])

            # Store the calculated spin operators in the class attributes
            spinsys.Sx[i_spin, :, :] = SxTemp
            spinsys.Sy[i_spin, :, :] = SyTemp
            spinsys.Sz[i_spin, :, :] = SzTemp

        return    

def calcAngMomMatrices(s: float):
    """
    Calculate the angular momentum matrices S_x, S_y, and S_z for a given spin quantum number s.

    Parameters
    ----------
    s : int or float
        Spin quantum number (can be integer or half-integer).

    Returns
    -------
    S_x : numpy.ndarray
        The matrix representation of the S_x (x-component of spin) operator.
    S_y : numpy.ndarray
        The matrix representation of the S_y (y-component of spin) operator.
    S_z : numpy.ndarray
        The matrix representation of the S_z (z-component of spin) operator.

    Notes
    -----
    The matrices are constructed in the basis of eigenstates of S_z, with m ranging from -s to +s.
    The raising (S_p) and lowering (S_m) operators are used to construct S_x and S_y.
    """  
    m_values = np.arange(-s, s + 1, 1)  # m = -s to +s

    # Create raising and lowering operators S_p (creation) and S_m (annihilation)
    S_p = np.zeros((len(m_values), len(m_values)), dtype=complex)
    S_m = np.zeros((len(m_values), len(m_values)), dtype=complex)

    # Fill in the elements of S_p and S_m
    for m1 in m_values:
        for m2 in m_values:
            if m1 == m2 + 1:
                S_p[int(m2 + s), int(m1 + s)] = np.sqrt((s - m2) * (s + m2 + 1))
            elif m1 == m2 - 1:
                S_m[int(m2 + s), int(m1 + s)] = np.sqrt((s + m2) * (s - m2 + 1))

    # Calculate S_x, S_y, and S_z
    S_x = (S_p + S_m) / 2
    S_y = (S_p - S_m) / (2j)
    S_z = np.diag(-m_values)

    return S_x, S_y, S_z

## Spin Hamiltonian Functions ## 
def calcEigEnergies(spinsys: SpinSys):
    """
    Constructs the spin Hamiltonian for the given spin system, solves for its eigenvalues and eigenvectors,
    and stores the sorted eigenenergies and corresponding eigenvectors in the spin system object.
    The function sequentially adds various interaction terms to the Hamiltonian, including Zeeman, tip field,
    zero-field splitting, exchange interaction, and dipolar coupling. It then diagonalizes the Hamiltonian,
    sorts the eigenenergies, and updates the spin system with the results.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing all relevant parameters and matrices. The results of the calculation
        are stored as attributes of this object.

    Returns
    -------
    None
        This function modifies the `spinsys` object in place, updating its Hamiltonian, sorted eigenenergies,
        eigenvectors, and related attributes.
    """
    
    # This function builds the Spin Hamiltonian and then solves it 
    spinsys.H = np.zeros((spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix), dtype=complex)  # Important to reset the spin Hamiltonian
    addZeeman(spinsys)
    addTipField(spinsys)
    addZeroField(spinsys)
    addExchangeInteraction(spinsys)
    calcDipolarCoupling(spinsys) # This two functions are split because one may also want to change the D0 entries himself
    addDipolarCoupling(spinsys)

    # Solving the Hamiltonian and Sorting its Result 
    E, Evec = np.linalg.eigh(spinsys.H)
    spinsys.E_sort = np.sort(E)
    Eindex = np.argsort(E)
    spinsys.E_All = spinsys.E_sort - spinsys.E_sort[0]
    spinsys.E_All_inGHz = spinsys.E_All * spinsys.meVtoGHzConversion
    spinsys.eigVectors = Evec[:, Eindex]

    # Old stuff...
    # Build the basis states in vector from the solutions of the Sz operator
    # Sz_Operator = np.sum(spinsys.Sz, axis=0)
    # E_b,Evec_b = np.linalg.eigh(-Sz_Operator)
    
    # sort them via the eigenvalue
    #Eindex_b = np.argsort(E_b)
    #self.basisVectors = Evec_b[:, Eindex_b]

    # Ensure the first component of the lowest energy eigenvector is positive
    # for i in range(self.dimensionOfMatrix):
    #     if np.real(self.eigVectors[0, i]) < 0:
    #         self.eigVectors[:, i] = -self.eigVectors[:, i]

    #self.calcEigStates()

def addZeeman(spinsys: SpinSys):
    """
    Adds the Zeeman term to the Hamiltonian of the spin system.
    The Zeeman term describes the interaction between the magnetic moment of each spin and an external magnetic field.
    This function updates the Hamiltonian `H` of the provided `spinsys` object by adding the Zeeman contribution for each spin.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the Hamiltonian (`H`), the number of spins (`NSpins`), the g-factor vector (`gfactorvector`),
        the Bohr magneton (`muB`), the external magnetic field vector (`B`), and the spin operator matrices (`Sx`, `Sy`, `Sz`).

    Returns
    -------
    None
        The function modifies the `H` attribute of the `spinsys` object in place.
    
    Notes
    -----
    The Zeeman term is added to each spin as a global magnetic field.
    """

    # Loop through each spin in the system
    for i_spin in range(spinsys.NSpins): 

        # Apply Zeeman interaction to Hamiltonian H
        spinsys.H += - spinsys.gfactorvector[i_spin] * spinsys.muB * spinsys.B[0] * spinsys.Sx[i_spin,:, :]
        spinsys.H += - spinsys.gfactorvector[i_spin] * spinsys.muB * spinsys.B[1] * spinsys.Sy[i_spin,:, :]
        spinsys.H += - spinsys.gfactorvector[i_spin] * spinsys.muB * spinsys.B[2] * spinsys.Sz[i_spin,:, :]

    return

def addTipField(spinsys: SpinSys):
    """
    Adds or subtracts the tip magnetic field contribution to the Hamiltonian for the readout spin.
    Depending on the value of `spinsys.tip`, this function either adds or subtracts the effect of the tip field
    on the readout spin in the spin system Hamiltonian. The contribution is calculated for each spatial component (x, y, z)
    using the corresponding spin operators and the tip field vector.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the Hamiltonian, spin operators, g-factor vector, Bohr magneton,
        tip field vector, and the index of the readout spin.

    Returns
    -------
    None    

    Notes
    -----
    - If `spinsys.tip` is 'a', the tip field is added to the Hamiltonian.
    - Otherwise, the tip field is subtracted from the Hamiltonian.
    - Only the readout spin is affected by the tip field.
    """

    # Adding the Tipfield to the Readout Spin only
    if spinsys.tip == 'a':
        # Add the tip field contribution to the Hamiltonian H
        spinsys.H += spinsys.gfactorvector[spinsys.ReadoutSpin] * spinsys.muB * spinsys.BTip[0] * spinsys.Sx[spinsys.ReadoutSpin,:, :]
        spinsys.H += spinsys.gfactorvector[spinsys.ReadoutSpin] * spinsys.muB * spinsys.BTip[1] * spinsys.Sy[spinsys.ReadoutSpin,:, :]
        spinsys.H += spinsys.gfactorvector[spinsys.ReadoutSpin] * spinsys.muB * spinsys.BTip[2] * spinsys.Sz[spinsys.ReadoutSpin,:, :]
    else:
        # Subtract the tip field contribution to the Hamiltonian H
        spinsys.H += - spinsys.gfactorvector[spinsys.ReadoutSpin] * spinsys.muB * spinsys.BTip[0] * spinsys.Sx[spinsys.ReadoutSpin,:, :]
        spinsys.H += - spinsys.gfactorvector[spinsys.ReadoutSpin] * spinsys.muB * spinsys.BTip[1] * spinsys.Sy[spinsys.ReadoutSpin,:, :]
        spinsys.H += - spinsys.gfactorvector[spinsys.ReadoutSpin] * spinsys.muB * spinsys.BTip[2] * spinsys.Sz[spinsys.ReadoutSpin,:, :]

    return

def addZeroField(spinsys: SpinSys):
    """
    Adds zero-field splitting (out-of-plane anisotropy) to the spin Hamiltonian.
    This function modifies the Hamiltonian of the given spin system by adding
    an out-of-plane anisotropy term for each spin. The anisotropy is determined
    by the Dvector attribute of the spin system and is applied as a quadratic
    term in the Sz operator for each spin.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object to which the zero-field splitting term will be added.
        It must have the attributes:
            - NSpins: int, number of spins in the system.
            - Dvector: array-like, anisotropy constants for each spin.
            - Sz: array-like, spin-z operators for each spin.
            - H: array-like, the Hamiltonian matrix to be modified.

    Returns
    -------
    None
        The function modifies the Hamiltonian in-place.
    """
    for i_spin in range(spinsys.NSpins):
        spinsys.H += spinsys.Dvector[i_spin]*(spinsys.Sz[i_spin,:,:] @ spinsys.Sz[i_spin,:,:])

    return

def addExchangeInteraction(spinsys: SpinSys):
    """
    Adds the exchange interaction terms to the spin system Hamiltonian.
    This function iterates over all unique pairs of spins in the given spin system
    and adds the exchange interaction contributions to the Hamiltonian matrix `H`.
    The exchange interaction is parameterized by the `Jvector` for each spin pair,
    and the terms are added for the x, y, and z spin components.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the Hamiltonian (`H`), the number of spins (`NSpins`),
        the exchange interaction vectors (`Jvector`), and the spin operator matrices (`Sx`, `Sy`, `Sz`).

    Returns
    -------
    None
        The function modifies the `H` attribute of the `spinsys` object in place.
    """
    # Adds the Exchange interaction to the Hamiltonian
    index_counter = 0  # Initialize the index counter (0-based indexing in Python)

    for i_spin in range(spinsys.NSpins):
        for j_spin in range(spinsys.NSpins):
            if i_spin < j_spin:  # We only look at the case where i_spin < j_spin
                # Add the spin-spin interaction terms to the Hamiltonian H
                spinsys.H += spinsys.Jvector[index_counter, 0] * spinsys.Sx[i_spin, :, :] @ spinsys.Sx[j_spin, :, :]
                spinsys.H += spinsys.Jvector[index_counter, 1] * spinsys.Sy[i_spin, :, :] @ spinsys.Sy[j_spin, :, :]
                spinsys.H += spinsys.Jvector[index_counter, 2] * spinsys.Sz[i_spin, :, :] @ spinsys.Sz[j_spin, :, :]
                # Increment the index counter
                index_counter += 1
    return
    
def calcDipolarCoupling(spinsys: SpinSys):
    """
    Calculates the dipolar coupling constants for each spin pair in the given spin system.
    This function iterates over all unique pairs of spins in the provided `SpinSys` object,
    computes the distance between each pair, and calculates the dipolar coupling constant
    based on their relative positions and physical constants. The results are stored in
    the `D0` attribute of the `SpinSys` object.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing information about the number of spins, their
        positions, physical constants, and storage arrays for dipolar couplings.

    Returns
    -------
    None
        The function updates the `D0` attribute of the `spinsys` object in place.
    """

    # Calculates the Dipolar Coupling Constants of each Spin-Pair based on their relative distance
    index_counter = 0
    for i_spin in range(spinsys.NSpins):
        for j_spin in range(spinsys.NSpins):
            if i_spin > j_spin:
                if spinsys.DipoleBool[index_counter]:
                    # Calculate the distance between atoms i_spin and j_spin
                    distance_temp = np.linalg.norm(
                        (spinsys.AtomPositions[i_spin, :] - spinsys.AtomPositions[j_spin, :]) * spinsys.latticeLengthXY * 1e-9
                    )
                # Calculate the dipolar coupling strength D0
                    spinsys.D0[j_spin, i_spin] = (
                        spinsys.gfactorvector[i_spin] *
                        spinsys.gfactorvector[j_spin] *
                        spinsys.muBHz ** 2 *
                        spinsys.mu0 /
                        (4 * np.pi * distance_temp ** 3) *
                        spinsys.JtomeVConversion
                    )
                    index_counter += 1

    return

def addDipolarCoupling(spinsys: SpinSys):
    """
    Adds dipolar coupling interactions to the spin Hamiltonian of a spin system.
    This function iterates over all unique pairs of spins in the given spin system and updates
    the system's Hamiltonian (`spinsys.H`) by adding the dipolar coupling terms. The dipolar
    interaction is calculated based on the positions of the spins and their respective spin
    operators. The function ensures that each pair is only considered once and avoids self-interaction.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing spin positions, spin operators, dipolar coupling
        constants, and the Hamiltonian matrix to be updated.
    Returns
    -------
    None
        The function modifies the Hamiltonian in place by adding the dipolar coupling terms.

    Notes
    -----
    The dipolar coupling term added to the Hamiltonian is of the form:
        D0 * (S1 · S2 - 3 (S1 · r̂)(S2 · r̂))
    where D0 is the dipolar coupling constant, S1 and S2 are spin operators, and r̂ is the
    normalized vector connecting the two spins.
    """
    
    # Loop over spins
    for i_spin in range(spinsys.NSpins):
        for j_spin in range(i_spin + 1):  # Only consider j_spin <= i_spin to avoid double counting
            if i_spin != j_spin:
                # Calculate the normalized connection vector (rhat)
                diff_vec = spinsys.AtomPositions[i_spin, :] - spinsys.AtomPositions[j_spin, :]
                rhat = diff_vec / np.linalg.norm(diff_vec)

                # Add the dipolar interaction to the Hamiltonian
                spinsys.H += (
                    spinsys.D0[j_spin, i_spin]
                    * (spinsys.Sx[j_spin, :, :] @ spinsys.Sx[i_spin, :, :]
                        + spinsys.Sy[j_spin, :, :] @ spinsys.Sy[i_spin, :, :]
                        + spinsys.Sz[j_spin, :, :] @ spinsys.Sz[i_spin, :, :])
                )

                # Subtract the 3*(D0) term with the projection of spin operators along rhat
                spinsys.H -= (
                    3 * spinsys.D0[j_spin, i_spin]
                    * ((spinsys.Sx[j_spin, :, :] * rhat[0]
                        + spinsys.Sy[j_spin, :, :] * rhat[1]
                        + spinsys.Sz[j_spin, :, :] * rhat[2])
                    @ (spinsys.Sx[i_spin, :, :] * rhat[0]
                        + spinsys.Sy[i_spin, :, :] * rhat[1]
                        + spinsys.Sz[i_spin, :, :] * rhat[2]))
                )
    
    return

def calcEigStates(spinsys: SpinSys):
    """
    Calculates and constructs the eigenstates of a spin system in Bra-Ket notation.
    This function generates the basis states, sorts the eigenvectors by their absolute values,
    and constructs human-readable representations of the eigenstates, including their energies
    and dominant basis state contributions. The results are stored as attributes in the provided
    `spinsys` object.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the Hamiltonian, eigenvectors, eigenvalues, and
        other relevant properties. The function updates this object with additional attributes
        representing the eigenstates in various notations.

    Returns
    -------
    None
        This function modifies the `spinsys` object in place and does not return a value.
    Attributes Added to `spinsys`
    -----------------------------
    states : np.ndarray of str
        Full eigenstate expressions in Bra-Ket notation with energies.
    statesOnlyLHS : np.ndarray of str
        Left-hand side of the eigenstate expressions (state label and energy).
    statesWithoutE : np.ndarray of str
        State labels without energy information.
    statesOnlyRHS : np.ndarray of str
        Right-hand side of the eigenstate expressions, showing only the top two basis state contributions.
    basisStates : np.ndarray of str
        Basis states in Bra-Ket notation.
    """

    # Create the basis states in Bra-Ket notation
    basisStates = np.empty(spinsys.dimensionOfMatrix, dtype=object)
    for i in range(spinsys.dimensionOfMatrix):
        basisStates[i] = "|"
        for j in range(spinsys.NSpins):
            if j != spinsys.NSpins - 1:
                basisStates[i] += str(np.real(spinsys.Sz[j, i, i])) + ","
            else:
                basisStates[i] += str(np.real(spinsys.Sz[j, i, i]))
        basisStates[i] += ">"

    # Initialize the states and other strings
    states = np.empty(spinsys.dimensionOfMatrix, dtype=object)
    statesOnlyLHS = np.empty(spinsys.dimensionOfMatrix, dtype=object)
    statesWithoutE = np.empty(spinsys.dimensionOfMatrix, dtype=object)
    statesOnlyRHS = np.empty(spinsys.dimensionOfMatrix, dtype=object)

    # Sort the absolute values of the eigenvectors
    SortedIndices = np.argsort(-np.abs(spinsys.eigVectors), axis=0)  # Sort by absolute value, descending
    SortedEigenvectors = np.empty_like(spinsys.eigVectors, dtype=complex)

    for j in range(spinsys.dimensionOfMatrix):
        SortedEigenvectors[:, j] = spinsys.eigVectors[SortedIndices[:, j], j]

    # Construct states in bracket notation
    for i in range(spinsys.dimensionOfMatrix):
        states[i] = f"|{i}> (E = {round(spinsys.E_All[i],4)}) = "
        statesOnlyLHS[i] = f"|{i}> (E = {round(spinsys.E_All[i],4)})"
        statesWithoutE[i] = f"|{i}>"
        statesOnlyRHS[i] = ""

        for j in range(spinsys.dimensionOfMatrix):
            if np.abs(SortedEigenvectors[j, i]) > 1e-4:  # Threshold for relevance
                sign = "+" if np.real(SortedEigenvectors[j, i]) >= 0 else ""
                states[i] += f"{sign}{SortedEigenvectors[j, i]}{basisStates[SortedIndices[j, i]]}"

                if j < 2:  # Only include the top two contributions in statesOnlyRHS
                    statesOnlyRHS[i] += f"{sign}{SortedEigenvectors[j, i]}{basisStates[SortedIndices[j, i]]}"

    # Store the results in the Sys object
    spinsys.states = states
    spinsys.statesOnlyLHS = statesOnlyLHS
    spinsys.statesWithoutE = statesWithoutE
    spinsys.statesOnlyRHS = statesOnlyRHS
    spinsys.basisStates = basisStates

    return
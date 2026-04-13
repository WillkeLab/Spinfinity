"""
Spin System Hamiltonian Construction and Analysis Module.

This module provides functionality for constructing and analyzing the
quantum mechanical Hamiltonian of multi-spin systems. It handles the
calculation of spin operators, eigenstate determination, and various
interaction terms including Zeeman, exchange, dipolar, and anisotropy
effects.

Key Features
------------
- Spin operator calculation using Kronecker products for multi-spin systems
- Angular momentum matrix generation for arbitrary spin quantum numbers
- Hamiltonian construction with multiple interaction terms
- Eigenstate calculation and analysis in Bra-Ket notation
- Stevens operator support for high-order magnetic anisotropy
- Dipolar coupling computation based on spatial coordinates
- Eigenstate overlap calculations for subsystem analysis

Main Components
---------------
Spin Operators
    - calc_SpinOperators: Constructs Sx, Sy, Sz matrices for all spins
    - calc_AngMomMatrices: Generates angular momentum matrices for a
      given spin quantum number
Hamiltonian Construction
    - calc_EigEnergies: Constructs full Hamiltonian and solves for eigenenergies
    - add_Zeeman: Adds Zeeman interaction term (external magnetic field)
    - add_TipField: Adds tip field contribution for readout spin
    - add_ZeroField: Adds zero-field splitting and anisotropy terms
    - add_ExchangeInteraction: Adds spin-spin exchange coupling
    - calc_DipolarCoupling: Calculates dipolar coupling constants
    - add_DipolarCoupling: Adds dipolar interaction to Hamiltonian
Eigenstate Analysis
    - calc_EigStates: Generates eigenstate representations in Bra-Ket notation
Stevens Operators
    - stevens_operator: Generates Stevens operator matrices for magnetic anisotropy
    - set_Stevens: Sets Stevens operator coefficients
    - anticommutator: Computes anticommutator of two operators
The module assumes spin systems are represented using the basis of eigenstates of Sz,
with states ordered from m = -s to m = +s. All interaction matrices are constructed
using Kronecker products to embed single-spin operators into the full Hilbert space.
"""
import numpy as np
import plotly.graph_objects as go

from spinfinity.SpinSys import SpinSys


# region SpinOperators
def calc_SpinOperators(spinsys: SpinSys):
    """
    Compute the spin operator matrices for all spins in the given `SpinSys` object.

    For each spin, the corresponding Sx, Sy, and Sz matrices are
    constructed using Kronecker products, such that each operator acts
    on the correct subspace of the total Hilbert space. The results are
    stored in the `spinsys.Sx`, `spinsys.Sy`, and `spinsys.Sz`
    attributes.
    This function modifies the `spinsys` object in place and does not return a value.

    Parameters
    ----------
    spinsys : SpinSys
        An instance of the SpinSys class containing information about the spin system,
        including the number of spins (`NSpins`), the list of spin
        quantum numbers (`Spins`), and the total Hilbert space
        dimension (`dimensionOfMatrix`). 

    Notes
    -----
    The function assumes that the `calc_AngMomMatrices` function is available 
    and returns the angular momentum matrices for a given spin quantum number. 
    The spin operator matrices are constructed such that each operator acts
    non-trivially only on its corresponding spin and as the identity on all other spins.
    """
    spinsys.Sx = np.zeros(
        (spinsys.NSpins, spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix),
        dtype=complex,
    )
    spinsys.Sy = np.zeros(
        (spinsys.NSpins, spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix),
        dtype=complex,
    )
    spinsys.Sz = np.zeros(
        (spinsys.NSpins, spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix),
        dtype=complex,
    )

    # Precompute identity matrices for Kronecker products
    identity_matrices = [np.eye(int(2 * spin + 1)) for spin in spinsys.Spins]

    # Loop through each spin
    for i_spin in range(spinsys.NSpins):
        # Calculate angular momentum matrices for the given spin type
        sigma_x, sigma_y, sigma_z = calc_AngMomMatrices(s=spinsys.Spins[i_spin])

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


def calc_AngMomMatrices(s: float):
    """
    Calculate the angular momentum matrices for a given spin quantum number s.

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
    The matrices are constructed in the basis of eigenstates of S_z, 
    with m ranging from -s to +s.
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

# endregion SpinOperators

# region Hamiltonian


def calc_EigStates(spinsys: SpinSys):
    """
    Construct the eigenstates of a spin system in Bra-Ket notation.

    This function generates the basis states, sorts the eigenvectors by
    their absolute values, and constructs human-readable
    representations of the eigenstates, including their energies and
    dominant basis state contributions. The results are stored as
    attributes in the provided `spinsys` object. This function modifies
    the `spinsys` object in place and does not return a value.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the Hamiltonian, eigenvectors,
        eigenvalues, and other relevant properties. The function updates
        this object with additional attributes representing the
        eigenstates in various notations.
    
    Notes
    -----
    Attributes added to `spinsys`:
    states : np.ndarray of str
        Full eigenstate expressions in Bra-Ket notation with energies.
    statesOnlyLHS : np.ndarray of str
        Left-hand side of the eigenstate expressions (state label and energy).
    statesWithoutE : np.ndarray of str
        State labels without energy information.
    statesOnlyRHS : np.ndarray of str
        Right-hand side of the eigenstate expressions, showing only the
        top two basis state contributions.
    basisStates : np.ndarray of str
        Basis states in Bra-Ket notation.
    """
    # Ensure the Hamiltonian is calculated and solved 
    if not hasattr(spinsys, 'eigVectors') or spinsys.eigVectors is None:
        calc_EigEnergies(spinsys)

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
    # Sort by absolute value, descending
    SortedIndices = np.argsort(-np.abs(spinsys.eigVectors), axis=0)  
    SortedEigenvectors = np.empty_like(spinsys.eigVectors, dtype=complex)

    for j in range(spinsys.dimensionOfMatrix):
        SortedEigenvectors[:, j] = spinsys.eigVectors[SortedIndices[:, j], j]

    # Construct states in bracket notation
    for i in range(spinsys.dimensionOfMatrix):
        states[i] = f"|{i}> (E = {round(spinsys.E_All[i], 4)}) = "
        statesOnlyLHS[i] = f"|{i}> (E = {round(spinsys.E_All[i], 4)})"
        statesWithoutE[i] = f"|{i}>"
        statesOnlyRHS[i] = ""

        for j in range(spinsys.dimensionOfMatrix):
            if np.abs(SortedEigenvectors[j, i]) > 1e-4:  # Threshold for relevance
                sign = "+" if np.real(SortedEigenvectors[j, i]) >= 0 else ""
                states[i] += (
                    f"{sign}{SortedEigenvectors[j, i]:.3f}"
                    f"{basisStates[SortedIndices[j, i]]}"
                )

                if j < 2:  # Only include the top two contributions in statesOnlyRHS
                    statesOnlyRHS[i] += (
                        f"{sign}{SortedEigenvectors[j, i]}"
                        f"{basisStates[SortedIndices[j, i]]}"
                    )

    # Store the results in the Sys object
    spinsys.states = states
    spinsys.statesOnlyLHS = statesOnlyLHS
    spinsys.statesWithoutE = statesWithoutE
    spinsys.statesOnlyRHS = statesOnlyRHS
    spinsys.basisStates = basisStates

    return


def calc_EigEnergies(spinsys: SpinSys):
    """
    Construct the spin Hamiltonian for the given spin system and solve for eigenpairs.

    This function sequentially adds various interaction terms to the
    Hamiltonian, including Zeeman, tip field, zero-field splitting,
    exchange interaction, and dipolar coupling. It then diagonalizes
    the Hamiltonian, sorts the eigenenergies, and updates the spin
    system with the results. The sorted eigenenergies and corresponding
    eigenvectors are stored in the spin system object. This function
    modifies the `spinsys` object in place, updating its Hamiltonian,
    sorted eigenenergies, eigenvectors, and related attributes.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing all relevant parameters and
        matrices. The results of the calculation are stored as
        attributes of this object.      
    """
    # Ensure the spin operators are calculated before constructing the Hamiltonian
    if not hasattr(spinsys, 'Sx') or spinsys.Sx is None:
        calc_SpinOperators(spinsys)
    
    # This section initializes the Spin Hamiltonian and then solves it 
    spinsys.H = np.zeros(
        (spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix),
        dtype=complex,
    )  # Important to reset the spin Hamiltonian
    add_Zeeman(spinsys)
    add_TipField(spinsys)
    add_ZeroField(spinsys)
    add_ExchangeInteraction(spinsys)
    add_ExchangeOffDiagonal(spinsys)
    # calc_DipolarCoupling(spinsys)  # These two functions are split
    # because one may want to change the D0 entries manually.
    add_DipolarCoupling(spinsys)

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
    # Eindex_b = np.argsort(E_b)
    # self.basisVectors = Evec_b[:, Eindex_b]

    # Ensure the first component of the lowest energy eigenvector is positive
    # for i in range(self.dimensionOfMatrix):
    #     if np.real(self.eigVectors[0, i]) < 0:
    #         self.eigVectors[:, i] = -self.eigVectors[:, i]

    # self.calc_EigStates()


def add_Zeeman(spinsys: SpinSys):
    """
    Add the Zeeman term to the Hamiltonian of the spin system.

    The Zeeman term describes the interaction between the magnetic
    moment of each spin and an external magnetic field. This function
    updates the Hamiltonian `H` of the provided `spinsys` object by
    adding the Zeeman contribution for each spin.
    The function modifies the `H` attribute of the `spinsys` object in place

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the Hamiltonian (`H`), the
        number of spins (`NSpins`), the g-factor vector
        (`gfactorvector`), the Bohr magneton (`muB`), the external
        magnetic field vector (`B`), and the spin operator matrices
        (`Sx`, `Sy`, `Sz`).
    
    Notes
    -----
    The Zeeman term is added to each spin as a global magnetic field.
    """
    # Loop through each spin in the system
    for i_spin in range(spinsys.NSpins): 

        # Apply Zeeman interaction to Hamiltonian H
        spinsys.H += (
            -spinsys.gfactorvector[i_spin]
            * spinsys.muB
            * spinsys.B[0]
            * spinsys.Sx[i_spin, :, :]
        )
        spinsys.H += (
            -spinsys.gfactorvector[i_spin]
            * spinsys.muB
            * spinsys.B[1]
            * spinsys.Sy[i_spin, :, :]
        )
        spinsys.H += (
            -spinsys.gfactorvector[i_spin]
            * spinsys.muB
            * spinsys.B[2]
            * spinsys.Sz[i_spin, :, :]
        )

    return


def add_TipField(spinsys: SpinSys):
    """
    Add the tip magnetic field contribution to the Hamiltonian for the readout spin.

    Depending on the value of `spinsys.tip`, this function either adds
    or subtracts the effect of the tip field on the readout spin in the
    spin system Hamiltonian. The contribution is calculated for each
    spatial component (x, y, z) using the corresponding spin operators
    and the tip field vector.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the Hamiltonian, spin
        operators, g-factor vector, Bohr magneton, tip field vector,
        and the index of the readout spin.

    Notes
    -----
    - If `spinsys.tip` is 'a', the tip field is added to the Hamiltonian.
    - Otherwise, the tip field is subtracted from the Hamiltonian.
    - Only the readout spin is affected by the tip field.
    """
    # Adding the Tipfield to the Readout Spin only
    if spinsys.tip == 'a':
        # Add the tip field contribution to the Hamiltonian H
        spinsys.H += (
            spinsys.gfactorvector[spinsys.ReadoutSpin]
            * spinsys.muB
            * spinsys.BTip[0]
            * spinsys.Sx[spinsys.ReadoutSpin, :, :]
        )
        spinsys.H += (
            spinsys.gfactorvector[spinsys.ReadoutSpin]
            * spinsys.muB
            * spinsys.BTip[1]
            * spinsys.Sy[spinsys.ReadoutSpin, :, :]
        )
        spinsys.H += (
            spinsys.gfactorvector[spinsys.ReadoutSpin]
            * spinsys.muB
            * spinsys.BTip[2]
            * spinsys.Sz[spinsys.ReadoutSpin, :, :]
        )
    else:
        # Subtract the tip field contribution to the Hamiltonian H
        spinsys.H += (
            -spinsys.gfactorvector[spinsys.ReadoutSpin]
            * spinsys.muB
            * spinsys.BTip[0]
            * spinsys.Sx[spinsys.ReadoutSpin, :, :]
        )
        spinsys.H += (
            -spinsys.gfactorvector[spinsys.ReadoutSpin]
            * spinsys.muB
            * spinsys.BTip[1]
            * spinsys.Sy[spinsys.ReadoutSpin, :, :]
        )
        spinsys.H += (
            -spinsys.gfactorvector[spinsys.ReadoutSpin]
            * spinsys.muB
            * spinsys.BTip[2]
            * spinsys.Sz[spinsys.ReadoutSpin, :, :]
        )

    return


def add_ZeroField(spinsys: SpinSys):
    """
    Build and add the anisotropy Hamiltonian for the spin system.

    Supports either standard parameters (D, E) or Stevens operators.
    This function modifies spinsys.H in-place.
    """
    # Check if Stevens operators are used
    # If not, use standard D, E parameters
    use_stevens = getattr(spinsys, "use_Stevens", False)
    if not use_stevens:
        # D * Sz^2 + E(Sx^2 - Sy^2)
        for i_spin in range(spinsys.NSpins):
            spinsys.H += (
                spinsys.Dvector[i_spin]
                * (spinsys.Sz[i_spin, :, :] @ spinsys.Sz[i_spin, :, :])
                + spinsys.Evector[i_spin]
                * (
                    spinsys.Sx[i_spin, :, :] @ spinsys.Sx[i_spin, :, :]
                    - spinsys.Sy[i_spin, :, :] @ spinsys.Sy[i_spin, :, :]
                )
            )
            
    else:
        for i_spin in range(spinsys.NSpins):
            # Dict for this spin, e.g., {(2,0): value, (4,2): value}
            stevens_dict = spinsys.Stevens[i_spin]  
            
            for (a, b), coeff in stevens_dict.items():
                if coeff == 0:
                    continue  # skip zero terms

                # Generate the Stevens operator O_a^b for this spin
                # Get the base operator for spin i
                Oab_local = stevens_operator(spinsys.Spins[i_spin], a, b)

                # Embed into the full Hilbert space via Kronecker products
                Oab_full = 1
                for j in range(spinsys.NSpins):
                    dim = int(2 * spinsys.Spins[j] + 1)
                    if j == i_spin:
                        Oab_full = np.kron(Oab_full, Oab_local)
                    else:
                        Oab_full = np.kron(Oab_full, np.eye(dim))
                # Add to the Hamiltonian
                spinsys.H += coeff * Oab_full


def add_ExchangeInteraction(spinsys: SpinSys):
    """
    Add the exchange interaction terms to the spin system Hamiltonian.

    This function iterates over all unique pairs of spins in the given spin system
    and adds the exchange interaction contributions to the Hamiltonian matrix `H`.
    The exchange interaction is parameterized by the `Jvector` for each spin pair,
    and the terms are added for the x, y, and z spin components.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the Hamiltonian (`H`), the
        number of spins (`NSpins`), the exchange interaction vectors
        (`Jvector`), and the spin operator matrices (`Sx`, `Sy`, `Sz`).
    """
    # Adds the Exchange interaction to the Hamiltonian
    index_counter = 0  # Initialize the index counter (0-based indexing in Python)

    for i_spin in range(spinsys.NSpins):
        for j_spin in range(spinsys.NSpins):
            if i_spin < j_spin:  # We only look at the case where i_spin < j_spin
                # Add the spin-spin interaction terms to the Hamiltonian H
                spinsys.H += (
                    spinsys.Jvector[index_counter, 0]
                    * spinsys.Sx[i_spin, :, :]
                    @ spinsys.Sx[j_spin, :, :]
                )
                spinsys.H += (
                    spinsys.Jvector[index_counter, 1]
                    * spinsys.Sy[i_spin, :, :]
                    @ spinsys.Sy[j_spin, :, :]
                )
                spinsys.H += (
                    spinsys.Jvector[index_counter, 2]
                    * spinsys.Sz[i_spin, :, :]
                    @ spinsys.Sz[j_spin, :, :]
                )
                # Increment the index counter
                index_counter += 1
    return


def add_ExchangeOffDiagonal(spinsys: SpinSys):
    """
    Add off-diagonal exchange interaction terms to the spin system Hamiltonian.

    This function computes and adds all pairwise spin-spin exchange interactions
    to the Hamiltonian matrix. It iterates through unique spin pairs (i < j) and
    adds six types of exchange coupling terms: Jxy, Jxz, Jyz, Jyx, Jzx, and Jzy.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the Hamiltonian matrix H, spin operators
        (Sx, Sy, Sz), exchange coupling constants (Jxy, Jxz, Jyz, Jyx, Jzx, Jzy),
        and the total number of spins NSpins.
    """
    index_counter = 0  # Initialize the index counter (0-based indexing in Python)

    for i_spin in range(spinsys.NSpins):
        for j_spin in range(spinsys.NSpins):
            if i_spin < j_spin:  # We only look at the case where i_spin < j_spin
                # Add the spin-spin interaction terms to the Hamiltonian H
                spinsys.H += (
                    spinsys.Jxy[index_counter]
                    * spinsys.Sx[i_spin, :, :]
                    @ spinsys.Sy[j_spin, :, :]
                )
                spinsys.H += (
                    spinsys.Jxz[index_counter]
                    * spinsys.Sx[i_spin, :, :]
                    @ spinsys.Sz[j_spin, :, :]
                )
                spinsys.H += (
                    spinsys.Jyz[index_counter]
                    * spinsys.Sy[i_spin, :, :]
                    @ spinsys.Sz[j_spin, :, :]
                )
                spinsys.H += (
                    spinsys.Jyx[index_counter]
                    * spinsys.Sy[i_spin, :, :]
                    @ spinsys.Sx[j_spin, :, :]
                )
                spinsys.H += (
                    spinsys.Jzx[index_counter]
                    * spinsys.Sz[i_spin, :, :]
                    @ spinsys.Sx[j_spin, :, :]
                )
                spinsys.H += (
                    spinsys.Jzy[index_counter]
                    * spinsys.Sz[i_spin, :, :]
                    @ spinsys.Sy[j_spin, :, :]
                )
                # Increment the index counter
                index_counter += 1
    return
    

def calc_DipolarCoupling(spinsys: SpinSys):
    """
    Calculate the dipolar coupling constants for each spin pair.

    This function iterates over all unique pairs of spins in the
    provided `SpinSys` object, computes the distance between each pair,
    and calculates the dipolar coupling constant based on their
    relative positions and physical constants. The results are stored
    in the `D0` attribute of the `SpinSys` object.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing information about the number of spins, their
        positions, physical constants, and storage arrays for dipolar couplings.
    """
    # Calculates the Dipolar Coupling Constants of each Spin-Pair 
    # based on their relative distance
    index_counter = 0
    for i_spin in range(spinsys.NSpins):
        for j_spin in range(spinsys.NSpins):
            if i_spin > j_spin:
                if spinsys.DipoleBool[index_counter]:
                    # Calculate the distance between atoms i_spin and j_spin
                    distance_temp = np.linalg.norm(
                        (
                            spinsys.AtomPositions[i_spin, :]
                            - spinsys.AtomPositions[j_spin, :]
                        )
                        * spinsys.latticeLengthXY
                        * 1e-9
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


def add_DipolarCoupling(spinsys: SpinSys):
    """
    Add dipolar coupling interactions to the spin Hamiltonian of a spin system.

    This function iterates over all unique pairs of spins in the given
    spin system and updates the system's Hamiltonian (`spinsys.H`) by
    adding the dipolar coupling terms. The dipolar interaction is
    calculated based on the positions of the spins and their respective
    spin operators. The function ensures that each pair is only
    considered once and avoids self-interaction.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing spin positions, spin
        operators, dipolar coupling constants, and the Hamiltonian
        matrix to be updated.
        
    Notes
    -----
    The dipolar coupling term added to the Hamiltonian is of the form:
        D0 * (S1 · S2 - 3 (S1 · r̂)(S2 · r̂))
    where D0 is the dipolar coupling constant, S1 and S2 are spin
    operators, and rhat is the normalized vector connecting the two
    spins.
    """
    if np.any(spinsys.DipoleBool):
        for i_spin in range(spinsys.NSpins):
            for j_spin in range(i_spin + 1):  # avoid double counting
                if i_spin != j_spin:
                    # Calculate the normalized connection vector (rhat)
                    diff_vec = (
                        spinsys.AtomPositions[i_spin, :]
                        - spinsys.AtomPositions[j_spin, :]
                    )
                    rhat = diff_vec / np.linalg.norm(diff_vec)

                    # Add the dipolar interaction to the Hamiltonian
                    spinsys.H += (
                        spinsys.D0[j_spin, i_spin]
                        * (spinsys.Sx[j_spin, :, :] @ spinsys.Sx[i_spin, :, :]
                            + spinsys.Sy[j_spin, :, :] @ spinsys.Sy[i_spin, :, :]
                            + spinsys.Sz[j_spin, :, :] @ spinsys.Sz[i_spin, :, :])
                    )

                    # Subtract the 3 * D0 term with the projection of
                    # spin operators along rhat.
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

# endregion Hamiltonian

# region StevensOperators


def anticommutator(A, B):
    """
    Compute the anticommutator of two operators.

    Parameters
    ----------
    A : array_like
        First operator, typically a matrix or linear operator.
    B : array_like
        Second operator, typically a matrix or linear operator.

    Returns
    -------
    ndarray
        The anticommutator A @ B + B @ A.

    Notes
    -----
    The anticommutator is commonly used in quantum mechanics and quantum field
    theory, particularly for fermionic systems where operators satisfy canonical
    anticommutation relations.
    """
    return A @ B + B @ A


def stevens_operator(S, k, q, return_latex=False):
    """
    Generate Stevens operator O_a^b for spin S.

    Parameters
    ----------
    S : float
        Spin quantum number (e.g. 1, 3/2, 2, etc.)
    k : int
        Rank of the Stevens operator (2, 4, 6)
    q : int
        Component index (from -q to +q)
    return_latex : bool
        Whether to return a LaTeX string for display purposes

    Returns
    -------
    Oab : ndarray
        Matrix representation of the Stevens operator
    latex_str : str (optional)
        LaTeX representation

    Raises
    ------
    ValueError
        If k is not in {2, 4, 6} or if q is out of the valid range for the given k.
    """
    # Get angular momentum matrices
    Sx_, Sy_, Sz_ = calc_AngMomMatrices(S)
    I_matrix = np.eye(Sx_.shape[0])
    ss = S * (S + 1)

    # Define ladder operators
    Sp = Sx_ + 1j * Sy_
    Sm = Sx_ - 1j * Sy_

    e = None
    latex = "$\\rm invalid$"

    if k == 2:
        if abs(q) == 0:
            e = 3 * Sz_ @ Sz_ - ss * I_matrix
            # latex = "$3\\hat{S}_z^2$"
        elif abs(q) == 1:
            if q > 0:
                e = 0.25 * (Sz_ @ (Sp + Sm) + (Sp + Sm) @ Sz_)
                # latex = "$1/2 [\\hat{S}_x, \\hat{S}_z]_+$"
            else:
                e = -0.25j * (Sz_ @ (Sp - Sm) + (Sp - Sm) @ Sz_)
                # latex = "$1/2 [\\hat{S}_y, \\hat{S}_z]_+$"
        elif abs(q) == 2:
            if q > 0:
                e = 0.5 * (
                    np.linalg.matrix_power(Sp, 2)
                    + np.linalg.matrix_power(Sm, 2)
                )
                # latex = "$1/2 (\\hat{S}_+^2 + \\hat{S}_-^2)$"
            else:
                e = -0.5j * (
                    np.linalg.matrix_power(Sp, 2)
                    - np.linalg.matrix_power(Sm, 2)
                )
                # latex = "$-i/2 (\\hat{S}_+^2 - \\hat{S}_-^2)$"

    elif k == 4:
        z2 = Sz_ @ Sz_
        z3 = z2 @ Sz_
        z4 = z2 @ z2
        ss2 = ss ** 2
        if abs(q) == 0:
            e = (35 * z4 - (30 * ss * I_matrix - 25 * I_matrix)
                  * z2 + (3 * ss2 - 6 * ss) * I_matrix)
            # latex = "$35 S_z^4 - (30 S(S+1)-25) S_z^2$"
        elif abs(q) == 1:
            pm = Sp + Sm if q > 0 else Sp - Sm
            ppm = (7 * z3 - (3 * ss * I_matrix + 1 * I_matrix) * Sz_)
            c = 1 if q > 0 else -1j
            e = 0.25 * c * anticommutator(ppm, pm)
        elif abs(q) == 2:
            if q > 0:
                pm = Sp @ Sp + Sm @ Sm 
                c = 1
            else: 
                pm = Sp @ Sp - Sm @ Sm
                c = -1j
            e = (0.25 * c 
                 * anticommutator((7 * z2 - ss * I_matrix - 5 * I_matrix), pm))
        elif abs(q) == 3:
            pm = (Sp @ Sp @ Sp + Sm @ Sm @ Sm) if q > 0 else \
                 (Sp @ Sp @ Sp - Sm @ Sm @ Sm)
            c = 1 if q > 0 else -1j
            e = 0.25 * c * anticommutator(Sz_, pm)
        elif abs(q) == 4:
            if q > 0:
                e = 0.5 * (Sp @ Sp @ Sp @ Sp + Sm @ Sm @ Sm @ Sm)
            else:
                e = -0.5j * (Sp @ Sp @ Sp @ Sp - Sm @ Sm @ Sm @ Sm)

    elif k == 6:
        if abs(q) > 6:
            return None
        if q == 0:
            e = (
                231 * Sz_ @ Sz_ @ Sz_ @ Sz_ @ Sz_ @ Sz_
                - (315 * ss * I_matrix - 735) * Sz_ @ Sz_ @ Sz_ @ Sz_
                + (105 * ss ** 2 * I_matrix - 525 * ss + 294) * Sz_ @ Sz_
                - (5 * ss ** 3 - 40 * ss ** 2 + 60 * ss) * I_matrix
            )
        elif abs(q) == 1:
            pm = (Sp + Sm) 
            ppm = (
                33 * Sz_ @ Sz_ @ Sz_ @ Sz_ @ Sz_
                - 30 * ss * I_matrix * Sz_ @ Sz_ @ Sz_
                + 15 * Sz_ @ Sz_ @ Sz_
                + 5 * ss ** 2 * I_matrix * Sz_
                - 10 * ss * I_matrix * Sz_
                + 12 * Sz_
            )
            pm = pm if q > 0 else (Sp - Sm) 
            c = 1 if q > 0 else -1j
            e = 0.25 * c * anticommutator(pm, ppm)
        elif abs(q) == 2:
            pm = Sp @ Sp + Sm @ Sm 
            ppm = (
                33 * Sz_ @ Sz_ @ Sz_ @ Sz_
                - 18 * ss * I_matrix * Sz_ @ Sz_
                - 123 * Sz_ @ Sz_
                + ss ** 2 * I_matrix
                + 10 * ss * I_matrix
                + 102 * I_matrix
            )
            pm = pm if q > 0 else Sp @ Sp - Sm @ Sm 
            c = 1 if q > 0 else -1j
            e = 0.25 * c * anticommutator(pm, ppm)
        elif abs(q) == 3:
            pm = Sp @ Sp @ Sp + Sm @ Sm @ Sm
            ppm = (11 * Sz_ @ Sz_ @ Sz_ - (3 * ss * I_matrix + 59) * Sz_)
            pm = pm if q > 0 else Sp @ Sp @ Sp - Sm @ Sm @ Sm
            c = 1 if q > 0 else -1j
            e = 0.25 * c * anticommutator(pm, ppm)
        elif abs(q) == 4:
            ppm = (11 * Sz_ @ Sz_ - ss * I_matrix - 38 * I_matrix)
            if q > 0:
                pm = (Sp @ Sp @ Sp @ Sp + Sm @ Sm @ Sm @ Sm)
                c = 1
            else:
                pm = (Sp @ Sp @ Sp @ Sp - Sm @ Sm @ Sm @ Sm)
                c = -1j
            e = 0.25 * c * anticommutator(pm, ppm)
        elif abs(q) == 5:
            pm = Sp @ Sp @ Sp @ Sp @ Sp + Sm @ Sm @ Sm @ Sm @ Sm
            pm = pm if q > 0 else Sp @ Sp @ Sp @ Sp @ Sp - Sm @ Sm @ Sm @ Sm @ Sm
            c = 1 if q > 0 else -1j
            e = 0.25 * c * anticommutator(pm, Sz_)
        elif abs(q) == 6:
            pm = Sp @ Sp @ Sp @ Sp @ Sp @ Sp + Sm @ Sm @ Sm @ Sm @ Sm @ Sm
            pm = (
                pm
                if q > 0
                else Sp @ Sp @ Sp @ Sp @ Sp @ Sp
                - Sm @ Sm @ Sm @ Sm @ Sm @ Sm
            )
            c = 1 if q > 0 else -1j
            e = 0.5 * c * pm
        
    else:
        raise ValueError("Invalid a value. Allowed: 2, 4, 6")

    if return_latex:
        return e, latex
    return e


def set_Stevens(spinsys, i_spin, k, q, value):
    """
    Set a Stevens operator coefficient for a given spin.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object.
    i_spin : int
        Index of the spin in the spin system.
    k : int
        Degree of the Stevens operator (e.g., 2, 4, 6).
    q : int
        Order of the Stevens operator (-q ≤ k ≤ q).
    value : float or complex
        Coefficient for the Stevens operator.

    Raises
    ------
    IndexError
        If `i_spin` is out of range for the number of spins in the system.
    TypeError
        If `spinsys.Stevens[i_spin]` is not a dictionary.
    """
    spinsys.use_Stevens = True  # Ensure Stevens operators are used
    if i_spin >= spinsys.NSpins:
        raise IndexError(
            f"Spin index {i_spin} out of range for system with "
            f"{spinsys.NSpins} spins."
        )
    
    if not hasattr(spinsys, "Stevens") or spinsys.Stevens is None:
        # Initialize empty dictionaries if not already done
        spinsys.Stevens = [{} for _ in range(spinsys.NSpins)]

    if not isinstance(spinsys.Stevens[i_spin], dict):
        raise TypeError(f"spinsys.Stevens[{i_spin}] must be a dictionary.")

    spinsys.Stevens[i_spin][(k, q)] = value


# endregion StevensOperators

# region Visualization


def plot_EigenMatrix(spinsys: SpinSys, plot_type: str = 'p'):
    """
    Visualize the eigenstate matrix of the spin system as a heatmap.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing eigenvectors, basis states,
        and other relevant properties.
    plot_type : str, optional
        Type of plot to generate:
        - 'p': Probability (default), displays the probability matrix
          using the 'hot' colormap.
        - 'a': Amplitude, displays the amplitude matrix using the 'coolwarm' colormap.

    Returns
    -------
    return_dict : dict
    A dictionary containing:
    - 'fig': plotly.graph_objects.Figure
        The plotly figure object containing the ESR spectrum plot.
    - 'm': numpy.ndarray
        The matrix used for the heatmap visualization. Containing the
        projection values according to p/a.

    Raises
    ------
    ValueError
        If `plot_type` is not 'p' or 'a'.
        
    Notes
    -----
    - The function checks if the eigenvectors are already calculated; if
      not, it computes them.
    """
    # Check if eigVectors are calculated, if not calculate them. They
    # are needed for the projection matrix.
    if not hasattr(spinsys, 'states') or spinsys.eigVectors is None:
        calc_EigStates(spinsys)
            
    projection = np.zeros(
        [spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix],
        dtype=complex,
    )
    for i in range(spinsys.dimensionOfMatrix):
        for j in range(spinsys.dimensionOfMatrix):
            projection[i, j] = np.dot(
                spinsys.eigVectors[:, i].conj().T,
                np.identity(spinsys.dimensionOfMatrix)[:, j],
            )

    if plot_type == 'p':
        m = abs(np.conjugate(projection) * projection)
        name0, colormap = 'Probability', 'hot'
    elif plot_type == 'a':
        m = np.real(projection) + np.imag(projection)
        name0, colormap = 'Amplitude', 'coolwarm'
    else:
        raise ValueError(
            "Invalid plot_type. Use 'p' for probability or 'a' "
            "for amplitude."
        )

    # Generate the figure and axes
    # Build a Plotly heatmap with square cells (equal aspect ratio)
    x_vals = list(range(spinsys.dimensionOfMatrix))
    y_vals = list(range(spinsys.dimensionOfMatrix))

    # Use the same flipped matrix and labels as the matplotlib version
    z_vals = np.flipud(m).tolist()
    x_ticktext = spinsys.basisStates
    y_ticktext = spinsys.statesOnlyLHS[::-1]

    # Map matplotlib-like colormaps to plotly colorscales
    if colormap == 'hot':
        plotly_cmap = 'Hot'
    elif colormap == 'coolwarm':
        plotly_cmap = 'RdBu'
    else:
        plotly_cmap = colormap  # fallback to whatever plotly accepts

    zmin = 0 if plot_type == 'p' else -1
    zmax = 1 if plot_type == 'p' else 1

    fig = go.Figure(
        data=go.Heatmap(
        z=z_vals,
        x=x_vals,
        y=y_vals,
        colorscale=plotly_cmap,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(title=name0),
        hoverongaps=False
        )
    )
    # make sure the heatmap uses the matrix origin at the bottom-left
    # Use the explicit x/y arrays and tick labels, then flip the y-axis
    # so row 0 is at the bottom.
    fig.update_traces(x=x_vals, y=y_vals, selector=dict(type='heatmap'))
    fig.update_xaxes(tickmode='array', tickvals=x_vals, ticktext=x_ticktext)
    fig.update_yaxes(
        tickmode='array',
        tickvals=y_vals,
        ticktext=y_ticktext,
        autorange='reversed',
    )
    # Make cells square by anchoring y-axis scale to x-axis
    fig.update_layout(
        width=800,
        height=600,
        title=(
            f"System Eigenstates, B = "
            f"[{spinsys.B[0]}, {spinsys.B[1]}, {spinsys.B[2]}]"
        ),
        xaxis=dict(
        title='Basis States',
        tickmode='array',
        tickvals=x_vals,
        ticktext=x_ticktext,
        constrain='domain'
        ),
        yaxis=dict(
        title='Eigenstates',
        tickmode='array',
        tickvals=y_vals,
        ticktext=y_ticktext,
        scaleanchor="x",
        scaleratio=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    # ensure colorbar is shown with an outline and ticks outside
    fig.data[0].update(
        colorbar=dict(
            title=name0,
            outlinecolor='black',
            outlinewidth=1,
            ticks='outside',
        )
    )

    # add a black rectangle around the heatmap cells to serve as a border
    dim = spinsys.dimensionOfMatrix
    fig.update_layout(
        shapes=[
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=-0.5,
                x1=dim - 0.5,
                y0=-0.5,
                y1=dim - 0.5,
                line=dict(color="black", width=2),
                fillcolor="rgba(0,0,0,0)"
            )
        ]
    )
    fig.show()
    
    return_dict = {
        'fig': fig,
        'm': m
    }
    
    return return_dict
    

def plot_ZeemanDiagramm(
    spinsys: SpinSys,
    B_array: np.ndarray = None,
    energyscale: str = 'meV',
):
    """
    Plot the Zeeman diagram over a specified magnetic field range.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing all relevant parameters and
        methods, including the magnetic field vector and energy levels.
    B_array : numpy.ndarray, optional
        An array of magnetic field values to use for the sweep. If None,
        a default range from 0 to 1 Tesla will be used with 200 points.

    Returns
    -------
    return_dict : dict
        A dictionary containing:
        - 'fig': plotly.graph_objects.Figure
            The plotly figure object containing the ESR spectrum plot.
        - 'B_array': numpy.ndarray
            The array of magnetic field values used in the sweep.
        - 'E_array': numpy.ndarray
            The array of energy eigenvalues corresponding to each magnetic field value.
    
    Notes
    -----
    - The function restores the original field value afterwards.
    """
    # Save the initial B-Vector
    Binitial = spinsys.B

    # Default range from 0 to 1 Tesla with 200 points
    if B_array is None:
        B_array = np.linspace(0, 1, 200)  

    N = len(B_array)
    E = np.zeros((N, spinsys.dimensionOfMatrix))
    
    for i in range(N):
        spinsys.B = [0, 0, B_array[i]]
        calc_EigEnergies(spinsys)
        E[i, :] = spinsys.E_sort

    # Resotre the initial solution
    spinsys.B = Binitial
    calc_EigEnergies(spinsys)
        
    # Build a Plotly figure with white background
    fig = go.Figure()
    for i in range(spinsys.dimensionOfMatrix):
        y = E[:, i] * spinsys.meVtoGHzConversion if energyscale == 'GHz' else E[:, i]
        fig.add_trace(go.Scatter(x=B_array, y=y, mode='lines', name=f"E_{i}"))

    fig.update_layout(
        title='Zeeman Diagramm',
        xaxis_title='Magnetic Field (T)',
        yaxis_title=f'Energy ({energyscale})',
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=600,
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')
    fig.show()
    
    return_dict = {
        'fig': fig,
        'B_array': B_array,
        'E_array': E}

    return return_dict


def plot_EnergyVsS(spinsys: SpinSys, type: str = 'z', energyscale: str = 'meV'):
    """
    Plot the energy eigenvalues versus the expectation values of Sx, Sy, or Sz.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing Hamiltonian, eigenvectors, and energies.
    type : str, optional
        The spin operator to use for the expectation value ('x', 'y', or 'z'). 
        Default is 'z'.

    Returns
    -------
    return_dict : dict
        A dictionary containing:
        - 'fig': plotly.graph_objects.Figure
            The plotly figure object containing the ESR spectrum plot.
        - 'E_array': numpy.ndarray
            The array of energy eigenvalues.
        - 'S_elements': numpy.ndarray
            The array of expectation values of the selected spin operator.

    Raises
    ------
    ValueError
        If `type` is not 'x', 'y', or 'z', or if `energyscale` is not 'meV' or 'GHz'.

    Notes
    -----
    If the eigenvectors of the spin system are not already calculated,
    the function will compute them.
    """
    if not hasattr(spinsys, 'eigVectors') or spinsys.eigVectors is None:
        calc_EigEnergies(spinsys)

    if type == 'z':
        S_Operator = np.sum(spinsys.Sz, axis=0)
    elif type == 'x':
        S_Operator = np.sum(spinsys.Sx, axis=0)
    elif type == 'y':
        S_Operator = np.sum(spinsys.Sy, axis=0)

    # Calculate the Sz matrix elements for each state
    S_elements = np.zeros(spinsys.dimensionOfMatrix)
    for i in range(spinsys.dimensionOfMatrix):
        S_elements[i] = np.real(
            spinsys.eigVectors[:, i].T.conj()
            @ S_Operator
            @ spinsys.eigVectors[:, i]
        )

    # Prepare the energy values
    if energyscale == 'GHz':
        energies = spinsys.E_All_inGHz
    elif energyscale == 'meV':
        energies = spinsys.E_All
    else:
        raise ValueError("energyscale must be 'meV' or 'GHz'")

    # Create the figure and axes
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=S_elements,
        y=energies,
        mode='markers',
        marker=dict(symbol='line-ew', color='black', size=40, line=dict(width=5)),
        hovertemplate='S: %{x}<br>Energy: %{y}<extra></extra>'
    ))
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        title=f'Energy vs S_{type} Matrix Elements',
        xaxis=dict(title=f'm_{type}', showgrid=True, gridcolor='lightgray'),
        yaxis=dict(title=f'Energy ({energyscale})', 
                   showgrid=True, gridcolor='lightgray'),
        width=700,
        height=500
    )
    fig.show()

    return_dict = {
        'fig': fig,
        'E_array': energies,
        'S_elements': S_elements
    }
    
    return return_dict


def plot_All(spinsys: SpinSys, type: str = 'z', energyscale: str = 'meV'):
    """
    Plot all relevant information for the spin system.
     
    The function plot energy levels, Zeeman diagram, 
    and energy vs spin expectation values.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing Hamiltonian, eigenvectors, and energies.
    type : str, optional
        The spin operator to use for the expectation value ('x', 'y', or 'z'). 
        Default is 'z'.
    energyscale : str, optional
        The energy scale to use for the plots ('meV' or 'GHz'). Default is 'meV'.
    """
    # make them all plot next to each other
    plot_EigenMatrix(spinsys, plot_type='p')
    plot_EnergyVsS(spinsys, type=type, energyscale=energyscale)
    plot_ZeemanDiagramm(spinsys, energyscale=energyscale)

    return 

# endregion Visualization

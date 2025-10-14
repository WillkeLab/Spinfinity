def e_transport(SpinSys: SpinSys):
    """
    Calculate transformed spin transport operators between tip and sample.
    This function computes the transformed Pauli spin operators and the identity operator
    between the tip and sample spin density matrices, taking into account their eigenbases
    and occupation probabilities. The transformation includes a square root prefactor
    derived from the eigenvalues of the density matrices, as described in the referenced
    spin transport formalism.
    Parameters
    ----------
    SpinSys : SpinSys
        An object containing the tip and sample spin density matrices (`ptip` and `psample`).
    Returns
    -------
    list of ndarray
        A list containing four 2x2 complex numpy arrays:
        [x, y, z, u], where
            x : ndarray
                Transformed sigma_x operator.
            y : ndarray
                Transformed sigma_y operator.
            z : ndarray
                Transformed sigma_z operator.
            u : ndarray
                Transformed identity operator.
    Notes
    -----
    - The transformation is performed in the eigenbasis of the tip and sample density matrices.
    - The square root prefactor ensures correct weighting according to occupation probabilities.
    - The function assumes `dmatrix` returns a valid 2x2 density matrix for the given input.
    """
    # Define the Pauli matrices and identity operator
    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)

    # Density matrices
    rho_tip = dmatrix(SpinSys.ptip)
    rho_sample = dmatrix(SpinSys.psample)

    # Eigendecompose
    evals_tip, evecs_tip = np.linalg.eigh(rho_tip)
    evals_sample, evecs_sample = np.linalg.eigh(rho_sample)

    pt = np.real(evals_tip)     # 
    ps = np.real(evals_sample)

    pf_pi_sqrt = np.sqrt(np.outer(ps, pt))  # shape (2, 2)

    # Transform operators
    x = (evecs_sample.conj().T @ (2 * sigma_x) @ evecs_tip) * pf_pi_sqrt / 2
    y = (evecs_sample.conj().T @ (2 * sigma_y) @ evecs_tip) * pf_pi_sqrt / 2
    z = (evecs_sample.conj().T @ (2 * sigma_z) @ evecs_tip) * pf_pi_sqrt / 2
    u = 2 * (evecs_sample.conj().T @ identity @ evecs_tip) * pf_pi_sqrt

    return [x, y, z, u]



def fstep(x):
    """
    Temperature-broadened step function used in IETS simulations.
    
    Parameters
    ----------
    x : ndarray
        Normalized energy difference (e.g., (E_j - E_i - V) / (kT))

    Returns
    -------
    y : ndarray
        Broadened step function values (same shape as x)
    """
    x = np.asarray(x, dtype=float)

    # Clean small values near zero
    x[np.abs(x) < 1e-5] = 0

    z = np.exp(x)
    denom = (z - 1)**2
    y = (1 + (x - 1) * z) / denom

    # Set exact x == 0 case to 0.5
    y[x == 0] = 0.5

    # Clean up NaNs (due to overflows etc.)
    y[np.isnan(y)] = 0

    return y


def dmatrix(p):
    """
    Constructs a 2x2 density matrix from a polarization vector.
    Parameters
    ----------
    p : array_like
        A 3-element array or list representing the polarization vector (p_x, p_y, p_z).
    Returns
    -------
    rho : ndarray
        A 2x2 complex-valued density matrix corresponding to the given polarization vector.
    Dialogtxt
    ---------
    - User: What does this function do?
    - GitHub Copilot: It creates a 2x2 density matrix for a spin-1/2 system from a given polarization vector using the Pauli matrices.
    """
 
    p = np.array(p, dtype=float)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)

    rho = 0.5 * (identity + p[0]*sigma_x + p[1]*sigma_y + p[2]*sigma_z)
    return rho

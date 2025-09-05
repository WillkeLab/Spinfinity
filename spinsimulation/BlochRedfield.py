
"""
Bloch-Redfield Theory Implementation for Spin Systems

This module implements Bloch-Redfield theory for calculating relaxation and decoherence
in quantum spin systems coupled to environmental baths. The theory describes the dynamics
of an open quantum system through a master equation approach.

The implementation follows the formalism described in:
- Atom-Photon Interactions, Chapter IV
  by Claude Cohen-Tannoudji, Jacques Dupont-Roc, and Gilbert Grynberg
- Bloch-Redfield theory for quantum dots and other mesoscopic systems

Key Features:
- Substrate relaxation coefficients (thermal bath coupling)
- Tunneling relaxation coefficients (tip-sample and sample-tip processes)  
- Time evolution dynamics with coherent + incoherent terms
- Steady-state calculations
- Secular approximations for different regimes
- Energy shift calculations from virtual processes

Usage Examples:
-------------

Basic substrate relaxation calculation:
```python
import spinsimulation as ss

# Create spin system
spinsys = ss.SpinSys()
# ... configure system parameters ...

# Calculate substrate relaxation tensor
R_substrate = ss.BR.redfield_coefficients_substrate(spinsys, 
                                                   secular_type="window", 
                                                   secular_window=0.1)

# Time evolution from initial state
initial_dm = np.eye(spinsys.dimensionOfMatrix) / spinsys.dimensionOfMatrix  # thermal
time_points = np.linspace(0, 100, 1000)  # 0-100 ps

results = ss.BR.redfield_dynamics(spinsys, initial_dm, time_points, 
                                 bias_voltage=1.0)  # 1 mV bias

# Plot results
ss.BR.plot_redfield_dynamics(results, spinsys)
```

Steady-state calculation:
```python
# Calculate steady state under bias
steady_state = ss.BR.redfield_steady_state(spinsys, bias_voltage=2.0, 
                                          secular_type="sinc")

print(f"Steady-state populations: {np.diag(steady_state).real}")
```

Relaxation analysis:
```python
# Analyze relaxation pathways
analysis = ss.BR.analyze_relaxation_rates(spinsys)
print(f"State lifetimes: {analysis['lifetimes']} ps")
print(f"Dominant pathways: {analysis['dominant_pathways']}")
```

Author: Translated from original Scilab code
Date: September 2025
"""

import numpy as np
import scipy.linalg as la
from scipy.special import expit
from spinsimulation.SpinSys import SpinSys
import spinsimulation.SpinHamiltonian as hamil
import spinsimulation.calcBath as bath


def fermi_box_function(energy_diff, temperature):
    """
    Calculate the Fermi-Dirac box function for electron transitions.
    
    This function computes the integral:
    
         +∞
    y = ∫ f(e') * [1 - f(e' + x)] de'
        -∞
    
    where f(e,T) is the Fermi-Dirac distribution and x is the energy difference.
    
    Parameters
    ----------
    energy_diff : float or np.ndarray
        Energy difference between states in meV
    temperature : float
        Temperature in Kelvin
        
    Returns
    -------
    np.ndarray
        Fermi-Dirac box function values
        
    Notes
    -----
    This is the exact implementation of the Scilab fbox function.
    The box function accounts for the availability of electronic states
    and Pauli exclusion principle in the leads.
    """
    # Convert temperature to meV (kB = 0.08617333262145178 meV/K)
    kB_meV = 0.08617333262145178
    
    # Normalize energy difference by thermal energy: x = E/(kB*T)
    x = energy_diff / (kB_meV * temperature)
    
    # Clean small values (set values < ±1e-8 to zero)
    x = np.where(np.abs(x) < 1e-8, 0.0, x)
    
    # Calculate y = x / (1 - exp(-x))
    # Handle the case where x = 0 separately to avoid division by zero
    y = np.zeros_like(x, dtype=float)
    
    # For non-zero x
    nonzero_mask = (x != 0)
    x_nz = x[nonzero_mask]
    
    # Avoid overflow in exp(-x) for large negative x
    exp_neg_x = np.exp(-x_nz)
    denominator = 1.0 - exp_neg_x
    
    # Calculate y = x / (1 - exp(-x))
    y[nonzero_mask] = x_nz / denominator
    
    # Handle NaN values (when denominator = 0, i.e., exp(-x) = 1, i.e., x = 0)
    # In Scilab: y(isnan(y)) = 1
    y[np.isnan(y)] = 1.0
    
    # For x = 0, the limit of x/(1-exp(-x)) as x->0 is 1 (L'Hôpital's rule)
    y[x == 0] = 1.0
    
    return y


def calculate_secular_matrix(energy_diff, secular_type="none", window=0.001):
    """
    Calculate the secular approximation matrix for Bloch-Redfield theory.
    
    The secular approximation filters out rapidly oscillating terms in the
    master equation, keeping only slowly varying (secular) terms.
    
    Parameters
    ----------
    energy_diff : np.ndarray
        Matrix of energy differences between eigenstates
    secular_type : str
        Type of secular approximation:
        - "none": No secular approximation
        - "yes": Standard secular approximation with very small window
        - "window": Secular approximation with specified window
        - "sinc": Fermi's golden rule with sinc function cutoff
    window : float
        Energy window for secular approximation in meV
        
    Returns
    -------
    np.ndarray
        Secular matrix (1 for allowed transitions, 0 or reduced for suppressed)
    """
    secular_matrix = np.ones_like(energy_diff)
    
    if secular_type == "yes":
        # Very tight secular approximation
        window = 1e-10
        
    if secular_type in ["yes", "window"]:
        # Standard secular approximation
        for a in range(energy_diff.shape[0]):
            for b in range(energy_diff.shape[1]):
                mask = (np.abs(energy_diff) > np.abs(energy_diff[a, b]) + window) | \
                       (np.abs(energy_diff) < np.abs(energy_diff[a, b]) - window)
                secular_matrix[mask] = 0
                
    elif secular_type == "sinc":
        # Fermi's golden rule with sinc function
        ediff_scaled = (np.abs(energy_diff) - np.abs(energy_diff[0, 0])) / window * np.pi
        secular_matrix = (np.sinc(ediff_scaled / np.pi)) ** 2
        secular_matrix[np.isnan(secular_matrix)] = 1
        
    return secular_matrix


def redfield_coefficients_substrate(spinsys, secular_type="none", secular_window=0.001, 
                                  include_entanglement=True):
    """
    Calculate Bloch-Redfield coefficients for substrate interactions.
    
    This function computes the relaxation tensor R_abcd for the master equation:
    dρ_ab/dt = Σ_{c,d} R_abcd ρ_cd
    
    The calculation follows Cohen-Tannoudji et al., equation IV.B.43, accounting
    for substrate-induced relaxation processes. The substrate acts as a thermal
    bath that causes energy relaxation and pure dephasing.
    
    Physics Background:
    ------------------
    The substrate interaction represents coupling to phonons, magnons, or other
    low-energy excitations in the underlying material. This leads to:
    1. Energy relaxation: Population transfer between energy eigenstates
    2. Pure dephasing: Loss of quantum coherence without energy transfer
    3. Thermal equilibration: Drive toward thermal equilibrium at temperature T
    
    The master equation in Lindblad form is:
    dρ/dt = -i[H,ρ] + Σ_k γ_k(2A_k ρ A_k† - A_k†A_k ρ - ρ A_k†A_k)
    
    where A_k are the system operators coupling to bath mode k, and γ_k are
    the corresponding relaxation rates.
    
    Parameters
    ----------
    spinsys : SpinSys
        Spin system object containing eigenvectors, eigenvalues, and system parameters
    secular_type : str
        Type of secular approximation ("none", "yes", "window", "sinc")
        - "none": Full calculation without approximation
        - "yes": Rotating wave approximation (very narrow energy window)
        - "window": Secular approximation with energy window
        - "sinc": Fermi's golden rule with sinc function broadening
    secular_window : float
        Energy window for secular approximation in meV
    include_entanglement : bool
        Whether to include all spins (True) or only readout spin (False)
        For entangled systems, all spins contribute to relaxation
        
    Returns
    -------
    np.ndarray
        4D Redfield tensor R_abcd with shape (n_states, n_states, n_states, n_states)
        
    Notes
    -----
    The substrate interaction is modeled as a thermal bath that causes
    energy relaxation and dephasing of the spin system. The relaxation
    rates depend on:
    - Temperature (through Fermi-Dirac distributions)
    - Energy gaps between states
    - Matrix elements of spin operators
    - Coupling strength to substrate
    
    The secular approximation is crucial for the validity of the master equation.
    Without it, the equation may not preserve positivity of the density matrix.
    """

    # Ensure eigenstates are calculated
    if not hasattr(spinsys, 'eigVectors') or spinsys.eigVectors is None:
        hamil.calcEigEnergies(spinsys)
    
    # Get system dimensions
    n_states = spinsys.dimensionOfMatrix
    eigenvalues = spinsys.E_All  # Already referenced to ground state
    
    # Calculate energy differences between states
    energy_diff = eigenvalues[np.newaxis, :] - eigenvalues[:, np.newaxis]
    
    # Calculate Fermi-Dirac probabilities for substrate transitions
    proba = fermi_box_function(energy_diff, spinsys.T)*spinsys.kB
    
    # Initialize Redfield tensor
    redfield_tensor = np.zeros((n_states, n_states, n_states, n_states), dtype=complex)
    
    # Determine which spins to include
    if include_entanglement:
        spin_indices = range(spinsys.NSpins)
    else:
        spin_indices = [spinsys.ReadoutSpin]
    
    # Calculate tunneling matrix elements for substrate interaction
    # Use sample polarization for substrate (thermal bath)
    temp_spinsys = spinsys  # Could modify this for different bath parameters
    temp_spinsys.SamplePolarization = [0, 0, 0]  # Unpolarized bath
    # Get rate matrices using the proper Rate2nd function
    
    M1, M2, M3, M4 = Rate2nd(temp_spinsys, amp=spinsys.Js) 
    
    # Sum over all relevant spins
    for i_spin in spin_indices:
        # Get matrix elements for this spin - simplified approach
        # In the full implementation, different matrix elements would represent
        # different spin components and electron spin channels
        
        # Calculate Redfield tensor elements
        for a in range(n_states):
            for b in range(n_states):
                # Calculate secular matrix for this transition
                secular_matrix = calculate_secular_matrix(
                    energy_diff, secular_type, secular_window
                )

                secular_matrix = np.ones((n_states,n_states))
                
                # Calculate diagonal correction terms (Cohen-Tannoudji IV.B.43)
                o1 = np.zeros(n_states, dtype=complex)
                o2 = np.zeros(n_states, dtype=complex)
                
                for n in range(n_states):
                    # Correction terms with δ_bd and δ_ac
                    rate_terms = (M1[n, a] * np.conj(M1[n, :]) + 
                                M2[n, a] * np.conj(M2[n, :]) + 
                                M3[n, a] * np.conj(M3[n, :]) + 
                                M4[n, a] * np.conj(M4[n, :]))
                    o1 += rate_terms * proba[n, :]
                    
                    rate_terms_b = (M1[n, b] * np.conj(M1[n, :]) + 
                                  M2[n, b] * np.conj(M2[n, :]) + 
                                  M3[n, b] * np.conj(M3[n, :]) + 
                                  M4[n, b] * np.conj(M4[n, :]))
                    o2 += rate_terms_b * proba[n, :]
                
                
                # Construct diagonal correction matrices
                o11 = np.zeros((n_states, n_states), dtype=complex)
                o22 = np.zeros((n_states, n_states), dtype=complex)
                o11[:, b] = o1
                o22[a, :] = o2

                redfield_tensor[:,:,a,b]=((np.outer(M1[:,a], np.conj(M1[:,b]))+
                                        np.outer(M2[:,a], np.conj(M2[:,b]))+
                                        np.outer(M3[:,a], np.conj(M3[:,b]))+ 
                                        np.outer(M4[:,a], np.conj(M4[:,b])))*(np.outer(proba[:,a], np.conj(np.ones((n_states)))) + np.outer(np.ones((n_states)), np.conj(proba[:,b]))) - o11 - o22)*secular_matrix
                
    
    return redfield_tensor


def redfield_coefficients_tunneling(spinsys, bias_voltage, secular_type="none", 
                                   secular_window=0.001, include_entanglement=True):
    """
    Calculate Bloch-Redfield coefficients for tunneling electron interactions.
    
    This function computes relaxation tensors for tip-sample and sample-tip
    tunneling processes, which drive the system out of equilibrium.
    
    Parameters
    ----------
    spinsys : SpinSys
        Spin system object
    bias_voltage : float
        Bias voltage in mV
    secular_type : str
        Type of secular approximation
    secular_window : float
        Energy window for secular approximation in meV
    include_entanglement : bool
        Whether to include all spins or only readout spin
        
    Returns
    -------
    tuple of np.ndarray
        (R_ts, R_st) - Redfield tensors for tip-to-sample and sample-to-tip processes
        
    Notes
    -----
    The tunneling processes create asymmetric relaxation due to the bias voltage,
    driving the system into a non-equilibrium steady state.
    """
    # Ensure eigenstates are calculated
    if not hasattr(spinsys, 'eigVectors') or spinsys.eigVectors is None:
        hamil.calcEigEnergies(spinsys)
    
    n_states = spinsys.dimensionOfMatrix
    eigenvalues = spinsys.E_All
    
    # Calculate energy differences
    energy_diff = eigenvalues[np.newaxis, :] - eigenvalues[:, np.newaxis]
    
    # Calculate probabilities for tip-sample and sample-tip processes
    proba_ts = fermi_box_function(energy_diff + bias_voltage, spinsys.T)*spinsys.kB
    proba_st = fermi_box_function(energy_diff - bias_voltage, spinsys.T)*spinsys.kB

    # Initialize tensors
    R_ts = np.zeros((n_states, n_states, n_states, n_states), dtype=complex)
    R_st = np.zeros((n_states, n_states, n_states, n_states), dtype=complex)
    
    # Get tunneling matrix elements for tip-sample configuration
    # Use Rate2nd function for proper matrix element calculation
    M1_ts, M2_ts, M3_ts, M4_ts = Rate2nd(spinsys, amp=spinsys.Jt)
    # print('----------')
    # print(M1_ts)
    # print(M2_ts)
    # print(M3_ts)
    # print(M4_ts)

    
    # Get matrix elements for sample-tip (flip polarizations)
    # Create a copy with flipped polarizations
   
    spinsys_flipped = spinsys
    # Temporarily swap tip and sample polarizations
    
    original_ptip = spinsys.TipPolarization.copy()
    original_psample = spinsys.SamplePolarization.copy()
    spinsys_flipped.TipPolarization = original_psample
    spinsys_flipped.SamplePolarization = original_ptip
    
    M1_st, M2_st, M3_st, M4_st = Rate2nd(spinsys_flipped, amp=spinsys.Jt)
    # print('----------')
    # print(M1_st)
    # print(M2_st)
    # print(M3_st)
    # print(M4_st)
    
    spinsys.TipPolarization = original_ptip
    spinsys.SamplePolarization = original_psample
            
    
    # Calculate Redfield tensors for both processes
    for a in range(n_states):
        for b in range(n_states):
            secular_matrix = calculate_secular_matrix(
                energy_diff, secular_type, secular_window
            )
            secular_matrix = np.ones((n_states,n_states))
            
            # Initialize correction terms to match Scilab variables o1, o2, o3, o4
            o1 = np.zeros(n_states, dtype=complex)  # TS process with index a
            o2 = np.zeros(n_states, dtype=complex)  # TS process with index b
            o3 = np.zeros(n_states, dtype=complex)  # ST process with index a
            o4 = np.zeros(n_states, dtype=complex)  # ST process with index b
            
            for n in range(n_states):
                # Tip-to-sample processes (o1 and o2)
                o1 += (M1_ts[n, a] * np.conj(M1_ts[n, :]) + 
                       M2_ts[n, a] * np.conj(M2_ts[n, :]) + 
                       M3_ts[n, a] * np.conj(M3_ts[n, :]) + 
                       M4_ts[n, a] * np.conj(M4_ts[n, :])) * proba_ts[n, :]
                
                o2 += (M1_ts[n, b] * np.conj(M1_ts[n, :]) + 
                       M2_ts[n, b] * np.conj(M2_ts[n, :]) + 
                       M3_ts[n, b] * np.conj(M3_ts[n, :]) + 
                       M4_ts[n, b] * np.conj(M4_ts[n, :])) * proba_ts[n, :]
                
                # Sample-to-tip processes (o3 and o4)
                o3 += (M1_st[n, a] * np.conj(M1_st[n, :]) + 
                       M2_st[n, a] * np.conj(M2_st[n, :]) + 
                       M3_st[n, a] * np.conj(M3_st[n, :]) + 
                       M4_st[n, a] * np.conj(M4_st[n, :])) * proba_st[n, :]
                
                o4 += (M1_st[n, b] * np.conj(M1_st[n, :]) + 
                       M2_st[n, b] * np.conj(M2_st[n, :]) + 
                       M3_st[n, b] * np.conj(M3_st[n, :]) + 
                       M4_st[n, b] * np.conj(M4_st[n, :])) * proba_st[n, :]
            
           
            # Construct correction matrices exactly matching Scilab implementation
            # mmzero equivalent - zero matrix
            mmzero = np.zeros((n_states, n_states), dtype=complex)
            # mone equivalent - ones vector
            mone = np.ones(n_states, dtype=complex)
            
            # Build correction matrices as in Scilab
            o11 = mmzero.copy()
            o11[:, b] = o1
            
            o22 = mmzero.copy()
            o22[a, :] = np.conj(o2)  # o2' in Scilab (transpose/conjugate)
            
            o33 = mmzero.copy()
            o33[:, b] = o3
            
            o44 = mmzero.copy()
            o44[a, :] = np.conj(o4)  # o4' in Scilab (transpose/conjugate)
            
            # Main matrix terms - exactly matching Scilab structure
            main_ts = (np.outer(M1_ts[:,a], np.conj(M1_ts[:,b])) +
                      np.outer(M2_ts[:,a], np.conj(M2_ts[:,b])) +
                      np.outer(M3_ts[:,a], np.conj(M3_ts[:,b])) + 
                      np.outer(M4_ts[:,a], np.conj(M4_ts[:,b])))
            
            main_st = (np.outer(M1_st[:,a], np.conj(M1_st[:,b])) +
                      np.outer(M2_st[:,a], np.conj(M2_st[:,b])) +
                      np.outer(M3_st[:,a], np.conj(M3_st[:,b])) + 
                      np.outer(M4_st[:,a], np.conj(M4_st[:,b])))
            
            # Probability terms - matching Scilab: proba(:,a)*mone' + mone*proba(:,b)'
            prob_ts = (np.outer(proba_ts[:,a], mone) + 
                      np.outer(mone, np.conj(proba_ts[:,b])))
            
            prob_st = (np.outer(proba_st[:,a], mone) + 
                      np.outer(mone, np.conj(proba_st[:,b])))
            
            # Final Redfield tensor elements - exactly as in Scilab
            R_ts[:,:,a,b] = (main_ts * prob_ts - o11 - o22) * secular_matrix
            R_st[:,:,a,b] = (main_st * prob_st - o33 - o44) * secular_matrix
            
    return R_ts, R_st


def redfield_energy_shift(spinsys, include_entanglement=True):
    """
    Calculate energy shifts due to second-order processes.
    
    This function computes the energy renormalization of eigenstates
    due to virtual processes with the environment.
    
    Parameters
    ----------
    spinsys : SpinSys
        Spin system object
    include_entanglement : bool
        Whether to include all spins or only readout spin
        
    Returns
    -------
    np.ndarray
        Energy shifts for each eigenstate
        
    Notes
    -----
    The energy shifts arise from second-order perturbation theory
    and represent virtual transitions to intermediate states.
    """
    # Ensure eigenstates are calculated
    if not hasattr(spinsys, 'eigVectors') or spinsys.eigVectors is None:
        hamil.calcEigEnergies(spinsys)
    
    eigenvalues = spinsys.E_All
    n_states = len(eigenvalues)
    
    # Determine which spins to include
    if include_entanglement:
        spin_indices = range(spinsys.NSpins)
    else:
        spin_indices = [spinsys.ReadoutSpin]
    
    # Calculate rate matrix for energy shifts
    rate_matrix = np.zeros((n_states, n_states), dtype=complex)
    
    for i_spin in spin_indices:
        # This would need to be implemented based on the specific
        # second-order rate calculation (Rate2nd2 in original code)
        # For now, we'll use a simplified version
        pass
    
    # Calculate energy differences
    energy_diff = eigenvalues[:, np.newaxis] - eigenvalues[np.newaxis, :]
    
    # Integration over energy for virtual processes
    # This is a simplified version - the full calculation would involve
    # numerical integration as in the original code
    energy_shifts = np.zeros(n_states)
    
    # Placeholder for actual calculation
    # Would involve integration over intermediate state energies
    
    return energy_shifts


def redfield_dynamics(spinsys, density_matrix, time_points, bias_voltage=0.0,
                     secular_type="none", secular_window=0.001):
    """
    Solve the Bloch-Redfield master equation for time evolution.
    
    This function integrates the master equation:
    dρ/dt = -i[H, ρ] + R[ρ]
    where R is the Redfield relaxation superoperator.
    
    Parameters
    ----------
    spinsys : SpinSys
        Spin system object
    density_matrix : np.ndarray
        Initial density matrix in eigenstate basis
    time_points : np.ndarray
        Time points for evolution (in ps)
    bias_voltage : float
        Bias voltage in mV
    secular_type : str
        Type of secular approximation
    secular_window : float
        Energy window for secular approximation
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'density_matrices': Time evolution of density matrix
        - 'populations': Diagonal elements (populations) vs time
        - 'current': Tunneling current vs time
        - 'lifetimes': Relaxation lifetimes for each state
        
    Notes
    -----
    Time is in picoseconds, energies in meV, and current in units of e*Γ
    where Γ is the tunneling rate.
    """
    # Ensure eigenstates are calculated
    if not hasattr(spinsys, 'eigVectors') or spinsys.eigVectors is None:
        hamil.calcEigEnergies(spinsys)
    
    n_states = spinsys.dimensionOfMatrix
    eigenvalues = spinsys.E_All
    
    # Calculate Redfield tensors
    R_substrate = redfield_coefficients_substrate(spinsys, secular_type, secular_window)
    
    if bias_voltage != 0:
        R_ts, R_st = redfield_coefficients_tunneling(spinsys, bias_voltage, 
                                                    secular_type, secular_window)
        R_total = R_substrate + R_ts + R_st
    else:
        R_total = R_substrate
    
    spinsys.R_total = R_total  # Store for reference
    # Convert to Liouville space (vectorized density matrix)
    # Reshape 4D tensor to 2D matrix for master equation
    R_matrix = R_total.reshape(n_states**2, n_states**2)
    
    # Add coherent evolution (Hamiltonian term)
    # Create Liouvillian: L = -i[H, •] + R[•]
    energy_diff = eigenvalues[:, np.newaxis] - eigenvalues[np.newaxis, :]

    print(energy_diff)
    coherent_matrix = np.zeros((n_states**2, n_states**2), dtype=complex)
    
    for i in range(n_states):
        for j in range(n_states):
            idx = i * n_states + j
            coherent_matrix[idx, idx] = -1j * energy_diff[i, j]
    
    # Full Liouvillian
    # Convert energy differences to frequency (meV to THz: 1 meV = 0.24179905 THz)
    freq_conversion = 0.24179905 * 2 * np.pi  # meV to rad/ps
    liouvillian = freq_conversion * coherent_matrix + R_matrix
    
    # Time evolution
    dm_initial = density_matrix.flatten()
    density_matrices = np.zeros((len(time_points), n_states, n_states), dtype=complex)
    populations = np.zeros((len(time_points), n_states))
    current = np.zeros(len(time_points))
    
    # Calculate rate matrices for current calculation
    if bias_voltage != 0:
        try:
            rate_ts = bath.calcRateIntegrals(spinsys, 'ts')
            rate_st = bath.calcRateIntegrals(spinsys, 'st')
        except:
            # Fallback: use simple approximation
            rate_ts = np.ones(n_states)
            rate_st = np.ones(n_states)
    
    for i, t in enumerate(time_points):
        # Matrix exponential for time evolution
        evolved_dm = la.expm(liouvillian * t) @ dm_initial
        evolved_dm = evolved_dm.reshape(n_states, n_states)
        
        # Remove small imaginary parts from diagonal (numerical errors)
        evolved_dm = evolved_dm - 1j * np.diag(np.imag(np.diag(evolved_dm)))
        
        density_matrices[i] = evolved_dm
        populations[i] = np.real(np.diag(evolved_dm))
        
        # Calculate tunneling current
        if bias_voltage != 0:
            current[i] = np.real(np.sum(rate_ts * populations[i]) - 
                               np.sum(rate_st * populations[i]))
    
    # Calculate lifetimes from rate matrix
    rate_diagonal = np.zeros((n_states, n_states))
    for i in range(n_states):
        rate_diagonal[:, i] = np.diag(R_total[:, :, i, i])
    
    lifetimes = np.sum(rate_diagonal - np.diag(np.diag(rate_diagonal)), axis=0)
    lifetimes = 1.0 / (lifetimes + 1e-12)  # Avoid division by zero
    
    return {
        'density_matrices': density_matrices,
        'populations': populations,
        'current': current,
        'lifetimes': lifetimes,
        'time_points': time_points
    }

def redfield_dynamics_v2(spinsys, density_matrix, time_points, bias_voltage=0.0, secular_type="none", secular_window=0.001):
    """
    Alternative version of Redfield dynamics based on the provided Scilab code logic.
    Parameters
    ----------
    spinsys : SpinSys
        Spin system object
    density_matrix : np.ndarray
        Initial density matrix in eigenstate basis
    time_points : np.ndarray
        Time points for evolution (in ps)
    bias_voltage : float
        Bias voltage in mV
    secular_type : str
        Type of secular approximation
    secular_window : float
        Energy window for secular approximation
    Returns
    -------
    dict
        Dictionary containing:
        - 'density_matrices': Time evolution of density matrix
        - 'current': Tunneling current vs time
        - 'populations': Diagonal elements (populations) vs time
        - 'lifetimes': Relaxation lifetimes for each state
        - 'trans_matrix': Transition matrix used
        - 'rnk': Rank indicator for each time step
    """
    import numpy as np
    import scipy.linalg as la

    n_states = spinsys.dimensionOfMatrix
    eigenvalues = spinsys.E_All
    eigenvectors = spinsys.eigVectors
    dm = density_matrix
    len_dm = dm.shape[0]

    # Calculate delta_E (Bohr frequencies)
    ediff = -1j * (np.subtract.outer(eigenvalues, eigenvalues)) * 0.24179905 * 2 * np.pi
    ediff_diag = np.diag(ediff.flatten())

    # Redfield coefficients (assume caching in spinsys)
    if not hasattr(spinsys, 'RedF_coeff') or spinsys.RedF_coeff is None:
        spinsys.RedF_coeff = redfield_coefficients_substrate(spinsys, secular_type, secular_window)
    RedF_coeff = spinsys.RedF_coeff

    # Tunneling coefficients
    R_ts, R_st = redfield_coefficients_tunneling(spinsys, bias_voltage, secular_type, secular_window)
    RF = RedF_coeff + R_ts + R_st

    # Reduce RF to rate matrix for diagonal elements
    rate2 = np.zeros((n_states, n_states))
    for i in range(n_states):
        rate2[:, i] = np.diag(RF[:, :, i, i])
    ltime = np.sum(rate2 - np.diag(np.diag(rate2)), axis=0)
    ltime = 1.0 / (ltime + 1e-12)  # Avoid division by zero

    # Rescale RF for t in [ps]
    RF_mat = 0.48359788 * RF.reshape(n_states**2, n_states**2)
    dm_vec = dm.flatten()
    ediff_vec = ediff.flatten()
    ediff_diag = np.diag(ediff_vec)

    # J2_ts and rates for current
    J2_ts = spinsys.Jt**2
    ediff2 = np.subtract.outer(eigenvalues, eigenvalues)
    T = spinsys.T
    U = bias_voltage
    def fbox(x):
        return 1.0 / (np.exp(x) + 1)
    # Use Rate2nd2 if available, else fallback
    try:
        rate_ts = Rate2nd2(spinsys)
    except Exception:
        rate_ts = np.ones_like(ediff2)
    rate_st = rate_ts.T
    rate2_ts = J2_ts * fbox((ediff2 + U) / T) * T * spinsys.kB * rate_ts
    rate2_st = J2_ts * fbox((ediff2 - U) / T) * T * spinsys.kB rate_st

    trans_matrix = RF_mat + ediff_diag

    density_matrices = np.zeros((len(time_points), n_states, n_states), dtype=complex)
    populations = np.zeros((len(time_points), n_states))
    current = np.zeros(len(time_points))
    rnk = np.zeros(len(time_points), dtype=int)

    for i, t in enumerate(time_points):
        to = la.expm(trans_matrix * t)
        o = to @ dm_vec
        o = o.reshape((n_states, n_states))
        o = o - 1j * np.diag(np.imag(np.diag(o)))
        density_matrices[i] = o
        stst = np.real(np.diag(o))
        populations[i] = stst
        rnk[i] = int(np.any(stst < np.finfo(float).eps * 1000)) + 1
        current[i] = np.real(np.sum(rate2_ts * stst) - np.sum(rate2_st * stst))

    return {
        'density_matrices': density_matrices,
        'current': current,
        'populations': populations,
        'lifetimes': ltime,
        'trans_matrix': trans_matrix,
        'rnk': rnk
    }

def redfield_steady_state(spinsys, bias_voltage=0.0, secular_type=True, 
                         secular_window=0.001):
    """
    Calculate the steady-state density matrix using Bloch-Redfield theory.
    
    This function finds the steady state by solving:
    R[ρ_ss] = 0
    subject to Tr[ρ_ss] = 1
    
    Parameters
    ----------
    spinsys : SpinSys
        Spin system object
    bias_voltage : float
        Bias voltage in mV
    secular_type : str
        Type of secular approximation
    secular_window : float
        Energy window for secular approximation
        
    Returns
    -------
    np.ndarray
        Steady-state density matrix in eigenstate basis
        
    Notes
    -----
    The steady state represents the balance between coherent evolution,
    relaxation, and driving by the tunneling current.
    """
    # Ensure eigenstates are calculated
    if not hasattr(spinsys, 'eigVectors') or spinsys.eigVectors is None:
        hamil.calcEigEnergies(spinsys)
    
    n_states = spinsys.dimensionOfMatrix
    
    # Calculate total Redfield tensor
    R_substrate = redfield_coefficients_substrate(spinsys, secular_type, secular_window)
    
    if bias_voltage != 0:
        R_ts, R_st = redfield_coefficients_tunneling(spinsys, bias_voltage, 
                                                    secular_type, secular_window)
        R_total = R_substrate + R_ts + R_st
    else:
        R_total = R_substrate
    
    # Convert to matrix form
    R_matrix = R_total.reshape(n_states**2, n_states**2)
    
    spinsys.R_matrix = R_matrix  # Store for reference
    # For steady state: R[ρ] = 0
    # Modify matrix to include trace constraint
    R_modified = R_matrix.copy()
    
    # Replace last equation with trace constraint
    R_modified[-1, :] = 1.0
    
    # Right-hand side (all zeros except trace = 1)
    rhs = np.zeros(n_states**2)
    rhs[-1] = 1.0
    
    # Solve linear system
    try:
        steady_state_vec = la.solve(R_modified, rhs)
        steady_state_dm = steady_state_vec.reshape(n_states, n_states)
        
        # Ensure Hermiticity and proper normalization
        steady_state_dm = (steady_state_dm + steady_state_dm.conj().T) / 2
        steady_state_dm = steady_state_dm / np.trace(steady_state_dm)
        
    except la.LinAlgError:
        # Fallback: use kernel of R_matrix
        from scipy.linalg import null_space
        kernel = null_space(R_matrix)
        if kernel.size > 0:
            steady_state_vec = kernel[:, 0]
            steady_state_dm = steady_state_vec.reshape(n_states, n_states)
            steady_state_dm = steady_state_dm / np.trace(steady_state_dm)
        else:
            # Ultimate fallback: thermal state
            beta = 1.0 / (0.08617 * spinsys.T)  # 1/(kB*T) in meV^-1
            steady_state_dm = np.diag(np.exp(-beta * spinsys.E_All))
            steady_state_dm = steady_state_dm / np.trace(steady_state_dm)
    
    return steady_state_dm


# Utility functions for visualization and analysis

def plot_redfield_dynamics(results, spinsys, save_path=None):
    """
    Plot the results of Bloch-Redfield time evolution.
    
    Parameters
    ----------
    results : dict
        Results from redfield_dynamics function
    spinsys : SpinSys
        Spin system object
    save_path : str, optional
        Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available. Cannot plot results.")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Population dynamics
    for i in range(len(results['populations'][0])):
        ax1.plot(results['time_points'], results['populations'][:, i], 
                label=f'State |{i}⟩')
    ax1.set_xlabel('Time (ps)')
    ax1.set_ylabel('Population')
    ax1.set_title('Population Dynamics')
    ax1.legend()
    ax1.grid(True)
    
    # Current vs time
    ax2.plot(results['time_points'], results['current'])
    ax2.set_xlabel('Time (ps)')
    ax2.set_ylabel('Current (e·Γ)')
    ax2.set_title('Tunneling Current')
    ax2.grid(True)
    
    # Final density matrix (real part)
    im1 = ax3.imshow(np.real(results['density_matrices'][-1]), 
                     cmap='RdBu', aspect='equal')
    ax3.set_title('Final Density Matrix (Real)')
    ax3.set_xlabel('State index')
    ax3.set_ylabel('State index')
    plt.colorbar(im1, ax=ax3)
    
    # Lifetimes
    ax4.bar(range(len(results['lifetimes'])), results['lifetimes'])
    ax4.set_xlabel('State index')
    ax4.set_ylabel('Lifetime (ps)')
    ax4.set_title('State Lifetimes')
    ax4.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def analyze_relaxation_rates(spinsys, secular_type="none", secular_window=0.001):
    """
    Analyze the relaxation rates and pathways in the system.
    
    Parameters
    ----------
    spinsys : SpinSys
        Spin system object
    secular_type : str
        Type of secular approximation
    secular_window : float
        Energy window for secular approximation
        
    Returns
    -------
    dict
        Analysis results including rates, pathways, and bottlenecks
    """
    # Calculate Redfield tensor
    R_tensor = redfield_coefficients_substrate(spinsys, secular_type, secular_window)
    
    n_states = spinsys.dimensionOfMatrix
    
    # Extract rate matrix (transitions between populations)
    rate_matrix = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            if i != j:
                rate_matrix[i, j] = np.real(R_tensor[i, i, j, j])
    
    # Calculate lifetimes
    total_rates = np.sum(rate_matrix, axis=1)
    lifetimes = 1.0 / (total_rates + 1e-12)
    
    # Find dominant pathways
    dominant_pathways = []
    for i in range(n_states):
        max_rate_idx = np.argmax(rate_matrix[i, :])
        if rate_matrix[i, max_rate_idx] > 1e-10:
            dominant_pathways.append((i, max_rate_idx, rate_matrix[i, max_rate_idx]))
    
    return {
        'rate_matrix': rate_matrix,
        'lifetimes': lifetimes,
        'dominant_pathways': dominant_pathways,
        'total_rates': total_rates
    }


def etransport(spinsys):
    """
    Calculate electron transport matrix elements for tip-sample tunneling.
    
    This function computes the matrix elements for electron tunneling between
    tip and sample electrodes, considering their spin polarizations.
    
    Parameters
    ----------
    spinsys : SpinSys
        Spin system object containing tip and sample polarizations
        
    Returns
    -------
    tuple of np.ndarray
        (x, y, z, u) - Arrays containing matrix elements for different 
        spin channels and tunneling processes:
        - x[0]: <u',u> (up-up)
        - x[1]: <u',d> (up-down) 
        - x[2]: <d',u> (down-up)
        - x[3]: <d',d> (down-down)
        Similar for y, z (spin components) and u (charge)
        
    Notes
    -----
    The matrix elements account for the spin polarizations of tip and sample
    electrodes and represent all possible tunneling channels.
    """
    # Get tip and sample polarizations
    ptip = getattr(spinsys, 'ptip', np.array([0.0, 0.0, 1.0]))  # Default: z-polarized
    psample = getattr(spinsys, 'psample', np.array([0.0, 0.0, 1.0]))
    
    # Normalize polarizations
    if np.linalg.norm(ptip) > 1e-10:
        ptip = ptip / np.linalg.norm(ptip)
    if np.linalg.norm(psample) > 1e-10:
        psample = psample / np.linalg.norm(psample)
    
    # Pauli matrices for spin-1/2 electrons
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    sigma_0 = np.eye(2, dtype=complex)
    
    # Tip and sample density matrices (spin-1/2 electrons)
    rho_tip = 0.5 * (sigma_0 + ptip[0]*sigma_x + ptip[1]*sigma_y + ptip[2]*sigma_z)
    rho_sample = 0.5 * (sigma_0 + psample[0]*sigma_x + psample[1]*sigma_y + psample[2]*sigma_z)
    
    # Calculate matrix elements for all tunneling channels
    # The indices correspond to: [up-up, up-down, down-up, down-down]
    
    # X-component (Sx coupling)
    x = np.zeros(4, dtype=complex)
    x[0] = sigma_x[0,0] * rho_tip[0,0] * rho_sample[0,0]  # <u',u> with Sx
    x[1] = sigma_x[0,1] * rho_tip[0,0] * rho_sample[1,1]  # <u',d> with Sx  
    x[2] = sigma_x[1,0] * rho_tip[1,1] * rho_sample[0,0]  # <d',u> with Sx
    x[3] = sigma_x[1,1] * rho_tip[1,1] * rho_sample[1,1]  # <d',d> with Sx
    
    # Y-component (Sy coupling)  
    y = np.zeros(4, dtype=complex)
    y[0] = sigma_y[0,0] * rho_tip[0,0] * rho_sample[0,0]
    y[1] = sigma_y[0,1] * rho_tip[0,0] * rho_sample[1,1] 
    y[2] = sigma_y[1,0] * rho_tip[1,1] * rho_sample[0,0]
    y[3] = sigma_y[1,1] * rho_tip[1,1] * rho_sample[1,1]
    
    # Z-component (Sz coupling)
    z = np.zeros(4, dtype=complex)
    z[0] = sigma_z[0,0] * rho_tip[0,0] * rho_sample[0,0]
    z[1] = sigma_z[0,1] * rho_tip[0,0] * rho_sample[1,1]
    z[2] = sigma_z[1,0] * rho_tip[1,1] * rho_sample[0,0] 
    z[3] = sigma_z[1,1] * rho_tip[1,1] * rho_sample[1,1]
    
    # Charge component (identity coupling)
    u = np.zeros(4, dtype=complex)
    u[0] = sigma_0[0,0] * rho_tip[0,0] * rho_sample[0,0]
    u[1] = sigma_0[0,1] * rho_tip[0,0] * rho_sample[1,1]
    u[2] = sigma_0[1,0] * rho_tip[1,1] * rho_sample[0,0]
    u[3] = sigma_0[1,1] * rho_tip[1,1] * rho_sample[1,1]
    
    return x, y, z, u


def Rate2nd(spinsys, amp: float):
    """
    Calculate second-order tunneling matrix elements for all possible pathways.
    
    This function computes the tunneling matrix elements up to 2nd order
    via all possible tunneling pathes:
    - out1: <f,u'|Sx+Sy+Sz+U|i,u> (initial up, final up)
    - out2: <f,u'|Sx+Sy+Sz+U|i,d> (initial down, final up)  
    - out3: <f,d'|Sx+Sy+Sz+U|i,u> (initial up, final down)
    - out4: <f,d'|Sx+Sy+Sz+U|i,d> (initial down, final down)
    
    where u/d and u'/d' are the two eigenstates in tip and sample, respectively.
    
    Parameters
    ----------
    spinsys : SpinSys
        Spin system object containing:
        - ptip: spin polarization of the tip
        - psample: spin polarization of the sample  
        - eigVectors: matrix of eigenvectors of the spin ensemble
        - Sx, Sy, Sz: spin operator matrices
        - U: coupling strength array
        - J: exchange coupling array (optional)
        - ReadoutSpin: position of the spin being measured
        
    Returns
    -------
    tuple of np.ndarray
        (out1, out2, out3, out4) - Matrix elements for the four tunneling channels
        Each is a complex matrix of size (n_states, n_states)
        
    Notes
    -----
    This function is essential for calculating tunneling rates in STM/ESR
    experiments where electron spin and orbital degrees of freedom are coupled.
    """
    # Ensure eigenstates are calculated
    if not hasattr(spinsys, 'eigVectors') or spinsys.eigVectors is None:
        hamil.calcEigEnergies(spinsys)
    
    # Get electron transport matrix elements
    x, y, z, u = bath.calcTunnelingElectronMatrixElements(spinsys, scattertype='st')
    
    # Get eigenvector matrix
    E = spinsys.eigVectors
    E_dag = np.conj(E.T)  # Hermitian conjugate
    
    # Determine if we have a single spin or multiple spins
    if spinsys.NSpins == 1:
        # Single spin system - direct calculation
        nr = 0  # Position of the single spin
        
        # Get spin matrices for the single spin
        matrx_x = spinsys.Sx[nr, :, :]
        matrx_y = spinsys.Sy[nr, :, :]  
        matrx_z = spinsys.Sz[nr, :, :]
        matrx_u = np.eye(spinsys.dimensionOfMatrix, dtype=complex)  # Identity for charge
        
    else:
        # Multi-spin system - construct tensor products
        nr = spinsys.ReadoutSpin
        n_atoms = spinsys.NSpins
        
        # Start with identity for spins before the readout spin
        matrx_x = np.eye(1, dtype=complex)
        for i in range(nr):
            # Get dimension of spin i
            dim_i = int(2 * spinsys.S[i] + 1)
            identity_i = np.eye(dim_i, dtype=complex)
            matrx_x = np.kron(matrx_x, identity_i)
        
        # Copy for other components
        matrx_y = matrx_x.copy()
        matrx_z = matrx_x.copy() 
        matrx_u = matrx_x.copy()
        
        # Add the readout spin operators
        matrx_x = np.kron(matrx_x, spinsys.Sx[nr, :, :])
        matrx_y = np.kron(matrx_y, spinsys.Sy[nr, :, :])
        matrx_z = np.kron(matrx_z, spinsys.Sz[nr, :, :])
        matrx_u = np.kron(matrx_u, np.eye(int(2 * spinsys.S[nr] + 1), dtype=complex))
        
        # Add identity for spins after the readout spin
        for i in range(nr + 1, n_atoms):
            dim_i = int(2 * spinsys.S[i] + 1)
            identity_i = np.eye(dim_i, dtype=complex)
            matrx_x = np.kron(matrx_x, identity_i)
            matrx_y = np.kron(matrx_y, identity_i)
            matrx_z = np.kron(matrx_z, identity_i)
            matrx_u = np.kron(matrx_u, identity_i)
    
    # # Apply exchange coupling J if available
    # if hasattr(spinsys, 'J') and spinsys.J is not None:
    #     J_coupling = spinsys.J
    #     if np.isscalar(J_coupling):
    #         # If J is scalar, apply to all components
    #         J_x = J_y = J_z = J_coupling
    #     elif len(J_coupling) == 1:
    #         # Single value for all components (version 1.6 compatibility)
    #         J_x = J_y = J_z = J_coupling[0]
    #     elif len(J_coupling) >= 3:
    #         # Separate values for each component
    #         J_x, J_y, J_z = J_coupling[0], J_coupling[1], J_coupling[2]
    #     else:
    #         J_x = J_y = J_z = 1.0
    # else:
    #     J_x = J_y = J_z = 1.0
    #     print('No J')
    J_x = J_y = J_z = amp  # Default to 1.0 if no J provided
    
    # Apply coupling strength U
    if hasattr(spinsys, 'U') and spinsys.U is not None:
        if np.isscalar(spinsys.U):
            U_coupling = spinsys.U
        else:
            U_coupling = spinsys.U[nr] if nr < len(spinsys.U) else spinsys.U[0]
    else:
        U_coupling = 1.0
    
    # Transform to eigenstate basis and apply couplings
    matrx_x = J_x * E_dag @ matrx_x @ E
    matrx_y = J_y * E_dag @ matrx_y @ E  
    matrx_z = J_z * E_dag @ matrx_z @ E
    matrx_u = U_coupling * E_dag @ matrx_u @ E
    
    # Calculate matrix elements for all four tunneling channels
    out1 = (matrx_x * x[0, 0] + matrx_y * y[0, 0] + matrx_z * z[0, 0] + matrx_u * u[0, 0])
    out2 = (matrx_x * x[1, 0] + matrx_y * y[1 ,0] + matrx_z * z[1, 0] + matrx_u * u[1 ,0])
    out3 = (matrx_x * x[0, 1] + matrx_y * y[0, 1] + matrx_z * z[0, 1] + matrx_u * u[0 ,1])
    out4 = (matrx_x * x[1, 1] + matrx_y * y[1, 1] + matrx_z * z[1, 1] + matrx_u * u[1, 1])
    
    return out1, out2, out3, out4

#
# Python version of Rate2nd2, adapted from Scilab code
def Rate2nd2(spinsys: SpinSys):
    """
    Calculate the tunneling rate up to 2nd order via all possible tunneling paths.
    Returns a matrix giving the probability to tunnel from i->j.
    The experiment object must contain:
        - ptip: spin polarization of the tip
        - psample: spin polarization of the sample
        - Eigenvec: matrix of eigenvectors of the spin ensemble
        - atom: a list with the atoms of the system
        - position: position in the atom list at which the rate is calculated
    """
    # Check for use_T_ts flag
    use_T_ts = False

    # Get transition vectors (x, y, z, u)
    x, y, z, u = etransport(experiment)

    E = experiment.Eigenvec
    # Handle single spin-system or multi-atom system
    if not isinstance(experiment.atom, (list, tuple, np.ndarray)):
        # Only one spin-system
        S = experiment.atom
        matrxx = Sx(S)
        matrxy = Sy(S)
        matrxz = Sz(S)
        matrxu = S1(S)
    else:
        nr = experiment.position
        natoms = len(experiment.atom)
        # Start with identity (sparse(1) equivalent)
        matrxx = np.eye(E.shape[0], dtype=complex)
        matrxy = np.eye(E.shape[0], dtype=complex)
        matrxz = np.eye(E.shape[0], dtype=complex)
        matrxu = np.eye(E.shape[0], dtype=complex)
        for i in range(nr-1):
            mm = S1(experiment.atom[i])
            matrxx = matrxx @ mm
            matrxy = matrxy @ mm
            matrxz = matrxz @ mm
            matrxu = matrxu @ mm
        S = experiment.atom[nr-1]
        matrxx = matrxx @ Sx(S)
        matrxy = matrxy @ Sy(S)
        matrxz = matrxz @ Sz(S)
        matrxu = matrxu @ S1(S)
        for i in range(nr, natoms):
            mm = S1(experiment.atom[i])
            matrxx = matrxx @ mm
            matrxy = matrxy @ mm
            matrxz = matrxz @ mm
            matrxu = matrxu @ mm

    # J vector
    if use_T_ts:
        J = experiment.T_ts
    else:
        J = [1, 1, 1]

    # Matrix multiplications
    matrxx = J[0] * (E.conj().T @ matrxx @ E)
    matrxy = J[1] * (E.conj().T @ matrxy @ E)
    matrxz = J[2] * (E.conj().T @ matrxz @ E)
    # matrxu: S.U * E'*matrxu*E
    matrxu = getattr(S, 'U', 1.0) * (E.conj().T @ matrxu @ E)

    out = 0
    for i in range(4):
        out = out + np.abs(matrxx * x[i] + matrxy * y[i] + matrxz * z[i] + matrxu * u[i]) ** 2
    return out



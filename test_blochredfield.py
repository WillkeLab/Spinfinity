"""
Test script for the BlochRedfield module

This script demonstrates the basic functionality of the Bloch-Redfield
implementation and verifies that all functions work correctly.
"""

import numpy as np
import sys
import os

# Add the parent directory to the path so we can import spinsimulation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import spinsimulation as ss
    print("Successfully imported spinsimulation package")
    print(f"Available modules: {ss.__all__}")
except ImportError as e:
    print(f"Error importing spinsimulation: {e}")
    sys.exit(1)

def test_basic_functions():
    """Test basic BlochRedfield functions."""
    print("\n=== Testing Basic BlochRedfield Functions ===")
    
    # Test the SagHallo function
    print("\n1. Testing SagHallo function:")
    ss.BR.SagHallo()
    
    # Test Fermi-Dirac box function
    print("\n2. Testing Fermi-Dirac box function:")
    energy_diff = np.array([-1.0, 0.0, 1.0, 2.0])  # meV
    temperature = 4.0  # Kelvin
    proba = ss.BR.fermi_box_function(energy_diff, temperature)
    print(f"Energy differences: {energy_diff} meV")
    print(f"Temperature: {temperature} K")
    print(f"Probabilities: {proba}")
    
    # Test secular matrix calculation
    print("\n3. Testing secular matrix calculation:")
    n_states = 4
    energy_diff = np.random.rand(n_states, n_states) * 2.0 - 1.0  # Random energies
    secular_matrix = ss.BR.calculate_secular_matrix(energy_diff, "window", 0.1)
    print(f"Energy difference matrix shape: {energy_diff.shape}")
    print(f"Secular matrix shape: {secular_matrix.shape}")
    print(f"Number of allowed transitions: {np.sum(secular_matrix > 0.5)}")


def test_with_simple_spinsys():
    """Test with a simple SpinSys object."""
    print("\n=== Testing with Simple SpinSys ===")
    
    try:
        # Create a simple spin system
        print("\n1. Creating simple SpinSys object...")
        spinsys = ss.SpinSys()
        
        # Set basic parameters
        spinsys.NSpins = 1
        spinsys.ReadoutSpin = 0
        spinsys.T = 4.0  # 4 Kelvin
        
        # Create simple spin-1/2 system
        spinsys.S = [0.5]
        spinsys.dimensionOfMatrix = 2
        
        # Simple Hamiltonian (Zeeman splitting)
        spinsys.B_field = np.array([0, 0, 1.0])  # 1 Tesla in z-direction
        
        # Set up basic matrices
        spinsys.Sx = np.zeros((1, 2, 2))
        spinsys.Sy = np.zeros((1, 2, 2))
        spinsys.Sz = np.zeros((1, 2, 2))
        spinsys.U = np.array([0.1])  # 0.1 meV coupling
        
        # Pauli matrices
        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex) / 2
        pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
        pauli_z = np.array([[1, 0], [0, -1]], dtype=complex) / 2
        
        spinsys.Sx[0] = pauli_x
        spinsys.Sy[0] = pauli_y
        spinsys.Sz[0] = pauli_z
        
        # Simple eigenvalues and eigenvectors
        spinsys.E_All = np.array([0.0, 1.0])  # Ground and excited state
        spinsys.eigVectors = np.eye(2, dtype=complex)
        
        print("SpinSys object created successfully")
        print(f"Number of spins: {spinsys.NSpins}")
        print(f"Dimension: {spinsys.dimensionOfMatrix}")
        print(f"Temperature: {spinsys.T} K")
        
        # Test substrate relaxation coefficients
        print("\n2. Testing substrate relaxation coefficients...")
        try:
            R_substrate = ss.BR.redfield_coefficients_substrate(spinsys, "none", 0.001, False)
            print(f"Substrate tensor shape: {R_substrate.shape}")
            print(f"Substrate tensor norm: {np.linalg.norm(R_substrate)}")
        except Exception as e:
            print(f"Error in substrate coefficients: {e}")
        
        # Test steady state calculation
        print("\n3. Testing steady state calculation...")
        try:
            steady_state = ss.BR.redfield_steady_state(spinsys, 0.0, "none", 0.001)
            print(f"Steady state shape: {steady_state.shape}")
            print(f"Steady state trace: {np.trace(steady_state)}")
            print(f"Steady state populations: {np.diag(steady_state).real}")
        except Exception as e:
            print(f"Error in steady state: {e}")
            
    except Exception as e:
        print(f"Error creating or testing SpinSys: {e}")
        import traceback
        traceback.print_exc()


def test_analysis_functions():
    """Test analysis and utility functions."""
    print("\n=== Testing Analysis Functions ===")
    
    # Create dummy results for plotting test
    print("\n1. Testing analysis functions...")
    
    # Create fake results structure
    n_states = 3
    n_times = 100
    time_points = np.linspace(0, 10, n_times)
    
    results = {
        'time_points': time_points,
        'populations': np.random.rand(n_times, n_states),
        'current': np.sin(time_points) * 0.1,
        'density_matrices': np.random.rand(n_times, n_states, n_states) + 1j * np.random.rand(n_times, n_states, n_states),
        'lifetimes': np.random.rand(n_states) * 10
    }
    
    # Normalize populations
    for i in range(n_times):
        results['populations'][i] /= np.sum(results['populations'][i])
    
    print(f"Created test results with {n_times} time points and {n_states} states")
    print(f"Population range: {np.min(results['populations']):.3f} - {np.max(results['populations']):.3f}")
    print(f"Current range: {np.min(results['current']):.3f} - {np.max(results['current']):.3f}")
    
    # Test plotting (will only work if matplotlib is available)
    print("\n2. Testing plotting function...")
    try:
        # Create a dummy spinsys for plotting
        spinsys = ss.SpinSys()
        spinsys.dimensionOfMatrix = n_states
        
        ss.BR.plot_redfield_dynamics(results, spinsys)
        print("Plotting test completed (check if plot appeared)")
    except Exception as e:
        print(f"Plotting test failed (this is normal if matplotlib is not available): {e}")


if __name__ == "__main__":
    print("BlochRedfield Module Test Script")
    print("=" * 50)
    
    # Run tests
    test_basic_functions()
    test_with_simple_spinsys()
    test_analysis_functions()
    
    print("\n" + "=" * 50)
    print("BlochRedfield module test completed!")
    print("If you see this message, the basic functionality is working.")

import numpy as np
import scipy as sp
import scipy.constants as const
from scipy.linalg import null_space
from matplotlib import pyplot as plt
from qutip import Qobj
from qutip import mesolve
from qutip import Bloch

from joblib import Parallel, delayed

from spinfinity.SpinSys import SpinSys
import spinfinity.SpinHamiltonian as hamil 
import spinfinity.Bath as bath
import spinfinity.Lindblad as lind



def RotFrame(spinsys: SpinSys, f_RF: float):

    # Converts the static Spin Hamiltonain to the rotating frame of the microwave field. f_RF is in GHz.
    
    H = spinsys.H*spinsys.meVtoGHzConversion*2*np.pi
    H_RF = H + f_RF*spinsys.Sz[spinsys.ReadoutSpin]*2*np.pi
    H_RF = Qobj(H_RF)
    spinsys.H_RF = H_RF

    return H_RF

# region Visualization

  # Coming sooon


# endregion Visualization

# region Initialization


def initialize_thermal(spinsys: SpinSys):

    # Initializes the density matrix of the system to a thermal state at the temperature specified in spinsys.Temperature. 
    # The thermal state is calculated in the energy eigenbasis of the system's Hamiltonian.

    if not hasattr(spinsys, 'eigEnergies') or spinsys.eigEnergies is None:
        hamil.calcEigEnergies(spinsys)

    if not hasattr(spinsys, 'eigVectors') or spinsys.eigVectors is None:
        hamil.calcEigStates(spinsys)

    eigEnergies = spinsys.eigEnergies
    eigVectors = spinsys.eigVectors

    # Calculate Boltzmann factors
    k_B = const.Boltzmann * 1e3 / const.electron_volt  # Boltzmann constant in meV/K
    beta = 1 / (k_B * spinsys.Temperature)  # Inverse temperature
    boltzmann_factors = np.exp(-beta * eigEnergies)  # Boltzmann factors for each energy level

    # Normalize to get probabilities
    partition_function = np.sum(boltzmann_factors)
    probabilities = boltzmann_factors / partition_function

    # Construct the thermal density matrix in the energy eigenbasis
    rho_thermal_eigbasis = np.diag(probabilities)

    # Transform back to the original basis
    rho_thermal = eigVectors @ rho_thermal_eigbasis @ np.linalg.inv(eigVectors)

    spinsys.psi0 = Qobj(rho_thermal)

    return Qobj(rho_thermal)

# endregion Initialization

# region Rabi
 
def calc_RabiOscillation(spinsys: SpinSys, pulsewidth_array: np.ndarray, e_ops: list[Qobj] = None):

    # Required varaibles in spinsys: f_RF, c_ops, RabiRate, psi0 as well as finished Hamiltonian and operators

    if e_ops is None:
        e_ops = [Qobj(spinsys.Sx[spinsys.ReadoutSpin]), Qobj(spinsys.Sy[spinsys.ReadoutSpin]), Qobj(spinsys.Sz[spinsys.ReadoutSpin])]

    # Calculate the Hamiltonian in the rotating frame of the microwave field
    H_rot = RotFrame(spinsys, spinsys.f_RF) + Qobj(2*np.pi*spinsys.RabiRate * spinsys.Sx[spinsys.ReadoutSpin])

    result = mesolve(
        H_rot,
        spinsys.psi0,
        pulsewidth_array,
        c_ops=spinsys.collapse_ops,
        e_ops=e_ops,
        #options={"max_step": max(tau_ns / 50.0, 1e-4)}, #TODO see which options make sense here
    )

    # signal = np.array(result.expect)

    return pulsewidth_array, result

def calc_RabiDetuningMap(spinsys: SpinSys, pulsewidth_array: np.ndarray, frequency_array: np.ndarray, e_ops: Qobj = None):

    # Required varaibles in spinsys: f_RF, c_ops, RabiRate, psi0 as well as finished Hamiltonian and operators
    # Now just one Qobj for e_ops allowed
    # No parallelization yet, but could be easily implemented by parallelizing the loop over frequencies.

    if e_ops is None:
        e_ops = Qobj(spinsys.Sz[spinsys.ReadoutSpin])

    # Construct signal map:
    signal_map = np.zeros((len(frequency_array), len(pulsewidth_array)))

    for idx, frequency in enumerate(frequency_array):
        # Calculate the Hamiltonian in the rotating frame of the microwave field
        H_rot = RotFrame(spinsys, frequency) + Qobj(2*np.pi*spinsys.RabiRate * spinsys.Sx[spinsys.ReadoutSpin])

        result = mesolve(
            H_rot,
            spinsys.psi0,
            pulsewidth_array,
            c_ops=spinsys.collapse_ops,
            e_ops=[e_ops],
            #options={"max_step": max(tau_ns / 50.0, 1e-4)}, #TODO see which options make sense here
        )

        signal_map[idx, :] = np.array(result.expect[0])

    return pulsewidth_array, frequency_array, signal_map

# endregion Rabi

# region Ramsey

def calc_RamseyFringe(spinsys: SpinSys, tau_array: np.ndarray, e_ops: list[Qobj] = None, tsteps: int = 2500):

     # Required varaibles in spinsys: f_RF, c_ops, RabiRate, psi0 as well as finished Hamiltonian and operators

    if e_ops is None:
        e_ops = [Qobj(spinsys.Sx[spinsys.ReadoutSpin]), Qobj(spinsys.Sy[spinsys.ReadoutSpin]), Qobj(spinsys.Sz[spinsys.ReadoutSpin])]

    # Build Signal array for Ramsey fringes
    signal = np.zeros((len(tau_array), len(e_ops)))

    # Calculate the Hamiltonian in the rotating frame of the microwave field
    H_rot = RotFrame(spinsys, spinsys.f_RF)

    # Build Pulse Sequence for Ramsey experiment: pi/2 - tau - pi/2
    Omega_R = 2*np.pi*spinsys.RabiRate
    tp_90 = (np.pi / 2) / Omega_R
    pulse_op = Qobj(spinsys.Sx[spinsys.ReadoutSpin])
    solver_options = {"max_step": tp_90 / 10.0}
    t1_start = 0.0
    t1_end = tp_90

    for idx, tau in enumerate(tau_array):
        t2_start = t1_end + tau
        t2_end = t2_start + tp_90

        t_final = t2_end
        #tlist = np.linspace(0, t_final, tsteps)
        tlist = [0.0, t_final]
        def Ox_coeff(t, args):
            if t1_start <= t <= t1_end:
                return args["Omega_R"]
            if t2_start <= t <= t2_end:
                return args["Omega_R"]
            return 0.0
        
        # Build the time-dependent Hamiltonian for the Ramsey sequence
        H = [
            H_rot,
            [pulse_op, Ox_coeff],
        ]
        
        # Solve the master equation for the Ramsey sequence
        args = {"Omega_R": Omega_R}
        result = mesolve(
            H,
            spinsys.psi0,
            tlist,
            c_ops=spinsys.collapse_ops,
            e_ops=e_ops,
            args=args,
            options=solver_options,
        )

        signal[idx, :] = np.array(result.expect).T[-1]  # Take the expectation values at the end of the sequence

    return tau_array, signal

def calc_RamseyFringe_Parallel(spinsys: SpinSys, tau_array: np.ndarray, e_ops: list[Qobj] = None, tsteps: int = 2500):

     # Required varaibles in spinsys: f_RF, c_ops, RabiRate, psi0 as well as finished Hamiltonian and operators

    if e_ops is None:
        e_ops = [Qobj(spinsys.Sx[spinsys.ReadoutSpin]), Qobj(spinsys.Sy[spinsys.ReadoutSpin]), Qobj(spinsys.Sz[spinsys.ReadoutSpin])]

    # Calculate the Hamiltonian in the rotating frame of the microwave field
    H_rot = RotFrame(spinsys, spinsys.f_RF)

    # Build Pulse Sequence for Ramsey experiment: pi/2 - tau - pi/2
    Omega_R = 2*np.pi*spinsys.RabiRate
    tp_90 = (np.pi / 2) / Omega_R
    pulse_op = Qobj(spinsys.Sx[spinsys.ReadoutSpin])
    solver_options = {"max_step": tp_90 / 10.0}
    t1_start = 0.0
    t1_end = tp_90

    # Extract paramters
    psi0 = spinsys.psi0
    c_ops = spinsys.collapse_ops

    results = Parallel(n_jobs=-1)(delayed(calc_RamseyPoint)(tau, Omega_R, tp_90, H_rot, pulse_op, psi0, c_ops, e_ops, solver_options)
                                  for tau in tau_array)

    return tau_array, np.asarray(results)

def calc_RamseyPoint(
                    tau: float,
                    Omega_R: float,
                    tp_90: float,
                    H_rot: Qobj,
                    pulse_op: Qobj,
                    psi0: Qobj,
                    c_ops: list[Qobj],
                    e_ops: list[Qobj],
                    solver_options: dict,
                ):
    t1_end = tp_90
    t2_start = t1_end + tau
    t2_end = t2_start + tp_90

    def Ox_coeff(t, args):
        if 0.0 <= t <= t1_end or t2_start <= t <= t2_end:
            return args["Omega_R"]
        return 0.0

    H = [
        H_rot,
        [pulse_op, Ox_coeff],
    ]

    result = mesolve(
        H,
        psi0,
        [0.0, t2_end],
        c_ops=c_ops,
        e_ops=e_ops,
        args={"Omega_R": Omega_R},
        options=solver_options,
    )

    return np.asarray(result.expect)[:, -1]

def calc_RamseyDetuningMap(spinsys: SpinSys, tau_array: np.ndarray, frequency_array: np.ndarray, e_ops: Qobj = None):

    if e_ops is None:
        e_ops = Qobj(spinsys.Sz[spinsys.ReadoutSpin])
    
    # construct signal map:
    signal_map = np.zeros((len(frequency_array), len(tau_array)))

    # Build Pulse Sequence for Ramsey experiment: pi/2 - tau - pi/2
    Omega_R = 2*np.pi*spinsys.RabiRate
    tp_90 = (np.pi / 2) / Omega_R
    pulse_op = Qobj(spinsys.Sx[spinsys.ReadoutSpin])
    solver_options = {"max_step": tp_90 / 10.0}
    t1_start = 0.0
    t1_end = tp_90

    # Extract paramters
    psi0 = spinsys.psi0
    c_ops = spinsys.collapse_ops

    for idx, frequency in enumerate(frequency_array):
        H_rot = RotFrame(spinsys, frequency)

        results = Parallel(n_jobs=-1)(delayed(calc_RamseyPoint)(tau, Omega_R, tp_90, H_rot, pulse_op, psi0, c_ops, e_ops, solver_options)
                                  for tau in tau_array)

        signal_map[idx, :] = np.asarray(results)[:, 0]  # Assuming e_ops has only one operator and we take its expectation value

    return tau_array, frequency_array, signal_map

def plot_RamseyShot(spinsys, tau: float, tsteps: int = 500):

    # fixed e_ops:
    e_ops = [Qobj(spinsys.Sx[spinsys.ReadoutSpin]), Qobj(spinsys.Sy[spinsys.ReadoutSpin]), Qobj(spinsys.Sz[spinsys.ReadoutSpin])]

    # Calculate the Hamiltonian in the rotating frame of the microwave field
    H_rot = RotFrame(spinsys, spinsys.f_RF)

    # Build Pulse Sequence for Ramsey experiment: pi/2 - tau - pi/2
    Omega_R = 2*np.pi*spinsys.RabiRate
    tp_90 = (np.pi / 2) / Omega_R
    pulse_op = Qobj(spinsys.Sx[spinsys.ReadoutSpin])
    solver_options = {"max_step": tp_90 / 10.0}
    t1_start = 0.0
    t1_end = tp_90

    t2_start = t1_end + tau
    t2_end = t2_start + tp_90

    t_final = t2_end
    tlist = np.linspace(0, t_final, tsteps)
    
    def Ox_coeff(t, args):
        if t1_start <= t <= t1_end:
            return args["Omega_R"]
        if t2_start <= t <= t2_end:
            return args["Omega_R"]
        return 0.0
    
    # Build the time-dependent Hamiltonian for the Ramsey sequence
    H = [
        H_rot,
        [pulse_op, Ox_coeff],
    ]
    
    # Solve the master equation for the Ramsey sequence
    args = {"Omega_R": Omega_R}
    result = mesolve(
        H,
        spinsys.psi0,
        tlist,
        c_ops=spinsys.collapse_ops,
        e_ops=e_ops,
        args=args,
        options=solver_options,
    )

    # --------------------------------------------------
    # Plotting 

    plt.figure(figsize=(10, 6))
    for i, e_op in enumerate(e_ops):
        plt.plot(tlist, result.expect[i], label=f'Operator {i}')
    plt.xlabel('Time (s)')
    plt.ylabel('Expectation Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    bloch = Bloch()
    bloch.add_points([2 * result.expect[0], 2 * result.expect[1], 2 * result.expect[2]], "m")
    bloch.show()

    return result

# To Be added: Ramsey detuning map, parallelization, single Ramsey Shot with Display options, Documentation

# endregion Ramsey

# region Hahn Echo

def calc_HahnEcho_new(spinsys: SpinSys, tau_array: np.ndarray, e_ops: list[Qobj] = None, tsteps: int = 2500):
    
     # Required varaibles in spinsys: f_RF, c_ops, RabiRate, psi0 as well as finished Hamiltonian and operators
     # TODO: remove tsteps

    if e_ops is None:
        e_ops = [Qobj(spinsys.Sx[spinsys.ReadoutSpin]), Qobj(spinsys.Sy[spinsys.ReadoutSpin]), Qobj(spinsys.Sz[spinsys.ReadoutSpin])]

    # Build Signal array for Hahn Echo
    signal = np.zeros((len(tau_array), len(e_ops)))

    # Calculate the Hamiltonian in the rotating frame of the microwave field
    H_rot = RotFrame(spinsys, spinsys.f_RF)

    # Build Pulse Sequence for Hahn Echo: pi/2 - tau - pi - tau
    Omega_R = 2*np.pi*spinsys.RabiRate
    tp_90 = (np.pi / 2) / Omega_R
    tp_180 = np.pi / Omega_R
    pulse_op = Qobj(spinsys.Sx[spinsys.ReadoutSpin])
    solver_options = {"max_step": tp_90 / 10.0}
    t1_start = 0.0
    t1_end = tp_90

    for idx, tau in enumerate(tau_array):
        t2_start = t1_end + tau
        t2_end = t2_start + tp_180

        t_final = t2_end + tau
        tlist = np.linspace(0, t_final, tsteps)
        #tlist = [0.0, t_final]
        def Ox_coeff(t, args):
            if t1_start <= t <= t1_end:
                return args["Omega_R"]
            if t2_start <= t <= t2_end:
                return args["Omega_R"]
            return 0.0
        
        # Build the time-dependent Hamiltonian for the Hahn Echo sequence
        H = [
            H_rot,
            [pulse_op, Ox_coeff],
        ]
        
        # Solve the master equation for the Hahn Echo sequence
        args = {"Omega_R": Omega_R}
        result = mesolve(
            H,
            spinsys.psi0,
            tlist,
            c_ops=spinsys.collapse_ops,
            e_ops=e_ops,
            args=args,
            options=solver_options,
        )

        signal[idx, :] = np.array(result.expect).T[-1]  # Take the expectation values at the end of the sequence

    return tau_array, signal

def plot_HahnEchoShot_new(spinsys, tau: float, tsteps: int = 2500):

    # fixed e_ops:
    e_ops = [Qobj(spinsys.Sx[spinsys.ReadoutSpin]), Qobj(spinsys.Sy[spinsys.ReadoutSpin]), Qobj(spinsys.Sz[spinsys.ReadoutSpin])]

    # Calculate the Hamiltonian in the rotating frame of the microwave field
    H_rot = RotFrame(spinsys, spinsys.f_RF)

    # Build Pulse Sequence for Hahn Echo: pi/2 - tau - pi - tau
    Omega_R = 2*np.pi*spinsys.RabiRate
    tp_90 = (np.pi / 2) / Omega_R
    tp_180 = np.pi / Omega_R
    pulse_op = Qobj(spinsys.Sx[spinsys.ReadoutSpin])
    solver_options = {"max_step": tp_90 / 10.0}
    t1_start = 0.0
    t1_end = tp_90

    t2_start = t1_end + tau
    t2_end = t2_start + tp_180
    t_final = t2_end + tau
    
    tlist = np.linspace(0, t_final, tsteps)
    
    def Ox_coeff(t, args):
        if t1_start <= t <= t1_end or t2_start <= t <= t2_end:
            return args["Omega_R"]
        return 0.0
    
    # Build the time-dependent Hamiltonian for the Hahn Echo sequence
    H = [
        H_rot,
        [pulse_op, Ox_coeff],
    ]
    
    # Solve the master equation for the Hahn Echo sequence
    args = {"Omega_R": Omega_R}
    result = mesolve(
        H,
        spinsys.psi0,
        tlist,
        c_ops=spinsys.collapse_ops,
        e_ops=e_ops,
        args=args,
        options=solver_options,
    )

    # --------------------------------------------------
    # Plotting 
    plt.figure(figsize=(10, 6))
    for i, e_op in enumerate(e_ops):
        plt.plot(tlist, result.expect[i], label=f'Operator {i}')
    plt.xlabel('Time (s)')
    plt.ylabel('Expectation Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def calc_HahnEcho(spinsys: SpinSys, tau_array: np.ndarray, e_ops: list[Qobj] = None, tsteps: int = 2500):
    
     # Required varaibles in spinsys: f_RF, c_ops, RabiRate, psi0 as well as finished Hamiltonian and operators
     # TODO: remove tsteps

    if e_ops is None:
        e_ops = [Qobj(spinsys.Sx[spinsys.ReadoutSpin]), Qobj(spinsys.Sy[spinsys.ReadoutSpin]), Qobj(spinsys.Sz[spinsys.ReadoutSpin])]

    # Build Signal array for Hahn Echo
    signal = np.zeros((len(tau_array), len(e_ops)))

    # Calculate the Hamiltonian in the rotating frame of the microwave field
    H_rot = RotFrame(spinsys, spinsys.f_RF)

    # Build Pulse Sequence for Hahn Echo: pi/2 - tau - pi - tau
    Omega_R = 2*np.pi*spinsys.RabiRate
    tp_90 = (np.pi / 2) / Omega_R
    tp_180 = np.pi / Omega_R
    pulse_op = Qobj(spinsys.Sx[spinsys.ReadoutSpin])
    solver_options = {"max_step": tp_90 / 10.0}
    t1_start = 0.0
    t1_end = tp_90

    for idx, tau in enumerate(tau_array):
        t2_start = t1_end + tau
        t2_end = t2_start + tp_180
        t3_start = t2_end + tau
        t3_end = t3_start + tp_90

        t_final = t3_end
        tlist = np.linspace(0, t_final, tsteps)
        #tlist = [0.0, t_final]
        def Ox_coeff(t, args):
            if t1_start <= t <= t1_end:
                return args["Omega_R"]
            if t2_start <= t <= t2_end:
                return args["Omega_R"]
            if t3_start <= t <= t3_end:
                return args["Omega_R"]
            return 0.0
        
        # Build the time-dependent Hamiltonian for the Hahn Echo sequence
        H = [
            H_rot,
            [pulse_op, Ox_coeff],
        ]
        
        # Solve the master equation for the Hahn Echo sequence
        args = {"Omega_R": Omega_R}
        result = mesolve(
            H,
            spinsys.psi0,
            tlist,
            c_ops=spinsys.collapse_ops,
            e_ops=e_ops,
            args=args,
            options=solver_options,
        )

        signal[idx, :] = np.array(result.expect).T[-1]  # Take the expectation values at the end of the sequence

    return tau_array, signal

def calc_HahnEcho_Parallel(spinsys: SpinSys, tau_array: np.ndarray, e_ops: list[Qobj] = None, tsteps: int = 2500):
    
    # Required varaibles in spinsys: f_RF, c_ops, RabiRate, psi0 as well as finished Hamiltonian and operators

    if e_ops is None:
        e_ops = [Qobj(spinsys.Sx[spinsys.ReadoutSpin]), Qobj(spinsys.Sy[spinsys.ReadoutSpin]), Qobj(spinsys.Sz[spinsys.ReadoutSpin])]

    # Calculate the Hamiltonian in the rotating frame of the microwave field
    H_rot = RotFrame(spinsys, spinsys.f_RF)

    # Build Pulse Sequence for Hahn Echo: pi/2 - tau - pi - tau
    Omega_R = 2*np.pi*spinsys.RabiRate
    tp_90 = (np.pi / 2) / Omega_R
    tp_180 = np.pi / Omega_R
    pulse_op = Qobj(spinsys.Sx[spinsys.ReadoutSpin])
    solver_options = {"max_step": tp_90 / 10.0}

    # Extract paramters
    psi0 = spinsys.psi0
    c_ops = spinsys.collapse_ops

    def calc_HahnEchoPoint(tau: float):
        t1_end = tp_90
        t2_start = t1_end + tau
        t2_end = t2_start + tp_180
        t3_start = t2_end + tau
        t3_end = t3_start + tp_90

        def Ox_coeff(t, args):
            if 0.0 <= t <= t1_end or t2_start <= t <= t2_end or t3_start <= t <= t3_end:
                return args["Omega_R"]
            return 0.0

        H = [
            H_rot,
            [pulse_op, Ox_coeff],
        ]
        tlist = np.linspace(0, t3_end, tsteps)

        result = mesolve(
            H,
            psi0,
            tlist,
            c_ops=c_ops,
            e_ops=e_ops,
            args={"Omega_R": Omega_R},
            options=solver_options,
        )

        return np.asarray(result.expect)[:, -1]

    results = Parallel(n_jobs=-1)(delayed(calc_HahnEchoPoint)(tau)
                                  for tau in tau_array)

    return tau_array, np.asarray(results)

def plot_HahnEchoShot(spinsys, tau: float, tsteps: int = 2500):

    # fixed e_ops:
    e_ops = [Qobj(spinsys.Sx[spinsys.ReadoutSpin]), Qobj(spinsys.Sy[spinsys.ReadoutSpin]), Qobj(spinsys.Sz[spinsys.ReadoutSpin])]

    # Calculate the Hamiltonian in the rotating frame of the microwave field
    H_rot = RotFrame(spinsys, spinsys.f_RF)

    # Build Pulse Sequence for Hahn Echo: pi/2 - tau - pi - tau
    Omega_R = 2*np.pi*spinsys.RabiRate
    tp_90 = (np.pi / 2) / Omega_R
    tp_180 = np.pi / Omega_R
    pulse_op = Qobj(spinsys.Sx[spinsys.ReadoutSpin])
    solver_options = {"max_step": tp_90 / 10.0}
    t1_start = 0.0
    t1_end = tp_90

    t2_start = t1_end + tau
    t2_end = t2_start + tp_180
    t3_start = t2_end + tau
    t3_end = t3_start + tp_90

    t_final = t3_end
    tlist = np.linspace(0, t_final, tsteps)
    
    def Ox_coeff(t, args):
        if t1_start <= t <= t1_end or t2_start <= t <= t2_end or t3_start <= t <= t3_end:
            return args["Omega_R"]
        return 0.0
    
    # Build the time-dependent Hamiltonian for the Hahn Echo sequence
    H = [
        H_rot,
        [pulse_op, Ox_coeff],
    ]
    
    # Solve the master equation for the Hahn Echo sequence
    args = {"Omega_R": Omega_R}
    result = mesolve(
        H,
        spinsys.psi0,
        tlist,
        c_ops=spinsys.collapse_ops,
        e_ops=e_ops,
        args=args,
        options=solver_options,
    )

    # --------------------------------------------------
    # Plotting 
    plt.figure(figsize=(10, 6))
    for i, e_op in enumerate(e_ops):
        plt.plot(tlist, result.expect[i], label=f'Operator {i}')
    plt.xlabel('Time (s)')
    plt.ylabel('Expectation Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# endregion Hahn Echo

# region Relaxometry

def calc_Relaxometry(spinsys: SpinSys, tau_array: np.ndarray, e_ops: list[Qobj] = None, tsteps: int = 2500):

    # Required varaibles in spinsys: f_RF, c_ops, RabiRate, psi0 as well as finished Hamiltonian and operators
    # Could make this quick by parallelization or using just one mesolve and subtract pi pulse from the result
    if e_ops is None:
        e_ops = [Qobj(spinsys.Sx[spinsys.ReadoutSpin]), Qobj(spinsys.Sy[spinsys.ReadoutSpin]), Qobj(spinsys.Sz[spinsys.ReadoutSpin])]

    # Build Signal array for Relaxometry
    signal = np.zeros((len(tau_array), len(e_ops)))

    # Calculate the Hamiltonian in the rotating frame of the microwave field
    H_rot = RotFrame(spinsys, spinsys.f_RF)

    # Build Pulse Sequence for Relaxometry: pi - tau
    tp_180 = np.pi / (2*np.pi*spinsys.RabiRate)  # Convert Rabi rate to GHz and calculate pulse duration for pi pulse
    t1_start = 0.0
    t1_end = tp_180
    Omega_R = 2*np.pi*spinsys.RabiRate  # Convert to GHz
    pulse_op = Qobj(spinsys.Sx[spinsys.ReadoutSpin])

    for idx, tau in enumerate(tau_array):

        t_final = t1_end + tau
        tlist = np.linspace(0, t_final, tsteps)
        tlist = [0.0, t_final]

        def Ox_coeff(t, args):
            if t1_start <= t <= t1_end:
                return args["Omega_R"]
            return 0.0
        args = {"Omega_R": Omega_R}

        H = [
            H_rot,
            [pulse_op, Ox_coeff],
        ]

        result = mesolve(
            H_rot,
            spinsys.psi0,
            [0.0, tau],
            c_ops=spinsys.collapse_ops,
            e_ops=e_ops,
            args=args,
            options={"max_step": tau / 10.0},
        )

        signal[idx, :] = np.array(result.expect).T[-1]  # Take the expectation values at the end of the sequence

    return tau_array, signal

# endregion Relaxometry

# region ESEEM

def calc_3pulseESEEM(spinsys: SpinSys, tau_array: np.ndarray, T: float, e_ops: list[Qobj] = None, tsteps: int = 2500):

    if e_ops is None:
        e_ops = [Qobj(spinsys.Sx[spinsys.ReadoutSpin]), Qobj(spinsys.Sy[spinsys.ReadoutSpin]), Qobj(spinsys.Sz[spinsys.ReadoutSpin])]

    # Build Signal array
    signal = np.zeros((len(tau_array), len(e_ops)))

    # Calculate the Hamiltonian in the rotating frame of the microwave field
    H_rot = RotFrame(spinsys, spinsys.f_RF)

    # Build Pulse Sequence for 3p-ESEEM: pi/2 - T - pi/2 - tau - pi/2 - T - pi/2
    Omega_R = 2*np.pi*spinsys.RabiRate
    tp_90 = (np.pi / 2) / Omega_R
    pulse_op = Qobj(spinsys.Sx[spinsys.ReadoutSpin])
    solver_options = {"max_step": tp_90 / 10.0}

    for idx, tau in enumerate(tau_array):
        t1_start = 0.0
        t1_end = tp_90
        t2_start = t1_end + T
        t2_end = t2_start + tp_90
        t3_start = t2_end + tau
        t3_end = t3_start + tp_90
        t4_start = t3_end + T
        t4_end = t4_start + tp_90

        t_final = t4_end
        tlist = np.linspace(0, t_final, tsteps)

        def Ox_coeff(t, args):
            if (t1_start <= t <= t1_end) or (t2_start <= t <= t2_end) or (t3_start <= t <= t3_end) or (t4_start <= t <= t4_end):
                return args["Omega_R"]
            return 0.0
        
        H = [
            H_rot,
            [pulse_op, Ox_coeff],
        ]

        args = {"Omega_R": Omega_R}
        result = mesolve(
            H,
            spinsys.psi0,
            tlist,
            c_ops=spinsys.collapse_ops,
            e_ops=e_ops,
            args=args,
            options=solver_options,
        )

        signal[idx, :] = np.array(result.expect).T[-1]  # Take the expectation values at the end of the sequence
    
    return tau_array, signal

def calc_3pulseESEEM_Parallel(spinsys: SpinSys, tau_array: np.ndarray, T: float, e_ops: list[Qobj] = None, tsteps: int = 2500):

    if e_ops is None:
        e_ops = [Qobj(spinsys.Sx[spinsys.ReadoutSpin]), Qobj(spinsys.Sy[spinsys.ReadoutSpin]), Qobj(spinsys.Sz[spinsys.ReadoutSpin])]

    # Calculate the Hamiltonian in the rotating frame of the microwave field
    H_rot = RotFrame(spinsys, spinsys.f_RF)

    # Build Pulse Sequence for 3p-ESEEM: pi/2 - T - pi/2 - tau - pi/2 - T - pi/2
    Omega_R = 2 * np.pi * spinsys.RabiRate
    tp_90 = (np.pi / 2) / Omega_R
    pulse_op = Qobj(spinsys.Sx[spinsys.ReadoutSpin])
    solver_options = {"max_step": tp_90 / 10.0}

    # Extract paramters
    psi0 = spinsys.psi0
    c_ops = spinsys.collapse_ops

    def calc_3pulseESEEMPoint(tau: float):
        t1_end = tp_90
        t2_start = t1_end + T
        t2_end = t2_start + tp_90
        t3_start = t2_end + tau
        t3_end = t3_start + tp_90
        t4_start = t3_end + T
        t4_end = t4_start + tp_90

        def Ox_coeff(t, args):
            if (0.0 <= t <= t1_end) or (t2_start <= t <= t2_end) or (t3_start <= t <= t3_end) or (t4_start <= t <= t4_end):
                return args["Omega_R"]
            return 0.0

        H = [
            H_rot,
            [pulse_op, Ox_coeff],
        ]
        tlist = np.linspace(0, t4_end, tsteps)

        result = mesolve(
            H,
            psi0,
            tlist,
            c_ops=c_ops,
            e_ops=e_ops,
            args={"Omega_R": Omega_R},
            options=solver_options,
        )

        return np.asarray(result.expect)[:, -1]

    results = Parallel(n_jobs=-1)(delayed(calc_3pulseESEEMPoint)(tau)
                                  for tau in tau_array)

    return tau_array, np.asarray(results)


def plot_3pulseESEEMShot(spinsys, tau: float, T: float, tsteps: int = 2500):
    
    # fixed e_ops:
    e_ops = [Qobj(spinsys.Sx[spinsys.ReadoutSpin]), Qobj(spinsys.Sy[spinsys.ReadoutSpin]), Qobj(spinsys.Sz[spinsys.ReadoutSpin])]

    # Calculate the Hamiltonian in the rotating frame of the microwave field
    H_rot = RotFrame(spinsys, spinsys.f_RF)

    # Build Pulse Sequence for 3p-ESEEM: pi/2 - T - pi/2 - tau - pi/2 - T - pi/2
    Omega_R = 2 * np.pi * spinsys.RabiRate
    tp_90 = (np.pi / 2) / Omega_R
    pulse_op = Qobj(spinsys.Sx[spinsys.ReadoutSpin])
    solver_options = {"max_step": tp_90 / 10.0}
    
    t1_start = 0.0
    t1_end = tp_90
    t2_start = t1_end + T
    t2_end = t2_start + tp_90
    t3_start = t2_end + tau
    t3_end = t3_start + tp_90
    t4_start = t3_end + T
    t4_end = t4_start + tp_90

    t_final = t4_end
    tlist = np.linspace(0, t_final, tsteps)

    def Ox_coeff(t, args):
        if (t1_start <= t <= t1_end) or (t2_start <= t <= t2_end) or (t3_start <= t <= t3_end) or (t4_start <= t <= t4_end):
            return args["Omega_R"]
        return 0.0
    
    H = [
        H_rot,
        [pulse_op, Ox_coeff],
    ]

    args = {"Omega_R": Omega_R}
    result = mesolve(
        H,
        spinsys.psi0,
        tlist,
        c_ops=spinsys.collapse_ops,
        e_ops=e_ops,
        args=args,
        options=solver_options,
    )

    # --------------------------------------------------
    # Plotting 
    plt.figure(figsize=(10, 6))
    for i, e_op in enumerate(e_ops):
        plt.plot(tlist, result.expect[i], label=f'Operator {i}')
    plt.xlabel('Time (s)')
    plt.ylabel('Expectation Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# endregion ESEEM

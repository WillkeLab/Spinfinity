import numpy as np
import scipy as sp
import scipy.constants as const

class SpinSys:
    """
    SpinSys(Spins)
    A class to represent a quantum spin system for spin simulations.

    Parameters
    ----------
    Spins : list
        List of spin quantum numbers for each spin in the system.
    Attributes
    ----------
    muB : float
        Bohr magneton in meV/T.
    meVtoGHzConversion : float
        Conversion factor from meV to GHz.
    g : float
        Default g-factor for spins (can be overwritten).
    Spins : list
        List of spin quantum numbers.
    NSpins : int
        Number of spins in the system.
    dimensionOfMatrix : int
        Dimension of the Hilbert space matrix representing the spin system.
    ReadoutSpin : int
        Index of the spin used for readout.
    AtomPositions : ndarray
        Positions of atoms in the system (shape: NSpins x 3).
    latticeLengthXY : ndarray
        Lattice constants in the XY plane [nm].
    T : float
        Temperature in Kelvin.
    B : list of float
        External magnetic field vector [T].
    BTip : list of float
        Magnetic field from the tip [T].
    BTipGradient : list of float
        Gradient of the tip magnetic field [T].
    tip : str
        Tip type: 'a' for antiferromagnetic, 'f' for ferromagnetic.
    H : ndarray
        Spin Hamiltonian matrix (complex).
    gfactorvector : list
        List of g-factors for each spin.
    Dvector : ndarray
        Out-of-plane anisotropy for each spin.
    Evector : ndarray
        In-plane anisotropy for each spin.
    D0 : ndarray
        Dipole interaction matrix.
    DipoleBool : ndarray
        Boolean array to activate/deactivate dipole interactions between spins.
    Jvector : ndarray
        Vector containing all possible spin pair interactions.
    V_DC : float
        DC bias voltage [V].
    TipPolarization : list of float
        Polarization vector of the tip.
    SamplePolarization : list of float
        Polarization vector of the sample.
    U : ndarray
        On-site energies for each spin.
    b0 : float
        Ratio of spin-independent tunneling electrons.
    G : float
        Experimental conductance at 0V [A/V].
    G_ss : list
        Sample-sample conductance for each spin.
    G_tt : int
        Enables (1) or disables (0) tip-tip scattering contribution.
    T02 : float
        Tunneling parameter.
    Jrs : float
        Exchange coupling parameter.
        
    Notes
    -----
    This class provides a framework for simulating quantum spin systems, including
    Hamiltonian construction, external field effects, anisotropy, dipole interactions,
    and tunneling parameters relevant for scanning probe experiments.
    """


    # Constants
    muB = 0.05788 # [meV/T]
    kB = const.Boltzmann / const.e * 1e3  # [meV/K] Boltzmann constant in meV/K
    meVtoGHzConversion = 1e-12*const.e/const.h # approx 241.8 GHz/meV

    ## Initialize the Spin System ##
    def __init__(self, Spins: list):

        self.g=2 # Default g-factor for spins, can be overwritten
        self.Spins=Spins
        self.NSpins=len(self.Spins)
        self.dimensionOfMatrix=1 # A system with N spins [S1,S2,S3,...,SN] is represented through a matrix of dimension (2*S1+1)*(2*S2+1)*...*(2*SN+1)
        for i in range(self.NSpins):  
            self.dimensionOfMatrix *= (2 * self.Spins[i] + 1)
        self.dimensionOfMatrix = int(self.dimensionOfMatrix)
        self.ReadoutSpin = 0 
        self.AtomPositions = np.zeros((self.NSpins,3))
        self.latticeLengthXY = np.array([0.2877, 0.2877, 0]) # [nm] #!!!Not sure if we want to keep this hard-coded...

        # External Parameters
        self.T = 1 # [K]
        self.B = [0.0,0.0,0.0] # [T]
        self.BTip = [0.0,0.0,0.0] # [T]
        self.BTipGradient = [0,0,0] # [T]
        self.tip = 'f' # 'a' for antiferromagentic and 'f' for ferromagnetic

        # Spin Hamiltoninan
        self.H = np.zeros((self.dimensionOfMatrix,self.dimensionOfMatrix),dtype=complex)
        self.gfactorvector = [self.g]*self.NSpins # Each spin gets his own g-factor, by default 2
        self.Dvector = np.zeros(self.NSpins) # Out of plane Anistropy as list for each spin 
        self.Evector = np.zeros(self.NSpins) # In plane Anisotropy as list for each spin 
        self.D0=np.zeros((self.NSpins,self.NSpins)) # Creates the Dipole Interaction Matrix
        self.DipoleBool = np.zeros((1,(self.NSpins * (self.NSpins - 1)) // 2),dtype=bool) # Dipole-Bool array to activate and deactivate the dipole interaction between specific spins
        self.Jvector = np.zeros(((self.NSpins * (self.NSpins - 1)) // 2, 3)) # A vector that contains all possible spin combinations

        # Tunneling Parameters
        self.V_DC = 0 # [V]
        self.TipPolarization = [0,0,0]
        self.SamplePolarization = [0,0,0]
        self.U = np.zeros(self.NSpins) 
        self.b0 = 0 # Ratio how many electrons that tunnel do not care about the spin!!
        self.G = 1e-9; # [A/V], Experimental conductance at 0V;
        self.G_ss = [1e-8]*self.NSpins # Sample-Sample conductance
        self.G_tt = 0 # 1 enables tip-tip scattering contribution, 0 disables it!

        self.T02 = 2e-5 
        self.Jrs = 0.1 
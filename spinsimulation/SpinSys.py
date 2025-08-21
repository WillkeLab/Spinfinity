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
    kB : float
        Boltzmann constant in meV/K.
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
        Another Tunneling parameter.
    
    Kondo Parameters
    J : ndarray
        Kondo exchange coupling for each spin.
    lt : float
        Intrinsic lifetime broadening (default: 0.005, i.e., 5 µeV).
    w : float
        Bandwidth parameter.
    precission : float
        Markus Ternes: Precision parameter; sets the minimal (in) state occupation to be calculated,
        and precission^2 the minimal matrix element to be evaluated. The calculation
        speed drastically depends on this value.
    tunneling parameters relevant for scanning probe experiments, and Kondo physics.
        
    Notes
    -----
    This class provides a framework for simulating quantum spin systems, including
    Hamiltonian construction, external field effects, anisotropy, dipole interactions,
    and tunneling parameters relevant for scanning probe experiments.
    """


    # Constants
    mu0 = 1.25663706127e-6  # [N/A^2]
    muB = 0.05788 # [meV/T]
    muBHz = 9.274010078328e-24 # [J/T]
    kB = const.Boltzmann / const.e * 1e3  # [meV/K] Boltzmann constant in meV/K
    meVtoGHzConversion = 1e-12*const.e/const.h # approx 241.8 GHz/meV
    GHztomeVConversion =1/meVtoGHzConversion # approx 4.1357 uV/GHz
    hbar_meV = const.hbar * 1e-12 / const.e  # [meV*s] Reduced Planck's constant in meV*s
    h_meV = const.h * 1e-12 / const.e  # [meV*s] Planck's constant in meV*s
    meVtoJConversion=1.602e-22; # approx 1.6e-22 J/meV
    JtomeVConversion=1/meVtoJConversion; # approx 6.24e21 meV/J
    
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
        self.G = 1e-12; # [A/V], Experimental conductance at 0V;
        self.G_ss = [1e-8]*self.NSpins # Sample-Sample conductance
        self.G_tt = 0 # 1 enables tip-tip scattering contribution, 0 disables it!

        self.T02 = 1 # Tunneling parameter, used in the tunneling calculation
        # Markus Ternes: T02 is the tunneling parameter, which is used to normalize the tunneling current, default set to 1 as in Ternes code
        self.Jrs = -0.04 # Kondo coupling strength in rate equation model, default set to -0.04 as in Ternes code

        # Kondo Parameters
        self.J = np.zeros(self.NSpins)
        self.lt = 0.005;        # intrinsic lifetime broadening 5 µeV
        self.w=20;              # bandwidth
        self.precission=0.001
        # %precission sets the mininal (in) state occupation to be calculated
        # and %precission^2 the minimal matrix element to be evaluated
        # the calculation speed drastically depends on that value
        
        # Stevens Operators
        self.Stevens = [{} for _ in range(self.NSpins)]
        self.use_Stevens = False  # Flag to indicate if Stevens operators are used

    ## Print the parameters of the Spin System ##
    def print_parameters(self):
        """
        Prints the parameters of the spin system and tunneling setup.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
            This method prints information to the standard output and does not return a value.

        Notes
        -----
        The method assumes that all relevant attributes (such as NSpins, AtomPositions,
        B, Dvector, Jvector, etc.) are already initialized in the class instance.
        """

        print("Spin System Parameters:")
        print(f"Number of Spins: {self.NSpins}")
        print(f"Dimension of Matrix: {self.dimensionOfMatrix}")
        print(f"Readout Spin: {self.ReadoutSpin}")
        print(f"Atom Positions: \n{self.AtomPositions}")
        # print(f"Lattice Length XY: {self.latticeLengthXY}")
        print(f"Temperature (T): {self.T} K")
        print(f"External Magnetic Field (B): {self.B} T")
        print(f"Tip Magnetic Field (BTip): {self.BTip} T")
        print(f"Tip Field Gradient (BTipGradient): {self.BTipGradient} T")
        print(f"Tip Type: {self.tip}")
        print(f"g-factor Vector: {self.gfactorvector}")
        print(f"Out of Plane Anisotropy (Dvector): {self.Dvector} meV")
        print(f"In Plane Anisotropy (Evector): {self.Evector} meV")
        print(f"Dipole Interaction Matrix (D0): \n{self.D0*self.meVtoGHzConversion} GHz")
        print(f"Dipole Interaction Bool Array (DipoleBool): {self.DipoleBool}")
        print(f"Exchange Interaction Vector (Jvector): \n{self.Jvector} meV")
        print(f"Exchange Interaction Vector (Jvector): \n{self.Jvector*self.meVtoGHzConversion} GHz")
        print()
        print("Tunneling Parameters:")
        print(f"DC Voltage (V_DC): {self.V_DC} mV")
        print(f"Tip Polarization: {self.TipPolarization}")
        print(f"Sample Polarization: {self.SamplePolarization}")
        print(f"U: {self.U}")
        print(f"b0: {self.b0}")
        print(f"Conductance (G): {self.G} A/V")
        print(f"Sample-Sample Conductance (G_ss): {self.G_ss} A/V")
        print(f"Tip-Tip Scattering Contribution (G_tt): {self.G_tt}")
        
        return
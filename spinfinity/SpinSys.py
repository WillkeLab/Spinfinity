"""
Define the :class:`SpinSys` container used throughout the simulation package.

This module provides a single class, :class:`SpinSys`, which
stores physical constants, system geometry, magnetic fields, interaction
terms, tunneling parameters, and spectroscopy settings in one mutable
object. It also includes helpers for parameter inspection and saving/
loading the object state.
"""

import os
import pickle

import numpy as np
import scipy.constants as const


class SpinSys:
    """
    SpinSys(Spins).

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
    tip : str
        Tip type: 'a' for antiferromagnetic, 'f' for ferromagnetic.
    gfactorvector : list
        List of g-factors for each spin.
    Dvector : ndarray
        Out-of-plane anisotropy for each spin.
    EvectorX : ndarray
        In-plane anisotropy in x direction for each spin.
    EvectorY : ndarray
        In-plane anisotropy in y direction for each spin.
    DipoleVector : ndarray
        Dipolar coupling strengths D0 [meV], one per spin pair.
    Jvector : ndarray
        Vector containing all possible spin pair interactions.
    Avector : ndarray
        Vector containing all possible hyperfine interactions.
    Qvector : ndarray
        Vector containing the quadrople moment of each (nuclear) spin.
    V_DC : float
        DC bias voltage [mV].
    TipPolarization : list of float
        Polarization vector of the tip.
    SamplePolarization : list of float
        Polarization vector of the sample.
    U : ndarray
        Dimensionless potential-scattering amplitude relative to the
        spin-exchange scattering amplitude, in the package convention.
    b0 : float
        Ratio of spin-independent tunneling electrons.
    G_st : float
        Experimental conductance at 0V [A/V].
    G_ss : list
        Sample-sample conductance for each spin.
    G_tt : int
        Enables (1) or disables (0) tip-tip scattering contribution.

    Kondo Parameters
    J : ndarray
        Signed dimensionless substrate exchange couplings, one per spin,
        in the legacy spin-code normalization. Negative values describe
        antiferromagnetic exchange. Distinct from Jvector in meV.
    lt : float
        Positive lifetime broadening in meV used by third-order spectroscopy.
        Default is 0.005 meV.
    w : float
        Positive electronic bandwidth parameter in meV. Default is 20 meV.
    J_tip : float, optional
        Signed dimensionless tip exchange coupling. Default is zero.
    """

    # Constants
    mu0 = 1.25663706127e-6  # [N/A^2]
    muBHz = const.physical_constants["Bohr magneton"][0]  # [J/T]
    muB = muBHz / const.e * 1e3  # [meV/T]
    kB = const.Boltzmann / const.e * 1e3  # [meV/K] Boltzmann constant in meV/K
    meVtoGHzConversion = 1e-12 * const.e / const.h  # approx 241.8 GHz/meV
    GHztomeVConversion = 1 / meVtoGHzConversion  # approx 0.0041357 meV/GHz
    hbar_meV = const.hbar * 1e3 / const.e  # [meV*s]
    h_meV = const.h * 1e3 / const.e  # [meV*s]
    meVtoJConversion = const.e * 1e-3  # [J/meV]
    JtomeVConversion = 1 / meVtoJConversion  # [meV/J]

    # Initialize the Spin System ##
    def __init__(self, Spins: list):
        self.Spins = Spins
        self.NSpins = len(self.Spins)
        # A system with N spins [S1,S2,S3,...,SN] is represented through a matrix
        # of dimension (2*S1+1)*(2*S2+1)*...*(2*SN+1)
        self.dimensionOfMatrix = 1
        for i in range(self.NSpins):
            self.dimensionOfMatrix *= 2 * self.Spins[i] + 1
        self.dimensionOfMatrix = int(self.dimensionOfMatrix)
        self.ReadoutSpin = 0
        self.AtomPositions = np.zeros((self.NSpins, 3))
        # Lattice constants in the XY plane [nm]
        self.latticeLengthXY = np.array([0.2877, 0.2877, 0])

        # External Parameters
        self.T = 1  # [K]
        self.B = [0.0, 0.0, 0.0]  # [T]
        self.BTip = [0.0, 0.0, 0.0]  # [T]
        self.tip = "f"  # 'a' for antiferromagentic and 'f' for ferromagnetic

        # Spin Hamiltoninan Parameters
        self.gfactorvector = [2.0] * self.NSpins
        self.Dvector = np.zeros(self.NSpins)
        self.EvectorX = np.zeros(self.NSpins)
        self.EvectorY = np.zeros(self.NSpins)
        self.DipoleVector = np.zeros(
            (self.NSpins * (self.NSpins - 1)) // 2,
            dtype=float,
        )
        self.Jvector = np.zeros(((self.NSpins * (self.NSpins - 1)) // 2, 3))
        self.Avector = np.zeros(((self.NSpins * (self.NSpins - 1)) // 2, 3))
        self.Qvector = np.zeros((self.NSpins, 3))
        # Off diagonal Elements of the Exchange Parameter J
        self.Jxy = np.zeros((self.NSpins * (self.NSpins - 1)) // 2, dtype=float)
        self.Jxz = np.zeros((self.NSpins * (self.NSpins - 1)) // 2, dtype=float)
        self.Jyz = np.zeros((self.NSpins * (self.NSpins - 1)) // 2, dtype=float)
        self.Jyx = np.zeros((self.NSpins * (self.NSpins - 1)) // 2, dtype=float)
        self.Jzx = np.zeros((self.NSpins * (self.NSpins - 1)) // 2, dtype=float)
        self.Jzy = np.zeros((self.NSpins * (self.NSpins - 1)) // 2, dtype=float)

        # Tunneling Parameters
        self.V_DC = 0  # [mV]
        self.TipPolarization = [0, 0, 0]
        self.SamplePolarization = [0, 0, 0]
        self.U = np.zeros(self.NSpins)
        self.b0 = 0  # Ratio how many electrons that tunnel do not care about the spin!!
        self.G_st = 1e-12  # [A/V], Experimental conductance at 0V;
        self.G_ss = [1e-8] * self.NSpins  # Sample-Sample conductance
        self.G_tt = 0  # 1 enables tip-tip scattering contribution, 0 disables it!

        # Kondo Parameters
        self.J = np.zeros(self.NSpins)
        self.lt = 0.005  # intrinsic lifetime broadening 5 µeV
        self.w = 20  # bandwidth
        self.precission = 0.001
        self.J_tip = 0.0  # tip coupling for 3rd order calculation
        # %precission sets the mininal (in) state occupation to be calculated
        # and %precission^2 the minimal matrix element to be evaluated
        # the calculation speed drastically depends on that value

        # Tunneling Parameters for Frota function
        self.AmpFrota = 0.1  # Amplitude of the Frota function
        self.theta_rFrota = 0.0  # Phase angle of the Frota function
        self.GammaFrota = 0.01  # Lifetime broadening for the Frota function, in meV
        self.epsilonFrota = 0.0  # Energy offset for the Frota function, in meV
        self.BkgndLinFrota = 0.0  # Background linear term for the Frota function
        self.OffsetFrota = 0.0  # Offset for the Frota function
        self.use_Frota = False  # Flag to indicate if Frota function is used

        # Stevens Operators
        self.Stevens = [{} for _ in range(self.NSpins)]
        self.use_Stevens = False  # Flag to indicate if Stevens operators are used

    # region PrintParameters

    def _round_to_significant_figures(self, value, sig_figs=4):
        """
        Round a value or array to specified number of significant figures.

        Parameters
        ----------
        value : float, int, np.ndarray, list, tuple
            Value to round
        sig_figs : int
            Number of significant figures (default: 4)

        Returns
        -------
        Rounded value in same format as input
        """
        if isinstance(value, (list, tuple)):
            return type(value)(
                [self._round_to_significant_figures(v, sig_figs) for v in value]
            )
        elif isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.number):
                # Only round if it's a numeric array
                rounded = np.array(
                    [
                        self._round_to_significant_figures(v, sig_figs)
                        for v in value.flat
                    ]
                )
                return rounded.reshape(value.shape)
            else:
                return value
        elif isinstance(value, (int, float, complex)):
            if value == 0:
                return 0
            try:
                if isinstance(value, complex):
                    real_part = self._round_to_significant_figures(value.real, sig_figs)
                    imag_part = self._round_to_significant_figures(value.imag, sig_figs)
                    return complex(real_part, imag_part)
                else:
                    # For real numbers
                    from math import floor, log10

                    return round(value, -int(floor(log10(abs(value)))) + (sig_figs - 1))
            except (ValueError, OverflowError):
                return value
        else:
            return value

    def print_parameters(self):
        """Print a categorized overview of configured model parameters."""
        params = self.get_parameters_dict()

        for category, param_dict in params.items():
            print(f"\n{category.upper()}")
            print("=" * len(category))
            for key, value in param_dict.items():
                if isinstance(value, np.ndarray):
                    rounded_value = self._round_to_significant_figures(value)
                    print(f"  {key}:")
                    if rounded_value.ndim == 1:
                        # 1D array - display as simple array
                        print(f"    {rounded_value}")
                    elif rounded_value.ndim == 2:
                        # 2D array - display as matrix
                        print(f"    {rounded_value}")
                    else:
                        # Higher dimensional arrays - show shape and summary
                        print(f"    Shape: {rounded_value.shape}")
                        print(f"    Type: {rounded_value.dtype}")
                elif isinstance(value, (list, tuple, int, float, complex)):
                    rounded_value = self._round_to_significant_figures(value)
                    print(f"  {key:<25}: {rounded_value}")
                else:
                    print(f"  {key:<25}: {value}")

    def get_parameters_dict(self):
        """
        Return parameters organized in a dictionary by category.

        Returns
        -------
        dict
            Dictionary with parameter categories as keys and parameter dicts as values.
        """
        params_dict = {
            "System": {
                "Spins": self.Spins,
                "NSpins": self.NSpins,
                "dimensionOfMatrix": self.dimensionOfMatrix,
                "ReadoutSpin": getattr(self, "ReadoutSpin", "Not set"),
                "Temperature": f"{self.T} K",
            },
            "Magnetic Fields": {
                "B_external": np.array(self.B) if hasattr(self, "B") else "Not set",
                "BTip": np.array(getattr(self, "BTip", [0, 0, 0])),
                "g_factors": (
                    np.array(getattr(self, "gfactorvector", []))
                    if getattr(self, "gfactorvector", None) is not None
                    else "Not set"
                ),
            },
            "Interactions": {},
            "Tunneling": {},
        }

        # Add interaction matrices if they exist
        if hasattr(self, "Jvector") and self.Jvector is not None:
            params_dict["Interactions"]["Exchange_J_meV"] = self.Jvector
            params_dict["Interactions"]["Exchange_J_GHz"] = (
                self.Jvector * self.meVtoGHzConversion
            )

        if hasattr(self, "Dvector") and self.Dvector is not None:
            params_dict["Interactions"]["Anisotropy_D_meV"] = np.array(self.Dvector)

        if hasattr(self, "EvectorX") and self.EvectorX is not None:
            params_dict["Interactions"]["Anisotropy_Ex_meV"] = np.array(self.EvectorX)
        if hasattr(self, "EvectorY") and self.EvectorY is not None:
            params_dict["Interactions"]["Anisotropy_Ey_meV"] = np.array(self.EvectorY)

        if hasattr(self, "DipoleVector") and self.DipoleVector is not None:
            params_dict["Interactions"]["Dipole_meV"] = self.DipoleVector
            params_dict["Interactions"]["Dipole_GHz"] = (
                self.DipoleVector * self.meVtoGHzConversion
            )

        if hasattr(self, "AtomPositions") and self.AtomPositions is not None:
            params_dict["System"]["AtomPositions"] = self.AtomPositions

        # Add tunneling parameters if they exist
        if hasattr(self, "V_DC"):
            params_dict["Tunneling"]["V_DC_mV"] = self.V_DC
        if hasattr(self, "TipPolarization"):
            params_dict["Tunneling"]["TipPolarization"] = np.array(self.TipPolarization)
        if hasattr(self, "SamplePolarization"):
            params_dict["Tunneling"]["SamplePolarization"] = np.array(
                getattr(self, "SamplePolarization", [0, 0, 0])
            )
        if hasattr(self, "G_st"):
            params_dict["Tunneling"]["G_st_A_per_V"] = self.G_st
        if hasattr(self, "G_ss"):
            params_dict["Tunneling"]["G_ss"] = (
                np.array(self.G_ss)
                if isinstance(self.G_ss, (list, tuple))
                else self.G_ss
            )
        if hasattr(self, "G_tt"):
            params_dict["Tunneling"]["G_tt"] = self.G_tt
        if hasattr(self, "U"):
            params_dict["Tunneling"]["U"] = self.U
        if hasattr(self, "b0"):
            params_dict["Tunneling"]["b0"] = self.b0

        params_dict["System"]["latticeLengthXY_nm"] = self.latticeLengthXY
        params_dict["Magnetic Fields"]["tip"] = self.tip

        params_dict["Interactions"].update(
            {
                "Hyperfine_A_meV": self.Avector,
                "Quadrupole_Q_meV": self.Qvector,
                "Exchange_Jxy_meV": self.Jxy,
                "Exchange_Jxz_meV": self.Jxz,
                "Exchange_Jyz_meV": self.Jyz,
                "Exchange_Jyx_meV": self.Jyx,
                "Exchange_Jzx_meV": self.Jzx,
                "Exchange_Jzy_meV": self.Jzy,
                "use_Stevens": self.use_Stevens,
                "Stevens_coefficients_meV": self.Stevens,
            }
        )

        return params_dict

    # endregion PrintParameters

    # region SaveFunctions

    def save(self, filename=None, filepath=None):
        """
        Save the SpinSys object to a pickle file.

        Parameters
        ----------
        filename : str, optional
            Name of the file to save. If None, generates a default name based on
            system parameters.
        filepath : str, optional
            Directory path where to save the file. If None, saves in current directory.

        Returns
        -------
        str
            Full path to the saved file.

        Examples
        --------
        >>> spinsys = SpinSys([1, 0.5])
        >>> spinsys.save('my_spinsystem.pkl')
        >>> spinsys.save(filename='experiment1.pkl', filepath='./data/')
        """
        # Generate default filename if not provided
        if filename is None:
            spin_str = "_".join([f"S{s}" for s in self.Spins])
            filename = f"SpinSys_{spin_str}_B{self.B[2]:.2f}T.pkl"

        # Set default filepath if not provided
        if filepath is None:
            filepath = "."

        # Ensure filepath exists
        os.makedirs(filepath, exist_ok=True)

        # Create full file path
        full_path = os.path.join(filepath, filename)

        # Save the object using pickle
        try:
            with open(full_path, "wb") as f:
                pickle.dump(self, f)
            print(f"SpinSys object successfully saved to: {full_path}")
            return full_path
        except Exception as e:
            print(f"Error saving SpinSys object: {e}")
            raise

    @staticmethod
    def load(filename: str, filepath: str):
        """
        Load a SpinSys object from a pickle file.

        Parameters
        ----------
        filename : str
            Name of the pickle file to load.
        filepath : str
            Directory path where the file is located.

        Returns
        -------
        SpinSys
            The loaded SpinSys object.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        """
        full_path = os.path.join(filepath, filename)

        try:
            with open(full_path, "rb") as f:
                spinsys = pickle.load(f)
            print(f"SpinSys object successfully loaded from: {full_path}")
            return spinsys
        except FileNotFoundError:
            print(f"Error: File '{full_path}' not found.")
            raise
        except Exception as e:
            print(f"Error loading SpinSys object: {e}")
            raise

    # endregion SaveFunctions

import numpy as np
import scipy as sp
import scipy.constants as const
from spinsimulation.SpinSys import SpinSys
import spinsimulation.SpinHamiltonian as hamil 
import spinsimulation.calcdIdV as dIdV
import matplotlib.pyplot as plt
import spinsimulation.calcBath as bath 
from scipy.signal import convolve
import plotly.graph_objects as go
import plotly.colors
from plotly.subplots import make_subplots


def e_epsilon(SpinSys: SpinSys):
    """
    This function calculates four different 4x4x4 complex tensors that represent
    various tunneling pathways in a spin system. The tensors are constructed using
    transport operators for both tip and sample, as well as their Hermitian conjugates.
    The calculation iterates over all combinations of spin operator indices and spin-½
    basis states.
    Parameters
    ----------
    SpinSys : SpinSys
        Spin system object containing the relevant spin operators and parameters.
    Returns
    -------
    e : np.ndarray
        The primary third-order ε-tensor for tunneling pathways (shape: 4x4x4).
    er : np.ndarray
        The reversed tunneling pathway tensor (shape: 4x4x4).
    en : np.ndarray
        The conjugate tunneling pathway tensor (shape: 4x4x4).
    enr : np.ndarray
        The reversed conjugate tunneling pathway tensor (shape: 4x4x4).
    Notes
    -----
    - The tensors are computed by contracting spin operator matrices with spin-½ basis vectors.
    - The function assumes the presence of an `e_transport` function and that `SpinSys`
      has attributes `TipPolarization` and `SamplePolarization`.
    - All tensors are complex-valued and represent different physical tunneling processes.
    
    Compute third-order spin ε-tensors for all tunneling directions.
    Returns:
        e, er, en, enr : np.ndarray
            4x4x4 tensors representing tunneling pathways.
    """
    # Transport operators for tip → sample
    [x,y,z,u] = bath.calcTunnelingElectronMatrixElements(SpinSys,scattertype='ss')
    tempSpinSys=SpinSys
    tempSpinSys.TipPolarization=tempSpinSys.SamplePolarization
    [xs,ys,zs,us] = bath.calcTunnelingElectronMatrixElements(tempSpinSys,scattertype='ss')

    # so1 = tip, so2 = sample, so3 = conjugate of tip
    so1 = [x, y, z, u]              # list of 4 matrices: [Sx, Sy, Sz, S1]
    so2 = [xs,ys,zs,us]           # list of 4 matrices
    so3 = [x.conj().T, y.conj().T, z.conj().T, u.conj().T]  # Hermitian conjugates
    mone = np.ones((1, 2)) # Spin-½ basis vector |↑⟩

    e_tensor   = np.zeros((4, 4, 4), dtype=complex)
    er_tensor  = np.zeros_like(e_tensor)
    en_tensor  = np.zeros_like(e_tensor)
    enr_tensor = np.zeros_like(e_tensor)

    for j in range(4):
        for k in range(4):
            for l in range(4):
                for state in [0, 1]:
                    term_e = (
                        (mone.T * so3[l][state, :][np.newaxis, :]) *
                        so2[k].T *
                        (mone * so1[j][:, state][:, np.newaxis])
                    )
                    e_tensor[j, k, l] += np.sum(term_e)

                    term_er = -(
                        (mone.T * so3[l][state, :][np.newaxis, :]) *
                        (mone * so1[k][:, state][:, np.newaxis]) *
                        so2[j].T
                    )
                    er_tensor[j, k, l] += np.sum(term_er)

                    term_en = -(
                        (mone * so1[l][:, state][:, np.newaxis]) *
                        so2[k].T *
                        (mone.T * so3[j][state, :][np.newaxis, :])
                    )
                    en_tensor[j, k, l] += np.sum(term_en)

                    term_enr = (
                        (mone * so1[l][:, state][:, np.newaxis]) *
                        (mone.T * so3[k][state, :][np.newaxis, :]) *
                        so2[j].T
                    )
                    enr_tensor[j, k, l] += np.sum(term_enr)

    return e_tensor, er_tensor, en_tensor, enr_tensor


def compute_third_order_rates(SpinSys: SpinSys, initial_state_idx):
    """
    Compute third-order tunneling rates for inelastic electron tunneling spectroscopy (IETS).
    This function calculates the third-order tunneling rates for a given spin system and initial state,
    returning rate matrices for different tunneling processes. The computation involves transforming
    spin operator matrices into the eigenbasis, constructing tensor contractions with symmetry factors,
    and accumulating contributions for each process.
    Markus Ternes (function M35):
    calculate the tunneling rate in 3rd order via all possible tunneling pathes
    output are 4 matrices which give the probabiliy to tunnel from in->i->j

    SpinSys : SpinSys
        Object containing spin operator matrices (Sx, Sy, Sz), eigenvectors, and coupling constants.
    initial_state_idx : int
        Index of the initial state in the eigenbasis.
    rate_e : np.ndarray
        'normal' matrix for tunneling from tip to sample
    rate_er : np.ndarray
        er is the 'time-reversed' matrix for tunneling from tip to sample.
    rate_en : np.ndarray
        en is the 'normal' matrix for tunneling from sample to tip
    rate_enr : np.ndarray
        enr is the 'time-reversed' matrix for tunneling from sample to tip
    
    Notes
    ---------
    - The function assumes that the spin operator matrices and eigenvectors are properly initialized in `SpinSys`.
    - The returned rate matrices can be used to analyze tunneling transitions and their contributions to IETS spectra.
    - The symmetry factors (e_sig, er_sig, en_sig, enr_sig) are computed externally and encode selection rules for the processes.
    """
    E = SpinSys.eigVectors
    e_sig, er_sig, en_sig, enr_sig = e_epsilon(SpinSys)
    
    dim = E.shape[0]

    # Initialize RateMatrix
    # if scattertype == "ss":
    #     RateMatrix = np.zeros((spinsys.NSpins, spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix))
    #     N = spinsys.NSpins
    #     Index = np.arange(N)
    # else:
    RateMatrix = np.zeros((1, SpinSys.dimensionOfMatrix, SpinSys.dimensionOfMatrix)) 
    N = 1
    Index = np.array([SpinSys.ReadoutSpin])

    # Use @ for matrix multiplication instead of * (which is element-wise)
    # Slightly different notation. The idea is to use the same language for the 3rd order code as in the 2nd order code.
    for k in range(N):
        MatrixX = np.conjugate(np.transpose(SpinSys.eigVectors)) @ SpinSys.Sx[Index[k], :, :] @ SpinSys.eigVectors
        MatrixY = np.conjugate(np.transpose(SpinSys.eigVectors)) @ SpinSys.Sy[Index[k], :, :] @ SpinSys.eigVectors
        MatrixZ = np.conjugate(np.transpose(SpinSys.eigVectors)) @ SpinSys.Sz[Index[k], :, :] @ SpinSys.eigVectors
        MatrixU = SpinSys.U[Index[k]] * np.conjugate(np.transpose(SpinSys.eigVectors)) @ SpinSys.eigVectors
    # matrxx = E.conj().T @ SpinSys.Sx[0] @ E
    # matrxy = E.conj().T @ SpinSys.Sy[0] @ E
    # matrxz = E.conj().T @ SpinSys.Sz[0] @ E
    # I = np.eye(SpinSys.Sx[0].shape[0])
    # matrxu=SpinSys.U * E.conj().T @ I @ E

    J = SpinSys.J[SpinSys.ReadoutSpin]*np.array([1, 1, 1])

    Smatrxx = J[0] * MatrixX
    Smatrxy = J[1] * MatrixY
    Smatrxz = J[2] * MatrixZ
    Smatrxu = np.zeros_like(MatrixX)

    TJ=np.array([1, 1, 1]);  #standard value - no changes implemented yet
    matrxx = TJ[0]*MatrixX
    matrxy = TJ[1]*MatrixY
    matrxz = TJ[2]*MatrixZ
    matrxu = -np.sign(TJ[0]) * np.sign(TJ[1]) * np.sign(TJ[2]) * MatrixU

    # so1 = tip, so2 = sample, so3 = conjugate of tip
    so1 = [matrxx, matrxy, matrxz, matrxu]              # list of 4 matrices: [Sx, Sy, Sz, S1]
    so2 = [Smatrxx,Smatrxy,Smatrxz,Smatrxu]           # list of 4 matrices
    so3 = so1

    mone = np.ones((1, dim)) # Spin-½ basis vector |↑⟩

    e_tensor   = 0
    er_tensor  = 0
    en_tensor  = 0
    enr_tensor = 0

    for j in range(4):
            for k in range(4):
                for l in range(4):
                    if e_sig[j][k][l] != 0:
                        term_e = (
                            (mone.T * so3[l][initial_state_idx, :][np.newaxis, :]) *
                            so2[k].T *
                            (mone * so1[j][:, initial_state_idx][:, np.newaxis])
                        )
                        e_tensor -= e_sig[j][k][l] * term_e

                    if er_sig[j][k][l] != 0:
                        term_er = (
                            (mone.T * so3[l][initial_state_idx, :][np.newaxis, :]) *
                            so1[k].T *
                            (mone * so2[j][:, initial_state_idx][:, np.newaxis])
                        )
                        er_tensor -= er_sig[j][k][l] * term_er

                    if en_sig[j][k][l] != 0:
                        term_en = (
                            (mone.T * so1[l][initial_state_idx, :][np.newaxis, :]) *
                            so2[k].T *
                            (mone * so3[j][:, initial_state_idx][:, np.newaxis])
                        )
                        en_tensor -= en_sig[j][k][l] * term_en

                    if enr_sig[j][k][l] != 0:
                        term_enr = (
                            (mone.T * so1[l][initial_state_idx, :][np.newaxis, :]) *
                            so3[k].T *
                            (mone * so2[j][:, initial_state_idx][:, np.newaxis])
                        )
                        enr_tensor -= enr_sig[j][k][l] * term_enr
    return e_tensor, er_tensor, en_tensor, enr_tensor



def spec3(SpinSys: SpinSys, initial_state_idx, V_array):
    """
    Calculate the third-order tunneling (Kondo-like) contribution to the IETS spectrum.
    This function computes the third-order tunneling contribution to the inelastic electron tunneling spectroscopy (IETS) spectrum for a given spin system. It evaluates the spectrum as a function of bias voltage, considering all relevant transitions and their rates, and returns the resulting spectral contributions.
    Parameters
    ----------
    SpinSys : SpinSys
        The spin system object containing all relevant parameters, eigenvalues, eigenvectors, and temperature.
    initial_state_idx : int
        Index of the initial state from which transitions are considered.
    V_array : array_like
        Array of bias voltages at which the spectrum is evaluated.
    Returns
    -------
    V_array : ndarray
        The input array of bias voltages.
    y : ndarray
        The calculated third-order tunneling contribution to the spectrum at each voltage.
    yr : ndarray
        The calculated third-order tunneling contribution for the reverse process at each voltage.
    Notes
    -----
    - The function requires the `compute_third_order_rates`, `occupations`, `ln_t2`, and `fstep` helper functions.
    - The precision threshold for including transitions is determined by `SpinSys.precission`.
    - The function assumes that the spin system object provides all necessary physical parameters.
    """

    lt = SpinSys.lt
    bath.calcPopulations(SpinSys)
    occ = SpinSys.Populations  # Occupation probabilities of the eigenstates
    w = SpinSys.w  # parameter 'w' from target atom
    ve = np.real(SpinSys.E_All)  # Eigenvalues of the system

    # Get rate matrices: you must implement M35 to return these 4 arrays
    rate, rater, raten, ratenr = compute_third_order_rates(SpinSys,initial_state_idx)

    # Adjust precision threshold
    prec = SpinSys.precission**2
    # To be added from Ternes Code: If SpinSys has T_ts, use it; otherwise, use SpinSys.precission
    #if hasattr(SpinSys, 'T_ts'):
        #prec = SpinSys.precission**2 * (np.sum(SpinSys.T_ts**2)**2) / 9
    #else:
    #    prec = SpinSys.precission**2

    speclength = len(V_array)

    y = np.zeros(speclength)
    yr = np.zeros(speclength)

    dim = SpinSys.eigVectors.shape[0]
    for mid in range(dim):
        for fin in range(dim):
            if (
                np.abs(rate[mid, fin]) +
                np.abs(raten[mid, fin]) +
                np.abs(rater[mid, fin]) +
                np.abs(ratenr[mid, fin])
            ) > prec:
                E_mi = ve[mid] - ve[initial_state_idx]
                E_fi = ve[fin] - ve[initial_state_idx]

                # These are vectorized versions across V_array
                tlog1 = ln_t2(E_mi + V_array, w, SpinSys.kB * SpinSys.T, lt)
                tlog2 = ln_t2(E_mi - V_array, w, SpinSys.kB * SpinSys.T, lt)

                ep = dIdV.tunnelingBroadenedStepFunction((E_fi - V_array) / (SpinSys.kB * SpinSys.T))
                en = dIdV.tunnelingBroadenedStepFunction((E_fi + V_array) / (SpinSys.kB * SpinSys.T))

                ytemp   = -occ[initial_state_idx] * 2 * np.real(rate[mid, fin] * ep * tlog2)
                ytempn  = -occ[initial_state_idx] * 2 * np.real(raten[mid, fin] * en * tlog1)
                ytempr  = -occ[initial_state_idx] * 2 * np.real(rater[mid, fin] * ep * tlog1)
                ytempnr = -occ[initial_state_idx] * 2 * np.real(ratenr[mid, fin] * en * tlog2)

                y  += ytemp + ytempn
                yr += ytempr + ytempnr

    return V_array, y, yr


def ln_t2(x, e0, T, lt):
    """
    Temperature-folded log convolution function (used in third-order tunneling spectra).
    
    Parameters
    ----------
    x : ndarray
        Voltage bias array (1D).
    e0 : float
        Half-bandwidth of the substrate electrons (meV).
    T : float
        Temperature (meV).
    lt : float
        Lifetime broadening (meV).

    Returns
    -------
    e : ndarray
        Convolved result (same shape as x).
    """

    x = np.asarray(x, dtype=float)
    xspan = x[0] - x[-1]  # assumes x is descending; flip if needed
    i = 2 * len(x)

    # Extended energy window for convolution kernel
    y = np.linspace(xspan, -xspan, i + 1)

    # Define complex logarithmic kernel
    complex_lt = lt * 1j

    g = np.log((e0 + np.abs(y)) / (y + complex_lt)) + np.pi * 1j / 2

    # Fermi-broadened kernel
    z = x / T
    ez = np.exp(z)
    denom = (ez - 1)
    denom_sq = denom**2
    denom_cu = denom**3
    f = -((ez + ez * (z - 1)) / denom_sq - 2 * ez * (ez * (z - 1) + 1) / denom_cu)

    # Handle z ≈ 0 where numerical instabilities occur
    f[np.abs(z) < 1e-3] = 1 / 6
    f[np.isnan(f)] = 0

    # Perform convolution (same size output)
    conv_result = convolve(f, g, mode="same")
    e = 2 * conv_result / (T * i) * np.abs(xspan)

    return e


def calcIETS_3rd( SpinSys: SpinSys, Vrange: list = [-10, 10], N: int = 100, norm: bool = False,
    plot: bool = False
):
    """
    Calculate IETS spectrum with third-order contributions (e.g. Kondo processes)

    Parameters
    ----------
    SpinSys : SpinSys
        Spin system object containing all relevant parameters and operators.
    Vrange : list, optional
        Voltage range as [Vmin, Vmax] (default: [-10, 10]).
    N : int, optional
        Number of voltage points (default: 100).
    norm : bool, optional
        If True, normalize the resulting spectrum (default: False).
    plot : bool, optional
        If True, plot the resulting spectrum (default: True).

    Returns
    -------
    V_array : np.ndarray
        Array of bias voltages (mV).
    dIdV_3rd : np.ndarray
        Third-order differential conductance spectrum (a.u.).
    """

    # Bias voltage array
    V_array = np.linspace(Vrange[0], Vrange[1], N)
    dIdV_3rd = np.zeros_like(V_array, dtype=float)

    [V_array, dIdV2] = dIdV.calcIETS_2nd(SpinSys, Vrange=[np.min(V_array), np.max(V_array)],N=N, norm=False)  # Calculate the second order IETS:
    dIdV_3rd += dIdV2

    # Calculate thermal calcPopulations
    p0 = np.exp(-SpinSys.E_All / (SpinSys.kB * SpinSys.T))
    pTot = np.sum(p0)
    SpinSys.p0 = p0 / pTot
    dim = SpinSys.eigVectors.shape[1]

    for i in range(dim):
        if SpinSys.p0[i] > 1e-6:
            # Calculate third-order contributions for each state
            if SpinSys.use_Frota:
                y1 = calcFrota(SpinSys, V_array)
            else:
                x, y1, y1r = spec3(SpinSys, i, V_array)
                y1 = y1+ y1r  # Combine forward and reverse contributions
                # Add contributions from the third-order processes
            dIdV_3rd += y1

    # Normalize result
    if norm == True:
        #dIdV_3rd = (dIdV_3rd - np.min(dIdV_3rd)) / (np.max(dIdV_3rd) - np.min(dIdV_3rd)) old nomrlaization to min and max
        dIdV_3rd = dIdV_3rd / dIdV_3rd[0]  # Normalize to the first value

    if plot:
        plt.figure(189)
        plt.plot(V_array, dIdV_3rd, '-b', linewidth=2)
        plt.ylabel('dI/dV (3rd order, a.u.)')
        plt.xlabel('Bias Voltage (mV)')
        plt.xlim([Vrange[0], Vrange[1]])
        plt.title('IETS Spectrum - Third Order (Kondo)')
        plt.grid(True)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.show()

    return V_array, dIdV_3rd


def calcFrota(SpinSys: SpinSys, V_array):
    """
    Calculate the Frota function for Kondo resonance in STS.

    Parameters:
        V_array : ndarray
            Voltage array (in Volts)
        a : float
            Amplitude
        theta_r : float
            Phase shift (in radians)
        Gamma : float
            Kondo width
        epsilon : float
            Resonance energy
        b : float
            Linear background coefficient
        c : float
            Constant background coefficient

    Returns:
        dIdV : ndarray
            Differential conductance dI/dV
    """
    numerator = 0.39 * 1j * SpinSys.GammaFrota
    denominator = V_array - SpinSys.epsilonFrota + 0.39 * 1j * SpinSys.GammaFrota
    sqrt_term = np.sqrt(numerator / denominator)
    frota = SpinSys.AmpFrota * np.imag(-1j * np.exp(1j * SpinSys.theta_rFrota) * sqrt_term)
    return frota + SpinSys.BkgndLinFrota * V_array + SpinSys.OffsetFrota

# region Visualization
def plotIETS(spinsys: SpinSys, Vrange: list=[-30, 30], N: int=100, norm: bool=False, order: str='3rd', plotAllSpins: bool=False):
    """
    Plot the Inelastic Electron Tunneling Spectroscopy (IETS) spectrum for a given spin system.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object for which the IETS spectrum is to be calculated.
    Vrange : list, optional
        The voltage range [min, max] in millivolts over which to calculate the spectrum (default is [-30, 30]).
    N : int, optional
        Number of points in the voltage range for the spectrum calculation (default is 400).
    norm : bool, optional
        If True, normalize the calculated spectrum (default is False).
    order : str, optional
        The order of the IETS calculation. Supported orders are '3rd' and '3rdR'.

    Raises
    ------
    ValueError
        If an unsupported order is specified.

    Returns
    -------
    return_dict : dict
        A dictionary containing:
        - 'fig': plotly.graph_objects.Figure
            The plotly figure object containing the ESR spectrum plot.
        - 'V_array': numpy.ndarray
            The voltage values used in the spectrum.
        - 'dIdVsignal': numpy.ndarray
            The calculated dI/dV signal values.
    Notes
    -----
    - the dimension of the dIdVsignal changes when plotAllSpins=True
    """
    # Define y-axis label according to normalization 
    if norm:
        y_label = 'Normalized dI/dV (a.u.)'
    else:
        y_label = 'dI/dV (S)'

    if plotAllSpins==False:

        # Initialize x and y arrays
        x = np.zeros(N) # Voltage 
        y = np.zeros(N) # dI/dV Signal

        # Calculate the IETS spectrum for the specified order
        if order == '3rd':
            x, y = calcIETS_3rd(spinsys, Vrange=Vrange, N=N, norm=norm)
        elif order == '3rdR':
            x, y = calcIETS_3rdR(spinsys, Vrange=Vrange, N=N, norm=norm) # Not implemented yet!
        else:
            raise ValueError("Unsupported order. Currently only '2nd' order is implemented.")
    
        # Plot the IETS spectrum using Plotly with white background and black border
        y_max = np.max(y)
        if y_max == 0:
            y_max = 1.0
        x0, x1 = float(np.min(x)), float(np.max(x))
        y0, y1 = 0.0, float(y_max * 1.1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name='IETS Spectrum',
            line=dict(color='black', width=2)
        ))

        fig.update_layout(
            title=f'IETS Spectrum ({order} Order)',
            xaxis_title='Bias Voltage (mV)',
            yaxis_title=y_label,
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=800,
            height=500,
            margin=dict(l=60, r=20, t=60, b=60)
        )

        # set explicit axis ranges and add a black rectangle border around the plotting area
        fig.update_xaxes(range=[x0, x1], showgrid=True, gridcolor='lightgray', zeroline=False)
        fig.update_yaxes(range=[y0, y1], showgrid=True, gridcolor='lightgray', zeroline=False)

        fig.update_layout(
            shapes=[dict(
            type="rect",
            xref="x",
            yref="y",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            line=dict(color="black", width=2),
            fillcolor="rgba(0,0,0,0)"
            )]
        )

        fig.show()

    elif plotAllSpins==True:

        # Initialize x and y arrays
        x = np.zeros(N)
        y = np.zeros((len(spinsys.Spins),N))

        for i in range(len(spinsys.Spins)):
            spinsys.ReadoutSpin = i
            #print(f"Calculating IETS for Spin {i} with ReadoutSpin={spinsys.ReadoutSpin}")
            if order == '3rd':
                x, y[i,:] = calcIETS_3rd(spinsys, Vrange=Vrange, N=N, norm=norm)
            elif order == '3rdR':
                x, y[i,:] = calcIETS_3rdR(spinsys, Vrange=Vrange, N=N, norm=norm)
            else:
                raise ValueError("Unsupported order. Currently only '2nd' '2ndR' and '3rd' order are implemented.")
            
            # Plot all IETS spectra in the same figure on top of each other
            # Only create the figure once, before the loop
            #if i == 0:
        # After computing x and y for all spins, build a Plotly figure similar to the single-spin case
        color_list = plotly.colors.qualitative.Plotly
        fig = go.Figure()

        # Add one trace per spin (overlayed)
        nspins = y.shape[0]
        for i in range(nspins):
            fig.add_trace(go.Scatter(
            x=x,
            y=y[i, :],
            mode='lines',
            line=dict(color=color_list[i % len(color_list)], width=2),
            name=f'Spin {i}'
            ))

        # Determine axis ranges similar to single-spin branch
        y_max = np.max(y) if np.max(y) != 0 else 1.0
        x0, x1 = float(np.min(x)), float(np.max(x))
        y0, y1 = 0.0, float(y_max * 1.1)

        fig.update_layout(
            title=f'IETS Spectrum ({order} Order) for All Spins',
            xaxis_title='Bias Voltage (mV)',
            yaxis_title=y_label,
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=800,
            height=500,
            margin=dict(l=60, r=20, t=60, b=60)
        )

        fig.update_xaxes(range=[x0, x1], showgrid=True, gridcolor='lightgray', zeroline=False)
        fig.update_yaxes(range=[y0, y1], showgrid=True, gridcolor='lightgray', zeroline=False)

        # add a black rectangle border around the plotting area
        fig.update_layout(
            shapes=[dict(
            type="rect",
            xref="x",
            yref="y",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            line=dict(color="black", width=2),
            fillcolor="rgba(0,0,0,0)"
            )]
        )

        fig.show()

    return_dict = {
        'fig': fig,
        'V_array': x,
        'dIdVsignal': y}      
        
    return return_dict

# endregion Visualization

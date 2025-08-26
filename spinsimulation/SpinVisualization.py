import numpy as np
import scipy as sp
import scipy.constants as const
import matplotlib.pyplot as plt
from spinsimulation.SpinSys import SpinSys
import spinsimulation.SpinHamiltonian as hamil 
import spinsimulation.calcBath as bath
import spinsimulation.calcdIdV as dIdV
import spinsimulation.calcESR as ESR
import spinsimulation.calcdIdV3 as dIdV3

# region Hamiltonian

def plotEigenMatrix(spinsys: SpinSys, plot_type: str='p'):
        """
        Visualizes the eigenstate matrix of the spin system as a heatmap.

        Parameters
        ----------
        spinsys : SpinSys
            The spin system object containing eigenvectors, basis states, and other relevant properties.
        plot_type : str, optional
            Type of plot to generate:
            - 'p': Probability (default), displays the probability matrix using the 'hot' colormap.
            - 'a': Amplitude, displays the amplitude matrix using the 'coolwarm' colormap.

        Returns
        -------
        None
            This function generates a matplotlib figure and does not return any value.

        Raises
        ------
        ValueError
            If `plot_type` is not 'p' or 'a'.
            
        Notes
        -----
        The color limits are set to [0, 1] for probability and [-1, 1] for amplitude.
        """

        # Check if eigVectors are calculated, if not calculate them. They are needed for the projection matrix.
        if not hasattr(spinsys, 'eigVectors') or spinsys.eigVectors is None:
            hamil.calcEigStates(spinsys)
               
        projection = np.zeros([spinsys.dimensionOfMatrix,spinsys.dimensionOfMatrix],dtype=complex)
        for i in range(spinsys.dimensionOfMatrix):
            for j in range(spinsys.dimensionOfMatrix):
                projection[i, j] = np.dot(spinsys.eigVectors[:, i].conj().T, np.identity(spinsys.dimensionOfMatrix)[:, j])


        if plot_type == 'p':
            m = abs(np.conjugate(projection) * projection)
            name0, colormap = 'Probability', 'hot'
        elif plot_type == 'a':
            m = np.real(projection) + np.imag(projection)
            name0, colormap = 'Amplitude', 'coolwarm'
        else:
            raise ValueError("Invalid plot_type. Use 'p' for probability or 'a' for amplitude.")

        
        # Generate the figure
        plt.figure(110)
        plt.imshow(np.flipud(m), cmap=colormap, aspect='auto') #We have to flip the projection matrix in order to have the first entry on the lower left corner
        c = plt.colorbar()
        plt.xlabel('Basis States')
        plt.ylabel('Eigenstates')
        plt.title(f"System Eigenstates, B_Ext = [{spinsys.B[0]}, {spinsys.B[1]}, {spinsys.B[2]}]")
        c.set_label(name0)
        plt.xticks(np.arange(spinsys.dimensionOfMatrix), spinsys.basisStates)
        plt.yticks(np.arange(spinsys.dimensionOfMatrix), spinsys.statesOnlyLHS[::-1])

        # Set color limits
        plt.clim(0, 1) if plot_type == 'p' else plt.clim(-1, 1)

        return
    
def plotZeemanDiagramm(spinsys: SpinSys,Brange: list=[0,1], N: int=200, energyscale: str='meV'):
    """
    Plots the Zeeman diagram for a given spin system over a specified magnetic field range.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing all relevant parameters and methods, including the magnetic field vector and energy levels.
    Brange : list of float, optional
        The range of magnetic field values (in Tesla) to sweep, given as [B_min, B_max]. Default is [0, 1].
    N : int, optional
        The number of points in the magnetic field sweep. Default is 200.

    Returns
    -------
    None
        This function does not return a value. It displays a plot of the Zeeman diagram.

    Notes
    -----
    - The function temporarily modifies the magnetic field vector of `spinsys` during the calculation and restores it to its original value afterwards.
    """

    # Save the initial B-Vector
    Binitial = spinsys.B
    B = np.linspace(Brange[0],Brange[1],N)
    E = np.zeros((N,spinsys.dimensionOfMatrix))
    
    for i in range(N):
        spinsys.B = [0,0,B[i]]
        hamil.calcEigEnergies(spinsys)
        E[i,:] = spinsys.E_sort

    # Resotre the initial solution
    spinsys.B = Binitial
    hamil.calcEigEnergies(spinsys)
        
    fig, ax = plt.subplots()
    for i in range(spinsys.dimensionOfMatrix): 
        ax.plot(B,E[:,i],label=f"E_{i}")
        if energyscale == 'GHz':
            ax.plot(B,E[:,i]*spinsys.meVtoGHzConversion,label=f"E_{i}")           
        elif energyscale == 'meV':
            ax.plot(B,E[:,i],label=f"E_{i}")          
    ax.set_xlabel('Magnetic Field (T)')
    ax.set_ylabel(f'Energy ({energyscale})')
    ax.set_title('Zeeman Diagramm')
    ax.legend()
    plt.show()

    return fig

def plotEnergyVsS(spinsys: SpinSys, type: str='z',energyscale: str='meV'):
    """
    Plots the energy eigenvalues of a spin system versus the expectation values of a spin operator (Sx, Sy, or Sz).

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing Hamiltonian, eigenvectors, and energies.
    type : str, optional
        The spin operator to use for the expectation value ('x', 'y', or 'z'). Default is 'z'.

    Returns
    -------
    None
        This function displays a scatter plot and does not return any value.
        
    Notes
    -----
    If the eigenvectors of the spin system are not already calculated, the function will compute them.
    The expectation value of the selected spin operator is calculated for each eigenstate and plotted against its energy.
    """
    
    if not hasattr(spinsys, 'eigVectors') or spinsys.eigVectors is None:
        hamil.calcEigEnergies(spinsys)

    if type == 'z':
        S_Operator = np.sum(spinsys.Sz, axis=0)
    elif type == 'x':
        S_Operator = np.sum(spinsys.Sx, axis=0)
    elif type == 'y':
        S_Operator = np.sum(spinsys.Sy, axis=0)

    # Calculate the Sz matrix elements for each state
    S_elements = np.zeros(spinsys.dimensionOfMatrix)

    for i in range(spinsys.dimensionOfMatrix):
        S_elements[i] = np.real(spinsys.eigVectors[:, i].T.conj() @ S_Operator @ spinsys.eigVectors[:, i])

    # Plot the energy vs Sz matrix elements
    plt.figure()
    if energyscale == 'GHz':
        plt.scatter(S_elements, spinsys.E_All_inGHz, c='k', marker='_', s=1000, linewidths=5)
    elif energyscale == 'meV':
        plt.scatter(S_elements, spinsys.E_All, c='k', marker='_', s=1000, linewidths=5)
    #plt.xlabel(r'$\langle \psi | S_z | \psi \rangle$')
    plt.xlabel('m_'+type)
    plt.ylabel(f'Energy ({energyscale})')   
    plt.title('Energy vs S_' + type +  ' Matrix Elements')
    plt.grid(True)
    plt.show()

    return plt

def plotAll(spinsys: SpinSys, type: str='z', energyscale: str='meV'):
    """
    Plots all relevant information for the spin system, including energy levels, Zeeman diagram, and energy vs spin expectation values.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing Hamiltonian, eigenvectors, and energies.
    type : str, optional
        The spin operator to use for the expectation value ('x', 'y', or 'z'). Default is 'z'.
    energyscale : str, optional
        The energy scale to use for the plots ('meV' or 'GHz'). Default is 'meV'.

    Returns
    -------
    None
        This function does not return a value. It displays several plots.
    """
    # make them all plot next to each other
    plotEigenMatrix(spinsys, plot_type='p')
    plotEnergyVsS(spinsys, type=type, energyscale=energyscale)
    plotZeemanDiagramm(spinsys, energyscale=energyscale)
    plotPopulations(spinsys, AllowPumping=True, Approach="Loth")
# endregion Hamiltonian

# region Bath
def plotPopulations(spinsys: SpinSys, AllowPumping: bool = True, Approach: str = "Loth"):
    """
    Plots the thermal populations of the spin system's states.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the necessary parameters and methods.
    AllowPumping : bool, optional
        Whether to allow pumping effects in the calculation (default is True).
    Approach : str, optional
        The approach to use for calculating populations (default is "Loth").

    Returns
    -------
    None
        This function generates a plot of the populations but does not return any value.
    
    Notes
    -----
    - The function assumes that the `spinsys` object has the necessary attributes and methods for rate calculations.
    """
    
    if not hasattr(spinsys, 'Populations') or spinsys.Populations is None:
            bath.calcPopulations(spinsys,AllowPumping=AllowPumping, Approach=Approach)
    if not hasattr(spinsys, 'basisSates') or spinsys.basisStates is None:
            hamil.calcEigStates(spinsys)
    
    x = np.arange(len(spinsys.Populations))
    width = 0.35

    fig, ax = plt.subplots()
    ax.barh(x + width/2, spinsys.pop_0, height=width, color='blue', alpha=0.5, label='Boltzmann Populations')
    ax.barh(x - width/2, spinsys.Populations, height=width, color='red', alpha=0.5, label='Tunneling Populations')
    ax.set_yticks(x)
    ax.set_yticklabels(spinsys.basisStates)
    ax.set_xlabel('Population')
    ax.set_ylabel('Basis States')
    ax.set_title('Population of Spin States')
    ax.legend()
    plt.tight_layout()
    plt.show()

    return

def plotRateContributions(spinsys: SpinSys):
    """
    Plot the contributions of transition rates between spin states.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the states, rates, and matrix dimension.
        It is expected to have the following attributes:
            - dimensionOfMatrix: int, the number of spin states.
            - statesWithoutE: list of str, labels for the spin states.
            - Rates: list of 2D numpy arrays, each representing the rate matrix for a process.

    Returns
    -------
    None
        This function displays a matplotlib plot and does not return any value.

    Notes
    -----
    The function assumes that `bath.calcRates(spinsys)` populates the `spinsys.Rates`
    attribute with the required rate matrices. The plot shows stacked bars for each
    transition, with each color representing a different rate contribution.
    """

    bath.calcRates(spinsys)

    labels = ['ts', 'st', 'ss','tt']
    colors = ['blue', 'orange', 'green', 'red']

    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(spinsys.dimensionOfMatrix):
        for f in range(spinsys.dimensionOfMatrix):
            if i != f:
                bottom=0
                for idx in range(4):
                    ax.bar(f'{spinsys.statesWithoutE[i]} → {spinsys.statesWithoutE[f]}', spinsys.Rates[idx][i, f], bottom=bottom, alpha=0.5, label=labels[idx], color=colors[idx])
                    bottom += spinsys.Rates[idx][i, f]
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
    ax.set_xlabel('Transitions')
    ax.set_ylabel('Rate')
    ax.set_title('Rate Contributions')
    ax.set_yscale('log')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    # Only show legend for the first bar (Rate_exc)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.show()
    
    return

# endregion Bath

# region dIdV

def plotIETS(spinsys: SpinSys, Vrange: list=[-30, 30], N: int=100, norm: bool=False, order: str='2nd', plotAllSpins: bool=False):
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
        The order of the IETS calculation. Currently, only '2nd' order is supported (default is '2nd').

    Raises
    ------
    ValueError
        If an unsupported order is specified.

    Returns
    -------
    None
        This function creates a plot of the IETS spectrum but does not return any value.
    """
    if plotAllSpins==False:
        # Calculate the IETS spectrum for the specified order
        if order == '2nd':
            x, y = dIdV.calcIETS_2nd(spinsys, Vrange=Vrange, N=N, norm=norm)
        elif order == '2ndR':
            x, y, _ = dIdV.calcIETS_2ndR(spinsys, Vrange=Vrange, N=N, norm=norm)
        elif order == '3rd':
            x, y = dIdV3.calcIETS_3rd(spinsys, Vrange=Vrange, N=N, norm=norm)
        elif order == '3rdR':
            x, y = dIdV3.calcIETS_3rdR(spinsys, Vrange=Vrange, N=N, norm=norm)
        else:
            raise ValueError("Unsupported order. Currently only '2nd' order is implemented.")
    
        # Plot the IETS spectrum
        plt.figure() 
        plt.plot(x, y, label='IETS Spectrum')
        plt.xlabel('Bias Voltage (mV)')
        plt.ylabel('dI/dV (a.u.)')
        plt.ylim([0, np.max(y) * 1.1])  # Set y-axis limits to avoid cutting off the top of the plot
        plt.title('IETS Spectrum ' + f'({order} Order)')
    elif plotAllSpins==True:
        fig, ax = plt.subplots()
        for i in range(len(spinsys.Spins)):
            spinsys.ReadoutSpin = i
            print(f"Calculating IETS for Spin {i} with ReadoutSpin={spinsys.ReadoutSpin}")
            if order == '2nd':
                x, y = dIdV.calcIETS_2nd(spinsys, Vrange=Vrange, N=N, norm=norm)
            elif order == '2ndR':
                x, y, _ = dIdV.calcIETS_2ndR(spinsys, Vrange=Vrange, N=N, norm=norm)
            elif order == '3rd':
                x, y = dIdV3.calcIETS_3rd(spinsys, Vrange=Vrange, N=N, norm=norm)
            elif order == '3rdR':
                x, y = dIdV3.calcIETS_3rdR(spinsys, Vrange=Vrange, N=N, norm=norm)
            else:
                raise ValueError("Unsupported order. Currently only '2nd' '2ndR' and '3rd' order are implemented.")
            
            # Plot all IETS spectra in the same figure on top of each other
            # Only create the figure once, before the loop
            #if i == 0:
            ax.plot(x, y, label=f'Spin {i}')

        # Set labels and title for the plot
        ax.set_xlabel('Bias Voltage (mV)')
        ax.set_ylabel('dI/dV (a.u.)')
        ax.set_title(f'IETS Spectrum ({order} Order) for All Spins')
        ax.legend()
        ax.set_ylim(bottom=0)  # start from 0

        plt.tight_layout()
        plt.show()
            
        
    return x,y 

# endregion dIdV

# region ESR

def plotESR_FreqSweep_Benjamin(spinsys: SpinSys, FreqRange: list = [10, 20], AllowPumping: bool = False, N: int = 1000, lw: float = 0.02, norm: bool = True):
    """
    Plot the ESR (Electron Spin Resonance) spectrum for a given spin system over a specified frequency range.
    This function computes the ESR spectrum using the provided spin system and plots the resulting signal.
    It also marks the resonance frequencies corresponding to recorded transitions in the spin system.
    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing resonance frequencies and recorded transitions.
    FreqRange : list of float, optional
        The frequency range [start, stop] in GHz over which to compute the ESR spectrum (default is [10, 20]).
    AllowPumping : bool, optional
        Whether to allow pumping effects in the ESR calculation (default is False).
    N : int, optional
        Number of frequency points in the sweep (default is 1000).
    lw : float, optional
        Linewidth for the ESR calculation (default is 0.02).
    norm : bool, optional
        Whether to normalize the ESR signal (default is True).
    Returns
    -------
    None
        The function generates a plot and does not return any value.
    Notes
    -----
    - Resonance frequencies and transitions are marked on the plot with different colors.
    - The ESR spectrum is calculated using `ESR.calcESR_Benjamin`.
    """
    
    # calculate the ESR spectrum
    freq_array, ESRsignal = ESR.calcESR_FreqSweep_Benjamin(spinsys, FreqRange=FreqRange, AllowPumping=AllowPumping,N=N, lw=lw, norm=norm, output=True)

    # Make the plot
    color_list = plt.cm.hsv(np.linspace(0, 1, len(spinsys.ResonanceFrequencies)))

    plt.figure()
    plt.plot(freq_array, ESRsignal, '-k', linewidth=2, label='ESR Signal')

    # Mark resonances
    for i, freq_res in enumerate(spinsys.ResonanceFrequencies):
        index = np.argmin(np.abs(freq_array - freq_res))
        amp = ESRsignal[index]
        transition_str = (f"|{spinsys.RecordedTransitions[i][0]}> -> |{spinsys.RecordedTransitions[i][1]}>; "
                            f"f = {np.round(freq_res,4)} GHz")
        plt.plot(freq_res, amp, color=color_list[i], marker='.', markersize=12, 
                label=transition_str, linewidth=2)

    plt.xlabel('Freq (GHz)')
    plt.ylabel('ΔI (a.u.)')
    plt.title('ESR-Simulation')
    plt.legend()
    plt.xlim([freq_array[0],freq_array[-1]])
    plt.show()

    return

def plotESR_FreqSweep_ESRrate(spinsys: SpinSys, FreqRange = [10, 20], N: int = 200, norm: bool = True):
    """
    This function computes the ESR spectrum based on an ESR driving rate for a given spin system over a specified frequency range
    and plots the resulting ESR signal. It also marks the resonance frequencies corresponding to allowed transitions in the spin
    system.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing energy levels and transition information.
    FreqRange : list of float, optional
        The frequency range [start, stop] in GHz over which to perform the sweep (default is [10, 20]).
    N : int, optional
        Number of frequency points in the sweep (default is 1000).
    norm : bool, optional
        Whether to normalize the ESR signal (default is True).

    Returns
    -------
    None
        This function generates a plot and does not return any value.
        
    Notes
    -----
    - The function updates `spinsys.RecordedTransitions` and `spinsys.ResonanceFrequencies` with the
      calculated transitions and their corresponding resonance frequencies.
    - Each resonance is marked on the plot with a unique color and labeled with the transition information.
    """

    # Calculate the ESR spectrum using the specified parameters
    freq_array, ESRsignal = ESR.calcESR_FreqSweep_ESRrate(spinsys, FreqRange=FreqRange, N=N, norm=norm)
    
    # Record the transitions
    spinsys.RecordedTransitions = ESR.calcRecordedTransitions(spinsys,freq_array[0],freq_array[-1])
    spinsys.ResonanceFrequencies = np.zeros(len(spinsys.RecordedTransitions))
    for i in range(len(spinsys.RecordedTransitions)):
        spinsys.ResonanceFrequencies[i] = abs(spinsys.E_All_inGHz[spinsys.RecordedTransitions[i][1]] - 
                                        spinsys.E_All_inGHz[spinsys.RecordedTransitions[i][0]])

    # Make the plot
    color_list = plt.cm.hsv(np.linspace(0, 1, len(spinsys.ResonanceFrequencies)))

    plt.figure()
    plt.plot(freq_array, ESRsignal, '-k', linewidth=2, label='ESR Signal')

    # Mark resonances
    for i, freq_res in enumerate(spinsys.ResonanceFrequencies):
        index = np.argmin(np.abs(freq_array - freq_res))
        amp = ESRsignal[index]
        transition_str = (f"|{spinsys.RecordedTransitions[i][0]}> -> |{spinsys.RecordedTransitions[i][1]}>; "
                            f"f = {np.round(freq_res,4)} GHz")
        plt.plot(freq_res, amp, color=color_list[i], marker='.', markersize=12, 
                label=transition_str, linewidth=2)
        
    plt.xlabel('Freq (GHz)')
    plt.ylabel('ΔI (a.u.)')
    plt.title('ESR-Simulation')
    plt.legend()
    plt.xlim([freq_array[0],freq_array[-1]])
    plt.show()

    return

def plotESR_Tipfieldsweep(spinsys: SpinSys, FreqRange: list = [10, 20], N_Freq: int = 500, lw: float = 0.02, norm: bool = True, BTipRange: list = [-0.05, 0.05], N_B: int = 100, AllowPumping: bool = False, Angle: float = 0):  
    """
    Plot ESR spectra as a function of tip magnetic field for a given spin system.
    
    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing all relevant spin and field parameters.
    FreqRange : list of float, optional
        Frequency range [min, max] in GHz for the ESR spectrum (default is [10, 20]).
    N_Freq : int, optional
        Number of frequency points in the ESR spectrum (default is 500).
    lw : float, optional
        Linewidth for the ESR peaks (default is 0.02).
    norm : bool, optional
        Whether to normalize the ESR signal (default is True).
    BTipRange : list of float, optional
        Range [min, max] of tip magnetic field values in Tesla (default is [-0.05, 0.05]).
    N_B : int, optional
        Number of tip field values to compute (default is 100).
    AllowPumping : bool, optional
        Whether to allow pumping effects in the ESR calculation (default is False).
    Angle : float, optional
        Angle (in degrees) of the tip field with respect to the z-axis (default is 0).

    Returns
    -------
    None
        This function displays a plot and does not return any value.
        
    Notes
    -----
    The function visualizes the ESR signal as a 2D color plot with frequency on the x-axis
    and tip field on the y-axis. A secondary y-axis shows the corresponding detuning in GHz.
    """
    # Save previous BTip value
    BTip_initial = spinsys.BTip.copy()

    # Predefine Arrays
    BTip_array = np.linspace(BTipRange[0],BTipRange[1],N_B,dtype=float)
    Freq_array = np.linspace(FreqRange[0],FreqRange[1],N_Freq,dtype=float)

    ESRsignal = np.zeros((N_B, N_Freq), dtype=float)
    
    # calculating the ESR signal for each Field 
    for i in range(N_B):
        spinsys.BTip[2]=BTip_array[i]*np.cos(Angle*np.pi/180) # BTip[2] is the z-component of the tip field
        spinsys.BTip[0]=BTip_array[i]*np.sin(Angle*np.pi/180)+1e-6 # BTip[0] is the x-component of the tip field
        spinsys.BTip[1]=0  # BTip[0] is the y-component of the tip field
        hamil.calcEigEnergies(spinsys)
        f , ESRsignal[i][:] = ESR.calcESR_Benjamin(spinsys, norm=norm,N=N_Freq,lw=lw,FreqRange=FreqRange,AllowPumping=AllowPumping, output=False)
        

    # Calculating the Detuning from the Tipfield
    Detuning=spinsys.gfactorvector[spinsys.ReadoutSpin] * spinsys.muB * (BTip_array + spinsys.B[2]) * spinsys.meVtoGHzConversion - spinsys.g * spinsys.muB * spinsys.B[2] * spinsys.meVtoGHzConversion


    # Plot the ESR Signal as a color plot
    fig, ax1 = plt.subplots()

    #Plot the matrix using imshow
    im = ax1.imshow(ESRsignal, cmap='Blues', interpolation='none', 
                    extent=[Freq_array[0], Freq_array[-1], BTip_array[0]*1e3, BTipRange[-1]*1e3], origin='lower', aspect='auto')

    # Add labels and title for primary Y-axis
    ax1.set_xlabel('Freq (GHz)')
    ax1.set_ylabel('B_Tip (mT)')
    ax1.set_title('Tipfield dependent ESR' + f' (Angle = {Angle}°)')

    # Create a secondary Y-axis
    ax2 = ax1.twinx()
    ax2.set_ylim(BTip_array[0], BTip_array[-1])
    y_ticks = ax1.get_yticks()
    detuning_labels = np.interp(y_ticks / 1e3, BTip_array, Detuning)
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(np.round(detuning_labels, 2))
    ax2.set_ylabel('Detuning (GHz)')

    cbar = plt.colorbar(im, ax=ax1)
    cbar.ax.set_position([0.9, 0.1, 0.03, 0.8])
    
    
    plt.show()

    # Restore the initial BTip value
    spinsys.BTip = BTip_initial
    hamil.calcEigEnergies(spinsys)

    return

def plotPopulations_ESR(spinsys: SpinSys, freq: float = 14):
    """
    Plot the populations of spin states for a given spin system under ESR conditions.
    This function calculates and visualizes the populations of spin states, including Boltzmann, tunneling, 
    and ESR-driven populations, for a provided spin system. The populations are displayed as horizontal bar plots 
    for easy comparison.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the relevant state and population information.
    freq : float, optional
        The frequency of the ESR field (in the same units as used in the simulation), by default 14.

    Returns
    -------
    None
        This function does not return any value. It displays a matplotlib plot of the populations.

    Notes
    -----
    - The function assumes that the `spinsys` object has attributes `pop_0`, `Populations`, `Populations_ESR`, and `basisStates`.
    - If the eigenstates are not already calculated, the function will compute them.
    - Requires matplotlib and numpy for plotting and array operations.
    """

    ESR.calcPopulations_ESR(spinsys, freq=freq)
    if not hasattr(spinsys, 'basisSates') or spinsys.basisStates is None:
            hamil.calcEigStates(spinsys)
    
    x = np.arange(len(spinsys.Populations))
    width = 0.2

    fig, ax = plt.subplots()
    ax.barh(x, spinsys.pop_0, height=width, color='blue', alpha=0.5, label='Boltzmann Populations')
    ax.barh(x + width, spinsys.Populations, height=width, color='red', alpha=0.5, label='Tunneling Populations')
    ax.barh(x - width, spinsys.Populations_ESR, height=width, color='green', alpha=0.5, label='ESR Populations')
    ax.set_yticks(x)
    ax.set_yticklabels(spinsys.basisStates)
    ax.set_xlabel('Population')
    ax.set_ylabel('Basis States')
    ax.set_title('Population of Spin States')
    ax.legend()
    plt.tight_layout()
    plt.show()

    return
# endregion ESR

# region Magnetization over Voltage 

def plotMagnetization_Voltage(spinsys: SpinSys, Vrange: list=[-200, 200], N: int=100, type: str='z', spins: list=[0], norm: bool=True, total: bool=True):
    """
    Plots the magnetization of specified spins and the total magnetization as a function of voltage.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the spin configuration and state.
    Vrange : list of float, optional
        The voltage range [V_min, V_max] (in mV) over which to sweep. Default is [-200, 200].
    N : int, optional
        Number of voltage points in the sweep. Default is 100.
    type : str, optional
        The type of magnetization component to calculate ('x', 'y', or 'z'). Default is 'z'.
    spins : list of int, optional
        List of spin indices to plot. Default is [0].
    norm : bool, optional
        If True, normalize the magnetization values. Default is True.
    total : bool, optional
        If True, plot the total magnetization. Default is True.

    Returns
    -------
    None
        This function displays a plot and does not return any value.

    Notes
    -----
    The function temporarily modifies the `V_DC` attribute of `spinsys` during the voltage sweep,
    and restores its original value after plotting. The calculations rely on external `bath` functions
    for rates, populations, and magnetization.
    """
    
    # Save the old voltage value
    V_initial = spinsys.V_DC

    if len(spins) > spinsys.NSpins:
        print("Error: Too many spins specified in the 'spin' argument.")
        return
    
    # Initialize arrays to store voltage and magnetization data
    V_array = np.linspace(Vrange[0], Vrange[1], N)
    M_array = np.zeros((N, spinsys.NSpins+1)) 

    for idx, V in enumerate(V_array):
        spinsys.V_DC = V
        bath.calcRates(spinsys)
        bath.calcPopulations(spinsys, AllowPumping=True, Approach="Loth")
        bath.calcMagnetization(spinsys, type=type, AllowPumping=True, Approach= "Loth")
        for i in range(spinsys.NSpins):
            if norm:
                M_array[idx, i] = spinsys.Mag[i]/spinsys.Mag_max[i]
            else:
                M_array[idx, i] = spinsys.Mag[i]
        
        if norm:
            M_array[idx, -1] = spinsys.Mag_tot/spinsys.Mag_tot_max
        else:
            M_array[idx, -1] = spinsys.Mag_tot

    # Plotting
    plt.figure(figsize=(8, 5))
    for idx, spin in enumerate(spins):
        plt.plot(V_array, M_array[:, spin], label=f'Spin {spin} Magnetization')
    if total:
        plt.plot(V_array, M_array[:, -1], label='Total Magnetization', linestyle='--', color='k')
    plt.xlabel('Voltage (mV)')
    plt.ylabel('Magnetization')
    plt.title('Magnetization vs Voltage')
    plt.legend()
    plt.show()

    # Restore the initial voltage value
    spinsys.V_DC = V_initial
    bath.calcRates(spinsys)
    bath.calcPopulations(spinsys, AllowPumping=True, Approach="Loth")
    bath.calcMagnetization(spinsys, type=type, AllowPumping=True, Approach= "Loth")
    
    return 

def plotPopulations_Voltage(
        spinsys: SpinSys,
        Vrange: list = [-200, 200],
        N: int = 200):
    """
    Plots the populations of spin states as a function of applied voltage.
    
    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the current state and parameters.
    Vrange : list, optional
        The voltage range [V_min, V_max] (in mV) over which to sweep. Default is [-200, 200].
    N : int, optional
        The number of voltage points to evaluate. Default is 200.

    Returns
    -------
    None
        This function does not return a value; it displays a plot of populations vs voltage.

    Notes
    -----
    - The function temporarily modifies `spinsys.V_DC` during the sweep and restores its original value at the end.
    - The bath object and its methods `calcRates` and `calcPopulations` must be accessible in the current scope.
    """

    # Save the old voltage value
    V_initial = spinsys.V_DC

    # Initialize arrays to store voltage and population data
    V_array = np.linspace(Vrange[0], Vrange[1], N)
    P_array = np.zeros((N, spinsys.dimensionOfMatrix))
    for idx, V in enumerate(V_array):
        spinsys.V_DC = V
        bath.calcRates(spinsys)
        bath.calcPopulations(spinsys, AllowPumping=True, Approach="Loth")
        P_array[idx, :] = spinsys.Populations

    # Plotting
    plt.figure(figsize=(8, 5))
    for idx in range(spinsys.dimensionOfMatrix):
        plt.plot(V_array, P_array[:, idx], label=f'State {idx}')
    plt.xlabel('Voltage (mV)')
    plt.ylabel('Populations')
    plt.title('Populations vs Voltage')
    plt.legend()
    plt.yscale('log')
    plt.show()

    # Restore the initial voltage value
    spinsys.V_DC = V_initial
    bath.calcRates(spinsys)
    bath.calcPopulations(spinsys, AllowPumping=True, Approach="Loth")

    return

def plotRates_Voltages(spinsys: SpinSys, Vrange: list=[-200, 200], N: int=100):
    """
    Plots the transition rates between spin states as a function of applied voltage.
    
    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the current state and parameters, including the
        voltage (`V_DC`) and the rates matrix (`Rates_Summed`).
    Vrange : list of float, optional
        The voltage range [V_min, V_max] (in mV) over which to sweep. Default is [-200, 200].
    N : int, optional
        The number of voltage points to evaluate. Default is 100.

    Returns
    -------
    None
        This function does not return a value. It displays a plot of the transition rates.
        
    Notes
    -----
    - The function temporarily modifies `spinsys.V_DC` during the sweep and restores its original
      value at the end.
    - The function assumes that `bath.calcRates(spinsys)` updates `spinsys.Rates_Summed` in-place.
    - Only off-diagonal transition rates (i ≠ f) are plotted.
    """

    # Save the old voltage value
    V_initial = spinsys.V_DC

    # Initialize arrays to store voltage and rate data
    V_array = np.linspace(Vrange[0], Vrange[1], N)
    R_array = np.zeros((N, spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix))
    
    for idx, V in enumerate(V_array):
        spinsys.V_DC = V
        bath.calcRates(spinsys)
        for i in range(spinsys.dimensionOfMatrix):
            for f in range(spinsys.dimensionOfMatrix):
                if i != f:
                    R_array[idx, i, f] = spinsys.Rates_Summed[i,f]

    # Plotting
    plt.figure(figsize=(10, 6))
    for i in range(spinsys.dimensionOfMatrix):
        for f in range(spinsys.dimensionOfMatrix):
            if i != f:
                plt.plot(V_array, R_array[:, i, f], label=f'Rate {i}→{f}')
    
    plt.xlabel('Voltage (mV)')
    plt.ylabel('Transition Rates')
    plt.title('Transition Rates vs Voltage')
    plt.yscale('log')
    plt.legend()
    plt.show()

    # Restore the initial voltage value
    spinsys.V_DC = V_initial
    bath.calcRates(spinsys)

    return

# endregion Magnetization over Voltage
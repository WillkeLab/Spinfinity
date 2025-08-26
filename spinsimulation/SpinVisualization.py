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
    
    import matplotlib.ticker as ticker

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
            #print(f"Calculating IETS for Spin {i} with ReadoutSpin={spinsys.ReadoutSpin}")
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
            
        
    return

# endregion dIdV

# region ESR

def plotESR_FreqSweep(spinsys: SpinSys, FreqRange: list = [10, 20], AllowPumping: bool = False, N: int = 1000, lw: float = 0.02, norm: bool = True):
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
    freq_array, ESRsignal = ESR.calcESR_Benjamin(spinsys, FreqRange=FreqRange, AllowPumping=AllowPumping,N=N, lw=lw, norm=norm, output=True)

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

    # Predefine Arrays
    BTip_array = np.linspace(BTipRange[0],BTipRange[1],N_B,dtype=float)
    Freq_array = np.linspace(FreqRange[0],FreqRange[1],N_Freq,dtype=float)

    ESRsignal = np.zeros((N_B, N_Freq), dtype=float)
    
    # calculating the ESR signal for each Field 
    for i in range(N_B):
        spinsys.BTip[2]=BTip_array[i]*np.cos(Angle*np.pi/180) # BTip[2] is the z-component of the tip field
        spinsys.BTip[0]=BTip_array[i]*np.sin(Angle*np.pi/180) # BTip[0] is the x-component of the tip field
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

    return

# endregion ESR
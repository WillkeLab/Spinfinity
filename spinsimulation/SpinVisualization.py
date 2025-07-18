import numpy as np
import scipy as sp
import scipy.constants as const
import matplotlib.pyplot as plt
from SpinSys import SpinSys
import SpinHamiltonian as hamil 
import calcdIdV as dIdV

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
    
def plotZeemanDiagramm(spinsys: SpinSys,Brange: list=[0,1], N: int=200):
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
        ax.plot(B,E[:,i]*spinsys.meVtoGHzConversion,label=f"E_{i}")
    ax.set_xlabel('Magnetic Field (T)')
    ax.set_ylabel('Energy (GHz)')
    ax.set_title('Zeeman Diagramm')
    ax.legend()
    plt.show()

    return

def plotEnergyVsSz(spinsys: SpinSys):
        """
        Plots the energy eigenvalues of a spin system versus the expectation values of the Sz operator.

        Parameters
        ----------
        spinsys : SpinSys
            An instance of the SpinSys class containing the spin system's Hamiltonian, eigenvectors, and eigenvalues.
            The object must have the attributes `eigVectors`, `E_All_inGHz`, `Sz`, and `dimensionOfMatrix`.

        Returns
        -------
        None
            This function displays a matplotlib scatter plot and does not return any value.

        Notes
        -----
        If the eigenvectors are not already computed and stored in `spinsys.eigVectors`, the function will
        call `hamil.calcEigEnergies(spinsys)` to compute them.
        """
        
        
        if not hasattr(spinsys, 'eigVectors') or spinsys.eigVectors is None:
            hamil.calcEigEnergies(spinsys)

        # Calculate the Sz matrix elements for each state
        Sz_elements = np.zeros(spinsys.dimensionOfMatrix)

        Sz_Operator = np.sum(spinsys.Sz, axis=0)
        for i in range(spinsys.dimensionOfMatrix):
            Sz_elements[i] = np.real(spinsys.eigVectors[:, i].T.conj() @ Sz_Operator @ spinsys.eigVectors[:, i])

        # Plot the energy vs Sz matrix elements
        plt.figure()
        plt.scatter(Sz_elements, spinsys.E_All_inGHz, c='b', marker='o')
        #plt.xlabel(r'$\langle \psi | S_z | \psi \rangle$')
        plt.xlabel('m_z')
        plt.ylabel('Energy (GHz)')
        plt.title('Energy vs Sz Matrix Elements')
        plt.grid(True)
        plt.show()

        return
    
# endregion Hamiltonian

# region dIdV

def plotIETS(spinsys: SpinSys, Vrange: list=[-30, 30], N: int=100, norm: bool=False, order: str='2nd'):
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

    # Calculate the IETS spectrum for the specified order
    if order == '2nd':
        x, y = dIdV.calcIETS_2nd(spinsys, Vrange=Vrange, N=N, norm=norm)
    elif order == '2ndR':
        x, y, _ = dIdV.calcIETS_2ndR(spinsys, Vrange=Vrange, N=N, norm=norm)
    else:
        raise ValueError("Unsupported order. Currently only '2nd' order is implemented.")
   
    # Plot the IETS spectrum
    plt.figure() 
    plt.plot(x, y, label='IETS Spectrum')
    plt.xlabel('Bias Voltage (mV)')
    plt.ylabel('dI/dV (a.u.)')
    plt.title('IETS Spectrum ' + f'({order} Order)')
    return 

# endregion dIdV
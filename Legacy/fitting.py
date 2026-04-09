from spinsimulation.dIdV3 import calcIETS_3rd
from spinsimulation.SpinSys import SpinSys
import plotly.graph_objects as go
import copy
import numpy as np
import cma



# region fit Kondo

# function to fit a Kondo peak at 0V 
def fit_Kondo(x_exp:np.ndarray, 
              y_exp:np.ndarray, 
              spinsys: SpinSys, 
              fixed_params:dict={'g':False, # g factor
                                 'J':False, # Kondo coupling
                                 'T_eff':False, # effective temperature
                                 'sigma_0': False, # vertical offset
                                 'V_offset': False # horizontal offset
                                 }, 
              weights:np.ndarray=np.array([]),
              plot:bool=False):
    """
    Fits a Kondo simulation (3rd order) to experimental data.

    Parameters
    ----------
    x_exp : np.ndarray
        The x-values of the experimental data (bias in mV). 
    y_exp : np.ndarray 
        The y-values of the experimental data (dI/dV in a.u. (normalized)).
    spinsys : Spinsys
        The spinsystem to be used for the simulation.
        The native parameters of the spinsystem will be used for the simulation.
    fixed_params : dict, optional
        A dictionary containing information which parameters are to remain fixed and which are to be fitted.
        The keys are the parameter names and the values are booleans indicating whether the parameter is fixed (contain value) or fitted (False).
        The default is {'g':False,'J':False, 'T_eff':False, 'sigma_0': False, 'V_offset': False}.
    weights : np.ndarray, optional
        Weights for the experimental data points. If not provided, all points are equally weighted.
        Can be adapted to give more weight to data points close to 0V.
        The weights should be the same length as x_exp and y_exp.
        The default is np.array([]), which means equal weighting.
    plot : bool, optional
        If True, the fit will be plotted together with the experimental data.
        The default is False.

    Returns
    -------
    x_sim : np.ndarray
        The x-values of the fitted simulation (bias in mV).
    y_sim : np.ndarray
        The y-values of the fitted simulation (dI/dV in a.u. (normalized)).
    params : dict
        A dictionary containing the fitted parameters:
        - 'g': g factor
        - 'J': Kondo coupling
        - 'T': effective temperature
        - 'sigma_0': vertical offset
        - 'V_offset': horizontal offset
    return_spin_sys : SpinSys
        A SpinSys object with the fitted parameters applied.

    Notes
    -----
    This function uses the CMA-ES algorithm to fit the Kondo peak to the experimental data.
    The fitting is done by minimizing the sum of squared errors (SSE) between the experimental data and the simulated data.
    The function assumes that the spinsys object has been initialized with the necessary parameters, which will be used as initial guesses.
    """

    # initialize the fitting class
    kondo_fitter = KondoFitter(spinsys=spinsys,
                               V_exp= x_exp,
                               dIdV_exp= y_exp,
                               weights=weights)
    # initial guess for the parameters
    initial_guess = np.zeros(7)
    # bounds for the parameters
    lower_bounds = np.zeros(7)
    upper_bounds = np.zeros(7)
    # for each of the parameters, check if it is fixed or not
    if fixed_params['g'] == False:
        initial_guess[0] = spinsys.gfactorvector[0] # not fixed
        lower_bounds[0] = 0.0
        upper_bounds[0] = 4.0
    else:
        initial_guess[0] = fixed_params['g']  # fixed value
        lower_bounds[0] = fixed_params['g']- 1e-12 # small tolerance to avoid numerical issues
        upper_bounds[0] = fixed_params['g']+ 1e-12 # small tolerance to avoid numerical issues

    if fixed_params['J'] == False:
        initial_guess[1], initial_guess[2], initial_guess[3] = spinsys.J
        lower_bounds[1], lower_bounds[2], lower_bounds[3] = -1.0, -1.0, -1.0
        upper_bounds[1], upper_bounds[2], upper_bounds[3] = 0.0, 0.0, 0.0
    else:
        initial_guess[1], initial_guess[2], initial_guess[3]  = fixed_params['J']
        lower_bounds[1], lower_bounds[2], lower_bounds[3] = fixed_params['J']- 1e-12  # small tolerance to avoid numerical issues
        upper_bounds[1], upper_bounds[2], upper_bounds[3] = fixed_params['J']+ 1e-12  # small tolerance to avoid numerical issues

    if fixed_params['T_eff'] == False:
        initial_guess[4] = spinsys.T
        lower_bounds[4] = 0.01
        upper_bounds[4] = 5.0
    else:
        initial_guess[4] = fixed_params['T_eff']
        lower_bounds[4] = fixed_params['T_eff']- 1e-12  # small tolerance to avoid numerical issues
        upper_bounds[4] = fixed_params['T_eff']+ 1e-12  # small tolerance to avoid numerical issues

    if fixed_params['sigma_0'] == False:
        initial_guess[5] = 0.0
        lower_bounds[5] = -5.0
        upper_bounds[5] = 5.0
    else:
        initial_guess[5] = fixed_params['sigma_0']
        lower_bounds[5] = fixed_params['sigma_0']- 1e-12  # small tolerance to avoid numerical issues
        upper_bounds[5] = fixed_params['sigma_0']+ 1e-12  # small tolerance to avoid numerical issues

    if fixed_params['V_offset'] == False:
        initial_guess[6] = 0.0
        lower_bounds[6] = -2.0
        upper_bounds[6] = 2.0
    else:
        initial_guess[6] = fixed_params['V_offset']
        lower_bounds[6] = fixed_params['V_offset']- 1e-12
        upper_bounds[6] = fixed_params['V_offset']+ 1e-12

    sigma0 = 0.5  # Initial step size
    
    # Parameter bounds
    # lower_bounds = [0.0,-1.0, 0.01, -5.0, 0.1]
    # upper_bounds = [4.0,0.0, 5.0, 5.0, 5.0]
    bounds = (lower_bounds, upper_bounds)
    # print(len(bounds))
    
    # CMA-ES options
    opts = {
        'bounds': bounds,
        'maxiter': 100,
        'popsize': 12,  # Population size (4 + 3*log(N) is default)
        'tolfun': 1e-6,
        'seed': 42
    }
    
    es = cma.CMAEvolutionStrategy(initial_guess, sigma0, opts)
    es.optimize(kondo_fitter.objective)

    params, loss =  es.result.xbest, es.result.fbest
    # use the params to plot fit and exp data together
    if plot:
        # another disposable spinsys to recalculate everything
        return_spin_sys = copy.deepcopy(spinsys)
        return_spin_sys.gfactorvector = [params[0]]
        return_spin_sys.J = [params[1], params[2] , params[3]]
        return_spin_sys.T = params[4]

        x_sim, y_sim = calcIETS_3rd(return_spin_sys, Vrange=[x_exp.min(), x_exp.max()], N=1000, norm=False, plot=False)
        # same but fewer points 
        # x_100, y_100 = calcIETS_3rd(disposable_spinsys, Vrange=[x_exp.min(), x_exp.max()], N=100, norm=False, plot=False)
        # hacky normalization
        # y_sim = y_sim / np.max(y_sim)  # Normalize to max value
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_exp, y=y_exp , mode='markers', name='Experimental Data'))
        fig.add_trace(go.Scatter(x=x_sim + params[6], 
                                 y= y_sim+ params[5], 
                                 mode='lines', name='Fit'))
        # fig.add_trace(go.Scatter(x=x_100, 
        #                          y= y_100, 
        #                          mode='lines', name='Fitted Kondo Peak N100'))
        fig.update_xaxes(title='Bias (mV)')
        fig.update_yaxes(title='dI/dV (a.u.)')
        fig.show()
    

    return  x_sim + params[6], y_sim + params[5], {'g': params[0],'J': [params[1], params[2] , params[3]], 'T':params[4],'sigma_0':params[5],'V_offset':params[6]}, return_spin_sys

    


# endregion fit Kondo



# region Loss functions 

def loss_func(dIdV_exp: np.ndarray, 
              dIdV_sim: np.ndarray, 
              weights: np.ndarray = np.array([])) -> float:
    """
    Calculates the loss between experimental and simulated data.

    Parameters
    ----------
    dIdV_exp : np.ndarray
        Experimental data (dI/dV).
    dIdV_sim : np.ndarray
        Simulated data (dI/dV).
    weights : np.ndarray, optional
        Weights for the data points. If not provided, all points are equally weighted.
        The default is np.array([]), which means equal weighting.

    Returns
    -------
    float
        The calculated loss value.
    """
    if weights.size == 0:
        weights = np.ones_like(dIdV_exp)
    
    return np.sum(weights * (dIdV_exp - dIdV_sim) ** 2) / np.sum(weights)


# endregion Loss functions


# region optimizer step

# endregion optimizer step

# region KondoFitter

# copy from ChatGPT
class KondoFitter:
    """
    A class to fit Kondo peaks in dI/dV data using a SpinSys object.
    
    This class contains the experimental and simulated data for the fit and has a disposable SpinSys object as 
    attribute which is used to recalculate the dI/dV data during the fitting process.
   
    Attributes
    ----------
    spinsys : SpinSys
        The SpinSys object containing the spin system parameters.
    V_exp : np.ndarray
        The experimental voltage data (bias in mV).
    dIdV_exp : np.ndarray
        The experimental dI/dV data (in arbitrary units, normalized).
    weights : np.ndarray
        Weights for the experimental data points. If not provided, all points are equally weighted.
        The default is np.array([]), which means equal weighting.
    n_evaluations : int
        Counter for the number of evaluations of the objective function.
    disposable_spinsys : SpinSys
        A disposable SpinSys object that is used to recalculate the dI/dV data during the fitting process.
        This is a copy of the original spinsys with the fitted parameters applied.

    Methods
    -------
    objective(params: np.ndarray) -> float
        The objective function to be minimized during the fitting process.
        It updates the disposable SpinSys with the fitted parameters and calculates the loss between the experimental and simulated data.
    """

    def __init__(self, spinsys: SpinSys, 
                 V_exp: np.ndarray, 
                 dIdV_exp: np.ndarray, 
                 weights: np.ndarray = np.array([])):
        """
        Initializes the KondoFitter with the spin system and experimental data.

        Parameters
        ----------
        spinsys : SpinSys
            The spin system object containing all relevant spin Hamiltonian parameters and state information.
        V_exp : np.ndarray
            The experimental voltage data (bias in mV).
        dIdV_exp : np.ndarray
            The experimental dI/dV data (in arbitrary units, normalized).
        weights : np.ndarray, optional
            Weights for the experimental data points. If not provided, all points are equally weighted.
            The default is np.array([]), which means equal weighting.
        """
        self.spinsys = copy.deepcopy(spinsys)
        # print('B field:', self.spinsys.B)
        self.V_exp = V_exp
        self.dIdV_exp = dIdV_exp
        self.n_evaluations = 0
        self.weights = weights if weights.size > 0 else np.ones_like(V_exp)
        
    def objective(self, params:np.ndarray) -> float:
        """
        Objective function for optimization.
        
        This function takes parameters, updates the disposable SpinSys with these parameters,
        runs the dI/dV calculation, and computes the loss between the experimental and simulated data.
        
        Parameters
        ----------
        params : np.ndarray
            Parameters for the SpinSys, including g-factor, Kondo coupling, effective temperature,
            vertical offset, and horizontal offset.

        Returns
        -------
        float
            The loss value calculated from the difference between experimental and simulated dI/dV data.
        """
        self.n_evaluations += 1

        
        # Update parameters
        sigma_0 = params[5]
        v_offset = params[6]
        # create a whole new spin system to recalculate everything 
        self.disposable_spinsys = SpinSys(Spins=[0.5])
        self.disposable_spinsys.B = self.spinsys.B
        # print(self.disposable_spinsys.B)
        self.disposable_spinsys.J = [params[1], params[2] , params[3]]
        self.disposable_spinsys.T = params[4] 
        self.disposable_spinsys.gfactorvector = [params[0]]
        
        try:
            
            # Calculate theoretical spectrum -> new spin system to ensure everything is recalculated
            V_theory, dIdV_theory = calcIETS_3rd(
                self.disposable_spinsys, 
                Vrange=[self.V_exp.min(), self.V_exp.max()], 
                N=1000,
                norm=False,
                plot=False
            )
            # select the values of dIdV_theory whose V_theory is closest to the respective V_exp
            dIdV_theory = np.interp(self.V_exp, V_theory + v_offset, dIdV_theory)
            # hacky normalization
            # dIdV_theory = dIdV_theory / np.max(dIdV_theory)  # Normalize to max value
            
            
            # calculate SSE
            loss = loss_func(dIdV_exp = self.dIdV_exp,
                             dIdV_sim=dIdV_theory + sigma_0,
                             weights=self.weights)
            
            # if self.n_evaluations % 100 == 0:
            #     print(f"Evaluation {self.n_evaluations}: loss = {loss:.4e}")
                # plot current state of fitting 
                # fig = go.Figure()
                # fig.add_trace(go.Scatter(x=self.V_exp, y=modified_exp, mode='markers', name='Experimental Data'))
                # fig.add_trace(go.Scatter(x=self.V_exp, y=dIdV_theory, mode='lines', name='Simulated Data'))
                # fig.update_xaxes(title='Bias (mV)')
                # fig.update_yaxes(title='dI/dV (a.u.)')
                # fig.show()
            
            return loss
            
        except Exception as e:
            print(f"Error: {e}")
            return 1e10



    

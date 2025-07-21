from .SpinSys import SpinSys #Hello
from . import SpinHamiltonian as hamil 
from . import calcdIdV as dIdV
from . import calcESR as ESR
from . import SpinVisualization as vis

# Make everything available when importing the package
__all__ = ['SpinSys', 'hamil', 'dIdV', 'ESR', 'vis']
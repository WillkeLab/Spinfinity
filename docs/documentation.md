# Documentation

This section contains the API and module documentation for Spinfinity.

## Available modules

- [Spin System](api/spinsys.md)
- [Spin Hamiltonian](api/spinhamiltonian.md)
- [Eigensystem Import](api/eigensystem_import.md)
- [Bath](api/bath.md)
- [dIdV](api/dIdV.md)
- [dIdV3](api/dIdV3.md)
- [Spin Pumping](api/spinpumping.md)
- [ESR](api/esr.md)
- [Lindblad](api/lindblad.md)
- [CoherentControl](api/coherentcontrol.md)

## Plot display settings

Set the default once per Python session or notebook:

```python
from spinfinity import plotting

plotting.set_display_mode("responsive")
```

Use `"fixed"` for each plot's preferred width, `"responsive"` to fill the
available width, or `"no"` to suppress display. These settings apply to all
package plotting functions. Override a single call with `show="fixed"`,
`show="responsive"`, or `show="no"`; the default `show=None` inherits the setting.


## Helperfunctions

The `helperfunctions` module provides a Fourier transform calculation and three fitting functions:

| Function | Model | Parameter order |
|---|---|---|
| `fit_fano` | Symmetric Lorentzian peak or dip with constant offset | `[A, x0, FWHM, C]` |
| `fit_exponential` | Exponential decay with constant offset | `[A, T, C]` |
| `fit_oscillation` | Exponentially damped cosine with constant offset | `[A, T, f, phi, C]` |

The fitting functions return `xfit`, `yfit`, `popt`, and `perr`: a dense plotting
grid, the fitted curve, the fitted parameters, and their estimated
one-standard-deviation uncertainties.
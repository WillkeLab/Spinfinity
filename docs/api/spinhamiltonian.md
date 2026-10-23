## Plot display

Spin Hamiltonian plots use fixed widths by default. Set the session default once:

```python
from spinfinity import SpinSys, hamil, plotting

plotting.set_display_mode("responsive")

spin = SpinSys([0.5])
spin.B = [0.0, 0.0, 0.5]
hamil.calc_EigStates(spin, rebuildH=True)
output = hamil.plot_EigenMatrix(spin)
```

Override the setting for one call with `show="fixed"`, `show="responsive"`,
or `show="no"`. With `show=None` (the default), the current global setting
is used. `"no"` still computes and returns the individual plot's results.
`plot_All` forwards the setting but retains its existing `None` return value.

All package plots use this setting. The Spin Hamiltonian fixed widths are
700 px for the eigenstate matrix and energy-versus-spin plot, and 600 px for
the two Zeeman diagrams. All have a height of 450 px.

For documentation builds, `SPINFINITY_PLOT_MODE=responsive` can select the
initial setting before importing Spinfinity. An explicit call to
`set_display_mode` takes precedence. The setting does not modify the
default Plotly renderer.

## API

:::spinfinity.SpinHamiltonian

See the [shared display settings](../documentation.md#plot-display-settings)
for custom figures and notebook display behavior.

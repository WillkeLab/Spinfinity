# Spin Hamiltonian Simulator GUI

This is a desktop GUI for driving the spin-Hamiltonian simulator library in `spinsimulation`.

## Features
- Left control panel with sliders and spinboxes for simulation parameters (S, D, E, J, T, Bz, Bx, By)
- Right plot area displaying dI/dV vs eV (using matplotlib)
- Responsive UI: simulations run in a background thread
- Save/load parameter presets as JSON
- Adapter method for simulation (currently returns a synthetic spectrum; replace with real call to `spinsimulation`)

## How to Run

1. Install requirements:
   ```bash
   pip install PyQt5 matplotlib numpy
   ```
2. Run the main window:
   ```bash
   python -m ternes_ldos.spin_gui.GUI.main_window
   ```

## Development Notes
- The adapter function `simulate_spectrum` in `main_window.py` should be replaced with a real call to the simulator.
- All UI code is in `main_window.py` for now, but can be split into smaller components as needed.




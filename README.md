# Spinfinity v1.1.0

Python package for simulating quantum spin systems in tunnel junction experiments. 

[Documentation website](https://willkelab.github.io/Spinfinity/) ·
[User guide](docs/user_guide.md) ·
[Tutorials](docs/user_guide.md#continue-with-the-tutorials)

## Intended Use

Spinfinity is intended for researchers working on quantum spin systems in
scanning probe microscopy. The package is designed for modeling spin structures in double-barrier tunnel junctions, solving their spin Hamiltonians, and simulating observables that can be compared to
experimental spectroscopy data.

Typical use cases include:

- modeling single-spin and multi-spin systems
- calculating eigenenergies and eigenstates of custom spin Hamiltonians
- simulating tunneling-induced transition rates and steady-state populations
- computing differential conductance spectra for comparison with experiment
- computing Electron Spin Resonance spectra and related driven dynamics
- simulating the time evolution of spin systems in the tunneling junction
- simulating pulsed ESR experiments such as Rabi oscillations

## Installation

Spinfinity requires **Python 3.11 or newer**. We recommend using a separate
Python environment.

Clone the repository:

```bash
git clone https://github.com/WillkeLab/Spinfinity.git
cd Spinfinity
```

With your environment activated, install from the project root:

```bash
python -m pip install -e .
```

The editable installation uses changes to the local source without
reinstalling the package. Restart an already-running notebook kernel
after changing imported code.

## Documentation
The documentation is available at
[willkelab.github.io/Spinfinity](https://willkelab.github.io/Spinfinity/).

To build it locally, install the documentation dependencies and Python
kernel support:

```bash
python -m pip install -e ".[docs]" ipykernel
```

Start the local documentation server:

```bash
python -m mkdocs serve
```

For parameter units, sign conventions, and basis definitions, see
[Units and conventions](docs/units_conventions.md).

## Examples

The [examples](examples/) directory contains additional notebooks covering
spin Hamiltonians, IETS, continuous-wave and pulsed ESR, Lindblad dynamics,
spin pumping and fitting.

## References

- Ternes, M., *Spin excitations and correlations in scanning tunneling spectroscopy*.
  New J. Phys. 17, 063016 (2015).
  [DOI: 10.1088/1367-2630/17/6/063016](https://doi.org/10.1088/1367-2630/17/6/063016)
- Loth, S., von Bergmann, K., Ternes, M. et al.,
  *Controlling the state of quantum spins with electric currents*.
  Nature Phys 6, 340–344 (2010).
  [DOI: 10.1038/nphys1616](https://doi.org/10.1038/nphys1616)
- Yang, K., Willke, P., Bae, Y. et al.,
  *Electrically controlled nuclear polarization of individual atoms*.
  Nature Nanotech 13, 1120–1125 (2018).
  [DOI: 10.1038/s41565-018-0296-7](https://doi.org/10.1038/s41565-018-0296-7)
- Yang, K., Phark, SH., Bae, Y. et al.,
  *Probing resonating valence bond states in artificial quantum magnets*.
  Nat Commun 12, 993 (2021).
  [DOI: 10.1038/s41467-021-21274-5](https://doi.org/10.1038/s41467-021-21274-5)

## Citation

If you use Spinfinity in scientific work, please cite the software together
with the relevant methodological references listed above.

Suggested citation text:

```bibtex
@software{Spinfinity,
  title   = {Spinfinity: A Python package for simulating quantum spin systems in tunnel junction experiments},
  author  = {Greule, Paul and Arnold, Jonas and Renz, Luise and Hardeweg, Lovis and Matusche, Johanna and Huang, Wantong and Au-Yeung, Kwan Ho and Ascheraden, Hannah and Schwenk, Johannes and Willke, Philip},
  year    = {2026},
  url     = {https://github.com/WillkeLab/Spinfinity}
}
``` 

## Support

For support, please contact
[paul.greule@kit.edu](mailto:paul.greule@kit.edu).

## Contributing

We welcome everyone to contribute to this project. Please contact
[paul.greule@kit.edu](mailto:paul.greule@kit.edu).

## License

Spinfinity is licensed under the GNU General Public License (GPL).
See [LICENSE](LICENSE) for the full license text.

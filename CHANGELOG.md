# Changelog

## [1.1.0] - 2026-10-23

### Added

- Third-order IETS calculations (`3rd` and `3rdR`).
- Lindblad time evolution, tunneling-induced collapse operators and phenomenological dephasing.
- Coherent-control simulations: Rabi oscillations, relaxometry, Ramsey, Hahn echo and phase-cycled three-pulse ESEEM.
- Hyperfine and quadrupole interaction helpers, eigensystem import, and FFT and fitting utilities.
- Scientific regression and smoke tests, GitLab CI, and expanded example notebooks.
- Six introductory tutorials, a user guide, and a units and conventions reference.

### Changed

- Minimum supported Python version is now 3.11.
- Rate-based ESR functions now accept the driving field `B1` in tesla instead of `RabiRate`.
- State escape lifetimes are stored as `T_state_tot` and `T_state_ss` instead of `T1_tot` and `T1_ss`; these are not generally longitudinal relaxation times.
- Hamiltonian construction is explicit through `build_Hamiltonian`; eigensystem calculations support `rebuildH=True`.
- Dipolar coupling is configured per spin pair; `calc_DipolarCoupling` now requires spin indices `i` and `j`.
- Plot functions support fixed-width, responsive and hidden display modes, with a global default setting.

### Fixed

- Stabilized rate integrals and the broadened tunneling step near zero and at large arguments; added exact zero-temperature handling.
- Corrected the mV-to-V conversion for background tunneling current.
- Prevented rate calculations from repeatedly rescaling the input tunneling conductance.
- Corrected accumulation of substrate-scattering escape rates and thermal population selection for tunneling current.

## [1.0.0]

- Previous public version.

# Units and conventions

Spinfinity uses ordinary numerical values and NumPy arrays. Units are
not attached to these objects and are not converted automatically.
Inputs must therefore use the units specified below.

This page describes the active Python implementation. Conventions from
other programs, including the Scilab code, should not be assumed to match
without checking operator definitions and prefactors.


## Parameter units

| Attribute or argument | Unit | Meaning |
|---|---|---|
| `Spins` | Dimensionless | Spin quantum numbers, for example `0.5`, `1`, or `2.5`. |
| `ReadoutSpin` | Dimensionless index | Zero-based index of the spin used for readout and RF excitation. |
| `gfactorvector` | Dimensionless | Signed g-factors used with the Bohr magneton. |
| `B`, `BTip` | T | Cartesian magnetic-field vectors in the order `[x, y, z]`. |
| `T` | K | Bath temperature. See the zero-temperature note below. |
| `Dvector`, `EvectorX`, `EvectorY` | meV | Coefficients of the quadratic anisotropy terms. |
| `Jvector` | meV | Diagonal pair-exchange coefficients `[Jxx, Jyy, Jzz]`. |
| `Jxy`, `Jxz`, `Jyz`, `Jyx`, `Jzx`, `Jzy` | meV | Off-diagonal pair-exchange coefficients. |
| `Avector` | meV | Diagonal pairwise hyperfine coefficients `[Axx, Ayy, Azz]`. |
| `Qvector` | meV | Quadrupolar Hamiltonian coefficients `[Qxx, Qyy, Qzz]`, not physical quadrupole moments. |
| `Stevens` | meV | Coefficients multiplying the package's dimensionless Stevens operators. |
| `DipoleVector` | meV | Dipolar coupling strengths `D0`, one value per spin pair. |
| `AtomPositions` | Dimensionless | Cartesian lattice coordinates, not positions directly in nm. |
| `latticeLengthXY` | nm | Three componentwise lattice spacings `[ax, ay, az]`, despite the historical attribute name. |
| `V_DC`, `V_array` | mV | Bias voltage and voltage-sweep values. |
| `G_st` | A/V = S | Conductance scale used to normalize tip–sample scattering. |
| `G_ss` | A/V = S | Conductance-equivalent substrate-scattering parameters, one per spin. |
| `G_tt` | Dimensionless | Switch: `0` disables and `1` enables tip–tip scattering. It is not a conductance. |
| `U` | Dimensionless | Potential-scattering amplitude relative to the spin-exchange scattering amplitude. |
| `b0` | Dimensionless | Fraction assigned to the spin-independent background channel. |
| `TipPolarization`, `SamplePolarization` | Dimensionless | Cartesian polarization vectors. Zero denotes an unpolarized reservoir; vector norm one denotes full polarization. |
| `J` | Dimensionless | Signed substrate exchange per spin in the legacy spin-code normalization; negative denotes antiferromagnetic exchange. Not `Jvector`. |
| `J_tip` | Dimensionless | Signed virtual tip exchange attribute of `Spinsys`used in third-order functions; default zero. Independent phenomenological parameter, not inferred from `G_tt`. |
| `w`, `lt` | meV | Electronic bandwidth and positive lifetime broadening used in third-order spectroscopy. |
| `energy_step` | meV | Internal integration spacing for third-order calculations; verify convergence by reducing it. |
| `B1` | T | Peak amplitude of the linearly oscillating RF magnetic field in the ESR-rate model. |
| `T2` | ns | Total transverse coherence time used in the ESR-rate model. |
| `T_phi` | ns | Phenomenological local pure-dephasing time, not total `T2`. Positive infinity disables this contribution. |
| `freq`, `freq_array`, `f_RF` | GHz | Ordinary frequency, not angular frequency. |
| `RabiRate` | GHz | Ordinary Rabi frequency used in coherent-control routines. The ESR-rate routines instead use `B1`. |
| `lw` | GHz | Lorentzian full width at half maximum in `ESR.calc_FreqSweep`. |
| `time_array`, `pulsewidth_array`, `tau` | ns | Evolution times, pulse durations, and delays in Lindblad and coherent-control routines. |

All Zeeman terms use the Bohr magneton. The code does not automatically
distinguish electronic and nuclear spins. When modeling a nucleus using
a nuclear g-factor defined with the nuclear magneton, use the equivalent
coefficient:

```text
g_effective = g_nuclear * mu_N / mu_B
```


## Calculated quantities

| Quantity | Unit | Convention |
|---|---|---|
| `H` | meV | Hamiltonian in the product-spin basis. |
| `E_sort` | meV | Sorted eigenenergies before subtracting the ground-state energy. |
| `E_All` | meV | Sorted eigenenergies relative to the ground state. |
| `E_All_inGHz` | GHz | Relative eigenenergies divided by Planck's constant. |
| `eigVectors` | Dimensionless | Normalized eigenvectors stored as columns. |
| `Sx`, `Sy`, `Sz` | Dimensionless | Spin operators with the factor of `hbar` removed. |
| `Populations`, `pop_0` | Dimensionless | Eigenstate occupation probabilities. |
| Density matrices | Dimensionless | Trace-one matrices. |
| `calc_RateIntegrals(...)` | meV | Electron phase-space integrals, not transition rates. |
| `MatrixST`, `MatrixTS`, `MatrixSS`, `MatrixTT` | Dimensionless | Squared scattering amplitudes before conductance and phase-space prefactors. |
| `Rates`, `Rates_Summed`, `ESRRates` | 1/s | Population-transition or scattering-event rates. |
| `K` | 1/s | Classical population-rate generator. |
| `T_state_tot`, `T_state_ss` | s | Inverse total or substrate-only inelastic escape rates of individual eigenstates. |
| `elastic_dephasing_rates` | 1/s | Pairwise coherence-decay rates, not population-transfer rates. |
| `collapse_ops`, `elastic_collapse_ops` | 1/sqrt(ns) | Collapse operators prepared for evolution with time in ns. |
| `H_RF` | rad/ns | Rotating-frame generator used by the time-evolution solver. |
| Tunnel currents, `I_off` | A | Signed currents. |
| Unnormalized rate-based ESR signal | A | Current difference `I_RF_on - I_RF_off`. |
| Unnormalized second-order `dI/dV` | A/V = S | Differential conductance. |
| `Mag`, `Mag_tot` | Dimensionless | Spin expectation values, not magnetic moments in SI units. |
| Fourier frequencies | GHz | When the supplied sampling times are in ns. |


## Physical constants and conversions

### Ordinary frequency versus angular frequency

Public frequency inputs are ordinary frequencies:

```text
omega [rad/ns] = 2*pi * f [GHz]
```

For time evolution, the Hamiltonian is converted from energy into
an angular-frequency generator:

```python
H_generator = H * meVtoGHzConversion * 2 * np.pi
```

Do not apply this conversion twice. In particular, `H_RF` is already
in `rad/ns`.

Bath rates are converted separately:

```text
rate [1/ns] = rate [1/s] * 1e-9
```

Consequently, an inelastic collapse operator is constructed as:

```text
L_ij = sqrt(rate_ij [1/s] * 1e-9) * |j><i|
```

Rates are multiplied by `1e-9`; collapse amplitudes are multiplied
by its square root.

## Spin and Hamiltonian conventions

### Spin operators

The matrices represent dimensionless angular momentum:

```text
Sz |m> = m |m>
[Sx, Sy] = i Sz
```

For spin one-half:

```text
Sx = sigma_x / 2
Sy = sigma_y / 2
Sz = sigma_z / 2
```

Do not insert an additional factor of `hbar` into Hamiltonian terms
that already use the package constants and coefficients.

### Zeeman term convention

The external-field contribution is:

```text
H_Z = -sum_i g_i * muB * B.S_i
```
"""Tests for Lindblad time evolution."""

import numpy as np
from qutip import basis

from spinfinity import SpinSys, bath, hamil, lind
from spinfinity import helperfunctions as hf


def test_spin_half_time_evolution_decay_matches_tunneling_T1():
    """Verify the fitted Lindblad decay time against the tunneling rates.

    Notes
    -----
    Reproduce the tunnel-junction setup in TImeEvolution_Spin_Half.ipynb.
    Initialize the excited state and use unfiltered inelastic collapse operators.
    Rates are in 1/s and evolution times are in ns. The fitted observable is
    2<Sz>, as returned by calc_TimeEvolution.
    """
    spin = SpinSys([0.5])
    spin.gfactorvector = [2.0]
    spin.B = [0.0, 0.0, 0.5]  # T
    spin.BTip = [0.0, 0.0, 0.0]  # T
    spin.T = 1.0  # K

    spin.TipPolarization = [0.0, 0.0, 0.5]
    spin.SamplePolarization = [0.0, 0.0, 0.0]
    spin.U = np.array([0.0])
    spin.G_ss = [2e-9]  # S
    spin.G_st = 0.2e-9  # S
    spin.G_tt = 1  # Dimensionless switch
    spin.V_DC = 0.1  # mV
    spin.b0 = 0.0

    hamil.calc_EigEnergies(spin, rebuildH=True)
    bath.calc_Rates(spin)

    rate_up = spin.Rates_Summed[0, 1]
    rate_down = spin.Rates_Summed[1, 0]
    T1_expected = 1e9 / (rate_up + rate_down)  # ns

    c_ops_T1 = lind.calc_collapseOperators(spin, 
                                           threshold_percent=None, 
                                           recalcRates=False)
    spin.collapse_ops = c_ops_T1
    time_array = np.linspace(0.0, 5 * T1_expected, 500)

    result = lind.calc_TimeEvolution(
        spin,
        psi_0=basis(2, 1),
        time_array=time_array,
        spin_idx_array=[0],
        Saxis="z",
    )

    _, _, parameters, _ = hf.fit_exponential(result.times, result.expect[0])
    T1_fit = parameters[1]

    assert np.all(np.isfinite(parameters))
    np.testing.assert_allclose(
        T1_fit,
        T1_expected,
        rtol=1e-5,
        atol=0.0,
        err_msg="Lindblad decay does not match T1 from the tunneling rates.",
    )


def test_two_spin_time_evolution_fft_matches_J_perp():
    """Verify the exchange-oscillation Fourier peak against J_perp.

    Notes
    -----
    Use two spin-1/2 particles with identical Zeeman energies, initialized in
    |down, up>. At zero detuning the flip-flop frequency is |J_perp|/h.
    Exchange inputs below are expressed as energy/h in GHz and converted to meV.
    Evolution times are in ns, so Fourier frequencies are in GHz.
    The comparison allows half an FFT bin for finite frequency resolution.
    """
    spin = SpinSys([0.5, 0.5])
    spin.gfactorvector = [2.0, 2.0]
    spin.B = [0.0, 0.0, 0.5]  # T
    spin.BTip = [0.0, 0.0, 0.0]  # T: equal Zeeman energies
    spin.T = 1.0  # K

    J_perp_GHz = 0.02
    J_parallel_GHz = 0.02
    exchange_meV = (
        np.array([J_perp_GHz, J_perp_GHz, J_parallel_GHz]) * spin.GHztomeVConversion
    )
    hamil.assign_ExchangeInteraction(spin, [(0, 1, exchange_meV)])
    spin.DipoleVector[:] = 0.0

    spin.TipPolarization = [0.0, 0.0, 0.5]
    spin.SamplePolarization = [0.0, 0.0, 0.0]
    spin.U = np.array([0.0, 0.0])
    spin.G_ss = [0.5e-9, 0.5e-9]  # S
    spin.G_st = 0.05e-9  # S
    spin.G_tt = 1
    spin.V_DC = 0.0  # mV
    spin.b0 = 0.0

    hamil.calc_EigEnergies(spin, rebuildH=True)
    bath.calc_Rates(spin)
    c_ops_T1 = lind.calc_collapseOperators(spin, 
                                            threshold_percent=None, 
                                            recalcRates=False)
    spin.collapse_ops = c_ops_T1

    result = lind.calc_TimeEvolution(
        spin,
        psi_0=basis(4, 2),  # |down, up>
        time_array=np.linspace(0.0, 2000.0, 1000),
        spin_idx_array=[0],
        Saxis="z",
    )

    signal = np.real(result.expect[0])
    signal = signal - np.mean(signal)
    frequencies, amplitudes = hf.calc_fft(
        times=result.times,
        signal=signal,
    )

    # Ignore the zero-frequency component and select the strongest peak.
    peak_index = 1 + np.argmax(np.abs(amplitudes[1:]))
    frequency_peak = frequencies[peak_index]
    frequency_resolution = frequencies[1] - frequencies[0]

    np.testing.assert_allclose(
        frequency_peak,
        abs(J_perp_GHz),
        rtol=0.0,
        atol=frequency_resolution / 2,
        err_msg="The exchange-oscillation FFT peak does not match J_perp/h.",
    )
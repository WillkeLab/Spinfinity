"""Scientific checks for coherent-control simulations."""

import numpy as np
from qutip import Qobj
from scipy.optimize import curve_fit
from scipy.linalg import expm

from spinfinity import CC, SpinSys, bath, hamil, lind
from spinfinity import helperfunctions as hf


def test_spin_half_rabi_resonant_decay_and_frequency():
    """Verify resonant Rabi decay, frequency, and the complete damped trace.

    Notes
    -----
    T1 is calculated from both inelastic transition rates. Two explicitly
    specified pure-dephasing times are tested with the same SpinSys.
    Elastic collapse operators are excluded to avoid additional dephasing.
    Times are in ns, frequencies in GHz, and spin expectations dimensionless.
    """
    spin = SpinSys([0.5])
    spin.B = [0.0, 0.0, 0.5]
    spin.BTip = [1e-9, 0.0, 0.0]
    spin.gfactorvector = [2.0]
    spin.tip = "f"

    spin.T = 0.5  # K; both upward and downward transitions are appreciable
    spin.V_DC = 0.0  # mV
    spin.G_st = 1e-10  # S
    spin.G_ss = [1e-9]  # S
    spin.G_tt = 1
    spin.U = np.array([0.0])
    spin.b0 = 0.0
    spin.TipPolarization = [0.0, 0.0, 0.0]
    spin.SamplePolarization = [0.0, 0.0, 0.0]

    hamil.calc_EigEnergies(spin, rebuildH=True)
    bath.calc_Rates(spin)

    rate_up = spin.Rates_Summed[0, 1]  # 1/s
    rate_down = spin.Rates_Summed[1, 0]  # 1/s
    assert np.isfinite(rate_up + rate_down)
    assert rate_up > 0 and rate_down > 0
    T1 = 1e9 / (rate_up + rate_down)  # ns, not a single-state lifetime

    relaxation_ops = lind.calc_collapseOperators(
        spin, threshold_percent=None, recalcRates=False
    )
    spin.psi0 = Qobj(spin.eigVectors[:, 0])
    spin.f_RF = float(spin.E_All_inGHz[1] - spin.E_All_inGHz[0])
    spin.RabiRate = 10e-3  # GHz

    CC.RotFrameSecular(spin, spin.f_RF)
    Sz = Qobj(spin.Sz_rot)
    solver_options = {
        "atol": 1e-11,
        "rtol": 1e-11,
        "store_states": False,
        "progress_bar": "",
    }

    gamma1 = 1 / T1  # 1/ns
    omega = 2 * np.pi * spin.RabiRate  # rad/ns
    z_equilibrium = (rate_down - rate_up) / (2 * (rate_up + rate_down))
    z_initial = 0.5

    for T_phi in (200.0, 600.0):
        # This normalization gives coherence decay at exactly 1/T_phi.
        spin.collapse_ops = relaxation_ops + [
            lind.calc_dephasingOperator(spin, spin_idx=0, T_phi=T_phi)
        ]

        gamma2 = gamma1 / 2 + 1 / T_phi
        T_Rabi_expected = 1 / (3 / (4 * T1) + 1 / (2 * T_phi))
        gamma_rabi = 1 / T_Rabi_expected
        assert omega > abs(gamma1 - gamma2) / 2, "Underdamped motion required."
        omega_damped = np.sqrt(omega**2 - (gamma1 - gamma2) ** 2 / 4)
        frequency_expected = omega_damped / (2 * np.pi)

        pulsewidths = np.linspace(0.0, 4 * T_Rabi_expected, 2401)
        times, result = CC.calc_RabiOscillation(
            spin, pulsewidths, e_ops=[Sz], solver_options=solver_options
        )
        signal = np.asarray(result.expect[0])
        np.testing.assert_array_equal(times, pulsewidths)
        assert np.all(np.isfinite(signal))

        # Exact resonant Bloch solution for ground-state preparation.
        z_steady = z_equilibrium * gamma1 * gamma2 / (gamma1 * gamma2 + omega**2)
        cosine_coefficient = z_initial - z_steady
        initial_derivative = -gamma1 * (z_initial - z_equilibrium)
        sine_coefficient = (
            initial_derivative + gamma_rabi * cosine_coefficient
        ) / omega_damped
        expected = z_steady + np.exp(-gamma_rabi * times) * (
            cosine_coefficient * np.cos(omega_damped * times)
            + sine_coefficient * np.sin(omega_damped * times)
        )
        np.testing.assert_allclose(
            signal,
            expected,
            rtol=0,
            atol=1e-7,
            err_msg=f"Incorrect resonant trace at T_phi={T_phi} ns.",
        )

        _, _, parameters, errors = hf.fit_oscillation(times, signal)
        amplitude_fit, T_Rabi_fit, frequency_fit, phase_fit, offset_fit = parameters
        assert np.all(np.isfinite(parameters))
        assert np.all(np.isfinite(errors))
        assert T_Rabi_fit > 0

        np.testing.assert_allclose(T_Rabi_fit, T_Rabi_expected, rtol=1e-4, atol=0)
        np.testing.assert_allclose(
            abs(frequency_fit), frequency_expected, rtol=1e-5, atol=0
        )

        fitted_trace = offset_fit + amplitude_fit * np.exp(-times / T_Rabi_fit) * (
            np.cos(2 * np.pi * frequency_fit * times + phase_fit)
        )
        relative_fit_rms = np.sqrt(np.mean((signal - fitted_trace) ** 2)) / np.ptp(
            signal
        )
        assert relative_fit_rms < 1e-6, f"Poor fit at T_phi={T_phi} ns."


def test_spin_half_rabi_detuned_frequency_amplitude_and_offset():
    """Verify ideal detuned Rabi fits and complete analytical time traces.

    Notes
    -----
    Two drive strengths and negative, zero, and positive detunings are tested.
    The initial state is the ground eigenstate and collapse operators are absent.
    Fits use an undamped cosine with free amplitude, frequency, phase, and offset.
    The amplitude is the cosine amplitude of Sz, not peak-to-peak contrast.
    """

    def cosine_model(time, amplitude, frequency, phase, offset):
        return offset + amplitude * np.cos(2 * np.pi * frequency * time + phase)

    spin = SpinSys([0.5])
    spin.B = [0.0, 0.0, 0.5]
    spin.BTip = [1e-9, 0.0, 0.0]
    spin.gfactorvector = [2.0]
    spin.tip = "f"
    hamil.calc_EigEnergies(spin, rebuildH=True)

    spin.psi0 = Qobj(spin.eigVectors[:, 0])
    spin.collapse_ops = []
    f_res = float(spin.E_All_inGHz[1] - spin.E_All_inGHz[0])
    spin.f_RF = f_res
    CC.RotFrameSecular(spin, spin.f_RF)
    Sz = Qobj(spin.Sz_rot)

    pulsewidths = np.linspace(0.0, 400.0, 1601)  # ns
    solver_options = {
        "atol": 1e-11,
        "rtol": 1e-11,
        "store_states": False,
        "progress_bar": "",
    }

    for rabi_rate in (10e-3, 25e-3):  # GHz
        spin.RabiRate = rabi_rate
        detunings = rabi_rate * np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        frequencies = f_res + detunings
        times, returned_frequencies, signal_map = CC.calc_RabiDetuningMap(
            spin,
            pulsewidths,
            frequencies,
            e_ops=Sz,
            solver_options=solver_options,
            n_jobs=1,
        )
        np.testing.assert_array_equal(times, pulsewidths)
        np.testing.assert_array_equal(returned_frequencies, frequencies)
        assert signal_map.shape == (len(frequencies), len(times))
        assert np.all(np.isfinite(signal_map))

        # Opposite detunings must give the same longitudinal spin signal.
        np.testing.assert_allclose(signal_map, signal_map[::-1], rtol=0, atol=1e-7)

        for index, detuning in enumerate(detunings):
            signal = signal_map[index]
            frequency_expected = np.hypot(rabi_rate, detuning)
            amplitude_expected = 0.5 * rabi_rate**2 / frequency_expected**2
            offset_expected = 0.5 * detuning**2 / frequency_expected**2
            expected = offset_expected + amplitude_expected * np.cos(
                2 * np.pi * frequency_expected * times
            )
            np.testing.assert_allclose(signal, expected, rtol=0, atol=1e-7)

            # Also exercise the single-trace API away from resonance.
            spin.f_RF = float(frequencies[index])
            single_times, single_result = CC.calc_RabiOscillation(
                spin, pulsewidths, e_ops=[Sz], solver_options=solver_options
            )
            np.testing.assert_array_equal(single_times, times)
            np.testing.assert_allclose(
                single_result.expect[0], signal, rtol=0, atol=1e-7
            )

            # FFT-based fit initialization, independent of the Rabi formula.
            dt = times[1] - times[0]
            fft_size = 8 * len(times)
            spectrum = np.fft.rfft(signal - np.mean(signal), n=fft_size)
            fft_frequencies = np.fft.rfftfreq(fft_size, d=dt)
            peak_index = np.argmax(np.abs(spectrum[1:])) + 1
            initial = [
                np.ptp(signal) / 2,
                fft_frequencies[peak_index],
                np.angle(spectrum[peak_index]),
                np.mean(signal),
            ]
            bounds = (
                [0, 0, -np.inf, -np.inf],
                [np.inf, 0.5 / dt, np.inf, np.inf],
            )
            parameters, covariance = curve_fit(
                cosine_model,
                times,
                signal,
                p0=initial,
                bounds=bounds,
                maxfev=20000,
            )
            amplitude_fit, frequency_fit, _, offset_fit = parameters
            assert np.all(np.isfinite(parameters))
            assert np.all(np.isfinite(covariance))

            np.testing.assert_allclose(
                frequency_fit, frequency_expected, rtol=1e-5, atol=0
            )
            np.testing.assert_allclose(
                amplitude_fit, amplitude_expected, rtol=1e-5, atol=1e-7
            )
            np.testing.assert_allclose(
                offset_fit, offset_expected, rtol=1e-5, atol=1e-7
            )

            fitted_trace = cosine_model(times, *parameters)
            relative_fit_rms = np.sqrt(np.mean((signal - fitted_trace) ** 2)) / np.ptp(
                signal
            )
            assert relative_fit_rms < 1e-6, "Poor detuned Rabi fit."

        # Restore the rotating-frame cache before the next drive comparison.
        spin.f_RF = f_res
        CC.RotFrameSecular(spin, spin.f_RF)


def test_spin_half_relaxometry_decay_matches_tunneling_T1():
    """Verify the fitted relaxometry time against both tunneling rates.

    Notes
    -----
    Reproduce the spin-1/2 relaxometry example at zero bias. Both pure-dephasing
    times must give the same longitudinal relaxation time. Readout and pure
    dephasing are aligned with the effective static field.
    Rates are in 1/s, times in ns, and the drive frequency in GHz.
    """
    spin = SpinSys([0.5])
    spin.gfactorvector = [2.0]
    spin.B = [0.0, 0.0, 0.5]  # T
    spin.BTip = [1e-9, 0.0, 0.0]  # T
    spin.tip = "f"

    spin.T = 0.2  # K
    spin.V_DC = 0.0  # mV
    spin.TipPolarization = [0.0, 0.0, 0.5]
    spin.SamplePolarization = [0.0, 0.0, 0.0]
    spin.U = np.array([0.0])
    spin.G_st = 0.1e-9  # S
    spin.G_ss = [1e-9]  # S
    spin.G_tt = 1
    spin.b0 = 0.0

    hamil.calc_EigEnergies(spin, rebuildH=True)
    bath.calc_Rates(spin)

    rate_up = spin.Rates_Summed[0, 1]  # ground -> excited, 1/s
    rate_down = spin.Rates_Summed[1, 0]  # excited -> ground, 1/s
    assert np.all(np.isfinite([rate_up, rate_down]))
    assert rate_up > 0 and rate_down > 0
    T1_expected = 1e9 / (rate_up + rate_down)  # ns

    relaxation_ops = lind.calc_collapseOperators(
        spin, threshold_percent=None, recalcRates=False
    )
    spin.f_RF = float(spin.E_All_inGHz[1] - spin.E_All_inGHz[0])
    spin.RabiRate = 50e-3  # GHz
    CC.RotFrameSecular(spin, spin.f_RF)
    Sz = Qobj(spin.Sz_rot)

    # The ground state has Sz = +1/2 in the package's Zeeman convention.
    z_equilibrium = (rate_down - rate_up) / (2 * (rate_up + rate_down))
    delays = np.linspace(0.0, 5 * T1_expected, 500)  # ns

    for T_phi in (200.0, 50.0):  # ns, as in the notebook
        spin.collapse_ops = relaxation_ops + [
            lind.calc_dephasingOperator(spin, spin_idx=0, T_phi=T_phi)
        ]
        lind.initialize_thermal(spin)

        times, result = CC.calc_Relaxometry(spin, delays, e_ops=[Sz])
        signal = np.asarray(result.expect[0])
        np.testing.assert_array_equal(times, delays)
        np.testing.assert_array_equal(result.times, delays)
        assert np.all(np.isfinite(signal))
        assert np.ptp(signal) > 0.1, "The sequence must produce measurable decay."

        _, _, parameters, errors = hf.fit_exponential(times, signal)
        amplitude_fit, T1_fit, offset_fit = parameters
        assert np.all(np.isfinite(parameters))
        assert np.all(np.isfinite(errors))
        assert T1_fit > 0

        np.testing.assert_allclose(
            T1_fit,
            T1_expected,
            rtol=1e-5,
            atol=0,
            err_msg=f"Incorrect relaxometry T1 at T_phi={T_phi} ns.",
        )
        np.testing.assert_allclose(offset_fit, z_equilibrium, rtol=0, atol=1e-7)

        # Use the actual post-pulse signal, not an ideal instantaneous pi pulse.
        expected = z_equilibrium + (signal[0] - z_equilibrium) * np.exp(
            -times / T1_expected
        )
        np.testing.assert_allclose(signal, expected, rtol=0, atol=1e-7)

        fitted = offset_fit + amplitude_fit * np.exp(-times / T1_fit)
        relative_fit_rms = np.sqrt(np.mean((signal - fitted) ** 2)) / np.ptp(signal)
        assert relative_fit_rms < 1e-6, f"Poor fit at T_phi={T_phi} ns."


def test_spin_half_ramsey_coherence_and_detuning():
    """Verify finite-pulse Ramsey traces, coherence times, and frequencies.

    Notes
    -----
    Compare the simulated sequence with an independent Bloch-equation solution
    including detuning and dissipation during both nominal pi/2 pulses.
    Nonzero-detuning fits include a longitudinal background with fixed T1.
    Zero detuning is checked against the full trace without an oscillation fit.
    Elastic collapse operators and additional frequency noise are excluded.
    Times are in ns, frequencies in GHz, and bath rates in 1/s.

    References
    ----------
    Ithier et al., Phys. Rev. B 72, 134519 (2005), Eqs. (11), (12), and (58).
    https://doi.org/10.1103/PhysRevB.72.134519
    Nicot et al., C. R. Chimie 11, 506-514 (2008), Eq. (2).
    https://doi.org/10.1016/j.crci.2007.11.004
    """
    spin = SpinSys([0.5])
    spin.gfactorvector = [2.0]
    spin.B = [0.0, 0.0, 0.5]  # T
    spin.BTip = [1e-9, 0.0, 0.0]  # T
    spin.tip = "f"

    spin.T = 0.2  # K
    spin.V_DC = 0.0  # mV
    spin.TipPolarization = [0.0, 0.0, 0.5]
    spin.SamplePolarization = [0.0, 0.0, 0.0]
    spin.U = np.array([0.0])
    spin.G_st = 0.1e-9  # S
    spin.G_ss = [1e-9]  # S
    spin.G_tt = 1
    spin.b0 = 0.0

    hamil.calc_EigEnergies(spin, rebuildH=True)
    bath.calc_Rates(spin)
    rate_up = spin.Rates_Summed[0, 1]
    rate_down = spin.Rates_Summed[1, 0]
    assert np.all(np.isfinite([rate_up, rate_down]))
    assert rate_up > 0 and rate_down > 0

    T1 = 1e9 / (rate_up + rate_down)  # ns
    gamma1 = 1 / T1
    z_equilibrium = (rate_down - rate_up) / (2 * (rate_up + rate_down))
    relaxation_ops = lind.calc_collapseOperators(
        spin, threshold_percent=None, recalcRates=False
    )

    f_res = float(spin.E_All_inGHz[1] - spin.E_All_inGHz[0])
    spin.f_RF = f_res
    spin.RabiRate = 50e-3  # GHz
    omega = 2 * np.pi * spin.RabiRate  # rad/ns
    pulse_duration = 1 / (4 * spin.RabiRate)  # ns
    CC.RotFrameSecular(spin, spin.f_RF)
    Sz = Qobj(spin.Sz_rot)
    detunings = (-50e-3, 0.0, 20e-3, 50e-3)  # GHz

    def ramsey_model(time, a_cos, a_sin, offset, drift, T2, frequency):
        return (
            offset
            + drift * np.exp(-time / T1)
            + np.exp(-time / T2)
            * (
                a_cos * np.cos(2 * np.pi * frequency * time)
                + a_sin * np.sin(2 * np.pi * frequency * time)
            )
        )

    for T_phi in (200.0, 50.0):  # ns
        spin.collapse_ops = relaxation_ops + [
            lind.calc_dephasingOperator(spin, spin_idx=0, T_phi=T_phi)
        ]
        lind.initialize_thermal(spin)
        T2_expected = 1 / (gamma1 / 2 + 1 / T_phi)
        gamma2 = 1 / T2_expected
        delays = np.linspace(0.0, 4 * T2_expected, 601)
        signals = []

        # Independent thermal spin expectation for this two-level Hamiltonian.
        boltzmann_ratio = np.exp(-(spin.E_All[1] - spin.E_All[0]) / (spin.kB * spin.T))
        z_initial = (1 - boltzmann_ratio) / (2 * (1 + boltzmann_ratio))
        initial = np.array([0.0, 0.0, z_initial, 1.0])

        for detuning in detunings:
            spin.f_RF = f_res + detuning
            times, values = CC.calc_Ramsey_Parallel(spin, delays, e_ops=[Sz], n_jobs=1)
            np.testing.assert_array_equal(times, delays)
            assert values.shape == (len(delays), 1)
            signal = values[:, 0]
            assert np.all(np.isfinite(signal))
            signals.append(signal)
            context = f"T_phi={T_phi} ns, detuning={detuning} GHz"

            # Affine Bloch generator acting on [<Sx>, <Sy>, <Sz>, 1].
            delta = 2 * np.pi * detuning  # rad/ns
            generator = np.array(
                [
                    [-gamma2, -delta, 0.0, 0.0],
                    [delta, -gamma2, -omega, 0.0],
                    [0.0, omega, -gamma1, gamma1 * z_equilibrium],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )
            pulse = expm(generator * pulse_duration)
            prepared = pulse @ initial

            # Apply the second pulse to the analytic free-evolution components.
            offset = pulse[2, 2] * z_equilibrium + pulse[2, 3]
            drift = pulse[2, 2] * (prepared[2] - z_equilibrium)
            a_cos = pulse[2, 0] * prepared[0] + pulse[2, 1] * prepared[1]
            a_sin = -pulse[2, 0] * prepared[1] + pulse[2, 1] * prepared[0]
            expected = ramsey_model(
                times, a_cos, a_sin, offset, drift, T2_expected, detuning
            )
            np.testing.assert_allclose(
                signal, expected, rtol=0, atol=1e-7, err_msg=context
            )

            # At zero detuning a separate oscillation frequency is not identifiable.
            if detuning == 0:
                continue

            # The simple oscillation fit is used only to initialize the full model.
            _, _, guess, _ = hf.fit_oscillation(times, signal)
            amplitude0, T20, frequency0, phase0, offset0 = guess
            fit_initial = [
                amplitude0 * np.cos(phase0),
                -amplitude0 * np.sin(phase0),
                offset0,
                0.0,
                T20,
                abs(frequency0),
            ]
            bounds = (
                [-np.inf, -np.inf, -np.inf, -np.inf, 0.0, 0.0],
                [np.inf, np.inf, np.inf, np.inf, np.inf, 0.5 / (times[1] - times[0])],
            )
            parameters, covariance = curve_fit(
                ramsey_model,
                times,
                signal,
                p0=fit_initial,
                bounds=bounds,
                maxfev=20000,
            )
            assert np.all(np.isfinite(parameters))
            assert np.all(np.isfinite(covariance))
            T2_fit, frequency_fit = parameters[4:]
            np.testing.assert_allclose(
                T2_fit, T2_expected, rtol=1e-4, atol=0, err_msg=context
            )
            np.testing.assert_allclose(
                frequency_fit, abs(detuning), rtol=1e-5, atol=0, err_msg=context
            )

            # Fits use a positive frequency, reversing the sine coefficient at -delta.
            expected_coefficients = [a_cos, np.sign(detuning) * a_sin, offset, drift]
            np.testing.assert_allclose(
                parameters[:4],
                expected_coefficients,
                rtol=1e-4,
                atol=1e-7,
                err_msg=context,
            )
            fitted = ramsey_model(times, *parameters)
            relative_fit_rms = np.sqrt(np.mean((signal - fitted) ** 2)) / np.ptp(signal)
            assert relative_fit_rms < 1e-6, context

        np.testing.assert_allclose(signals[0], signals[-1], rtol=0, atol=1e-7)

    spin.f_RF = f_res
    CC.RotFrameSecular(spin, spin.f_RF)


def test_c13_eseem_strong_drive_matches_ideal_signals():
    """Compare relaxation-free Hahn and three-pulse ESEEM with ideal envelopes.

    Notes
    -----
    Uses the C13 example's 0.5 T field at 45 degrees and hyperfine
    principal values [2, 2, 8] MHz. The electron is fully polarized
    along the field and the nucleus is unpolarized, not thermalized.
    No collapse operators are included.

    The 1 GHz Rabi frequency is a numerical hard-pulse control within
    the rotating-wave and electron-secular approximations. Absolute
    tolerances refer to normalized envelopes and allow finite-pulse
    deviations; they are not solver precision targets.
    """
    spin = SpinSys([0.5, 0.5])
    spin.ReadoutSpin = 0

    B0 = 0.5  # T
    theta = np.deg2rad(45.0)
    n = np.array([np.sin(theta), 0.0, np.cos(theta)])
    spin.B = B0 * n
    spin.BTip = [0.0, 0.0, 0.0]

    ge = 2.0023
    gamma_n = 10.7084e-3  # GHz/T, ordinary gyromagnetic ratio
    spin.gfactorvector = [
        ge,
        gamma_n / (spin.muB * spin.meVtoGHzConversion),
    ]
    A_perp = 2.0e-3  # GHz
    A_parallel = 8.0e-3  # GHz
    spin.Avector[0] = np.array([A_perp, A_perp, A_parallel]) * spin.GHztomeVConversion
    hamil.calc_EigEnergies(spin, rebuildH=True)

    rho_e = np.array([[1 + n[2], n[0] - 1j * n[1]], [n[0] + 1j * n[1], 1 - n[2]]]) / 2
    spin.psi0 = Qobj(np.kron(rho_e, np.eye(2) / 2))
    spin.collapse_ops = []
    spin.f_RF = ge * spin.muB * B0 * spin.meVtoGHzConversion
    spin.RabiRate = 1.0  # GHz

    CC.RotFrameSecular(spin, spin.f_RF)
    readout = [Qobj(spin.Sz_rot)]

    # Independent ideal-pulse prediction from the physical parameters.
    nu_N = gamma_n * B0
    A_eff = A_parallel * np.cos(theta) ** 2 + A_perp * np.sin(theta) ** 2
    B_hf = abs((A_parallel - A_perp) * np.sin(theta) * np.cos(theta))
    nu_plus = np.hypot(-nu_N + A_eff / 2, B_hf / 2)
    nu_minus = np.hypot(-nu_N - A_eff / 2, B_hf / 2)
    k = (nu_N * B_hf / (nu_plus * nu_minus)) ** 2

    delays = np.linspace(0.0, 2000.0, 201)  # ns
    sine_plus = np.sin(np.pi * nu_plus * delays)
    sine_minus = np.sin(np.pi * nu_minus * delays)
    hahn_expected = 1 - 2 * k * (sine_plus * sine_minus) ** 2

    _, hahn = CC.calc_HahnEcho_pluspihalf_Parallel(
        spin, tau_array=delays, e_ops=readout, n_jobs=1
    )

    # The final readout gives <Sz_rot> = V2 / 2.
    np.testing.assert_allclose(
        2 * hahn[:, 0],
        hahn_expected,
        rtol=0,
        atol=1e-2,
        err_msg="Hahn ESEEM differs from the ideal-pulse envelope.",
    )

    tau = 1 / (2 * nu_minus)  # ns; strong nu_plus modulation
    stimulated_expected = 1 - k / 4 * (
        (1 - np.cos(2 * np.pi * nu_minus * tau))
        * (1 - np.cos(2 * np.pi * nu_plus * (delays + tau)))
        + (1 - np.cos(2 * np.pi * nu_plus * tau))
        * (1 - np.cos(2 * np.pi * nu_minus * (delays + tau)))
    )

    _, stimulated = CC.calc_3pulseESEEM_pluspihalf_PhaseCycled8(
        spin, T_array=delays, tau=tau, e_ops=readout, n_jobs=1
    )

    # The weighted-mean eight-step readout gives <Sz_rot> = V3 / 4.
    np.testing.assert_allclose(
        4 * stimulated[:, 0],
        stimulated_expected,
        rtol=0,
        atol=6e-3,
        err_msg="Three-pulse ESEEM differs from the ideal-pulse envelope.",
    )


def test_n14_eseem_strong_drive_matches_ideal_signals():
    """Compare quadrupole-driven Hahn and three-pulse ESEEM with ideal envelopes.

    Notes
    -----
    Uses S=1/2 and I=1, a 0.5 T field along z, isotropic hyperfine
    coupling A=-6.5 MHz, and quadrupole parameters K=1 MHz and eta=1.
    The conventional quadrupole coupling constant is chi=4*K.

    The electron is fully polarized and the nucleus is unpolarized.
    Collapse operators are absent. The reference includes the one-third
    spectator contribution from the nuclear m_I=0 state.

    The 1 GHz Rabi drive is a numerical hard-pulse control within the
    rotating-wave and electron-secular approximations. Absolute
    tolerances allow finite-pulse errors in the normalized signals.
    Times are in ns and ordinary frequencies in GHz.
    """
    spin = SpinSys([0.5, 1.0])
    spin.ReadoutSpin = 0

    B0 = 0.5  # T
    spin.B = [0.0, 0.0, B0]
    spin.BTip = [0.0, 0.0, 0.0]

    ge = 2.0023
    gamma_n = 3.077e-3  # GHz/T
    spin.gfactorvector = [
        ge,
        gamma_n / (spin.muB * spin.meVtoGHzConversion),
    ]

    A_iso = -6.5e-3  # GHz
    K = 1.0e-3  # GHz
    eta = 1.0
    spin.Avector[0] = np.full(3, A_iso) * spin.GHztomeVConversion
    spin.Qvector[1] = K * np.array([eta - 1, -eta - 1, 2.0]) * spin.GHztomeVConversion
    hamil.calc_EigEnergies(spin, rebuildH=True)

    spin.psi0 = Qobj(np.kron(np.diag([1.0, 0.0]), np.eye(3) / 3))
    spin.collapse_ops = []
    spin.f_RF = ge * spin.muB * B0 * spin.meVtoGHzConversion
    spin.RabiRate = 1.0  # GHz

    CC.RotFrameSecular(spin, spin.f_RF)
    readout = [Qobj(spin.Sz_rot)]

    # Analytical prediction for the mixed m_I=+/-1 nuclear subspace.
    nu_N = gamma_n * B0
    delta = -nu_N + np.array([0.5, -0.5]) * A_iso
    r_nuclear = np.hypot(delta, K * eta)
    nu_plus, nu_minus = 2 * r_nuclear
    k_mix = (K * eta * A_iso / np.prod(r_nuclear)) ** 2

    delays = np.linspace(0.0, 2000.0, 201)  # ns
    sine_plus = np.sin(np.pi * nu_plus * delays)
    sine_minus = np.sin(np.pi * nu_minus * delays)
    hahn_expected = 1 - 4 * k_mix / 3 * (sine_plus * sine_minus) ** 2

    _, hahn = CC.calc_HahnEcho_pluspihalf_Parallel(
        spin, tau_array=delays, e_ops=readout, n_jobs=1
    )
    np.testing.assert_allclose(
        2 * hahn[:, 0],
        hahn_expected,
        rtol=0,
        atol=1e-2,
        err_msg="N14 Hahn ESEEM differs from the ideal-pulse envelope.",
    )

    tau = 1 / (2 * nu_minus)  # ns
    stimulated_expected = 1 - k_mix / 6 * (
        (1 - np.cos(2 * np.pi * nu_minus * tau))
        * (1 - np.cos(2 * np.pi * nu_plus * (delays + tau)))
        + (1 - np.cos(2 * np.pi * nu_plus * tau))
        * (1 - np.cos(2 * np.pi * nu_minus * (delays + tau)))
    )

    _, stimulated = CC.calc_3pulseESEEM_pluspihalf_PhaseCycled8(
        spin, T_array=delays, tau=tau, e_ops=readout, n_jobs=1
    )
    np.testing.assert_allclose(
        4 * stimulated[:, 0],
        stimulated_expected,
        rtol=0,
        atol=1e-2,
        err_msg="N14 three-pulse ESEEM differs from the ideal-pulse envelope.",
    )
import numpy as np
import scipy.constants as const

from spinfinity import ESR, SpinSys, bath, hamil
from spinfinity import helperfunctions as hf


def test_spin_half_phenomenological_esr_analytical_spectrum():
    """Compare the phenomenological ESR spectrum with a spin-1/2 reference.

    Notes
    -----
    The reference uses only field geometry and SI constants, not the package's
    eigenvectors or populations. The tip field is tilted in the y-z plane.
    """
    spin = SpinSys([0.5])
    spin.B = [0.0, 0.0, 0.5]  # T
    spin.BTip = [0.0, 0.02, 0.01]  # T
    spin.gfactorvector = [2.0]
    spin.tip = "f"
    spin.T = 0.5  # K
    lw = 0.02  # GHz; FWHM

    # Independent analytical reference for a ferromagnetic tip.
    B_tip = np.asarray(spin.BTip, dtype=float)
    B_eff = np.asarray(spin.B, dtype=float) + B_tip
    B_magnitude = np.linalg.norm(B_eff)
    direction = B_eff / B_magnitude
    B_parallel = np.dot(B_tip, direction)
    B_perpendicular_sq = np.sum((B_tip - B_parallel * direction) ** 2)

    mu_B = const.physical_constants["Bohr magneton"][0]
    energy_gap = abs(spin.gfactorvector[0]) * mu_B * B_magnitude  # J
    f_expected = energy_gap / const.h * 1e-9  # GHz
    population_difference = np.tanh(
        energy_gap / (2 * const.Boltzmann * spin.T)
    )
    A_expected = (
        population_difference * B_perpendicular_sq / 4 * abs(B_parallel)
    )

    freq_array = np.linspace(
        f_expected - 10 * lw, f_expected + 10 * lw, 2001
    )
    expected_signal = A_expected / (
        1 + 4 * ((freq_array - f_expected) / lw) ** 2
    )

    freqs, signal = ESR.calc_FreqSweep(
        spin,
        freq_array=freq_array,
        AllowPumping=False,
        lw=lw,
        norm=False,
        output=False,
    )

    np.testing.assert_array_equal(freqs, freq_array)
    np.testing.assert_allclose(
        signal,
        expected_signal,
        rtol=1e-10,
        atol=A_expected * 1e-12,
    )


def test_spin_half_esr_rate_amplitude_linewidth_and_resonance():
    """
    Verify the amplitude, FWHM and resonance of a spin-1/2 ESR-rate spectrum.

    Notes
    -----
    This test compares the signed current change with the analytical two-level steady 
    state and the resonance with an independent SI Zeeman expression.
    """
    spin = SpinSys([0.5])
    spin.B = [0, 0, 0.5]
    spin.BTip = [0, 0, 0]
    spin.tip = "f"
    spin.gfactorvector = [2.0]

    spin.T = 0.5
    spin.V_DC = 10.0
    spin.TipPolarization = [0, 0, 0.5]
    spin.SamplePolarization = [0, 0, 0]
    spin.U = np.array([0.0])
    spin.G_st = 0.1e-9
    spin.G_ss = [1e-9]
    spin.G_tt = 1
    spin.b0 = 0.0

    T2 = 50.0  # ns
    B1 = 5e-3  # T
    freq_array = np.linspace(12, 16, 2000)  # GHz

    hamil.build_Hamiltonian(spin)
    hamil.calc_EigEnergies(spin)
    bath.calc_Rates(spin)

    # Independent resonance reference: g * mu_B * |B + BTip| / h.
    mu_B = const.physical_constants["Bohr magneton"][0]
    g = spin.gfactorvector[0]
    B_eff = np.asarray(spin.B) + np.asarray(spin.BTip)
    f_expected = g * mu_B * np.linalg.norm(B_eff) / const.h * 1e-9

    gamma = g * mu_B / const.hbar * 1e-9  # rad/ns/T
    Sx_energy = spin.eigVectors.conj().T @ spin.Sx[0] @ spin.eigVectors
    Omega = gamma * B1 * abs(Sx_energy[0, 1])  # rad/ns

    Gamma_up = spin.Rates_Summed[0, 1]  # 1/s
    Gamma_down = spin.Rates_Summed[1, 0]  # 1/s
    Gamma_total = Gamma_up + Gamma_down
    T1 = 1e9 / Gamma_total  # ns
    assert T2 <= 2 * T1, "The total T2 must not exceed 2*T1."

    # Solve the RF-off two-level populations independently of null_space.
    p_off = np.array([Gamma_down, Gamma_up]) / Gamma_total

    s = Omega**2 * T1 * T2
    FWHM_expected = np.sqrt(1 + s) / (np.pi * T2)  # GHz
    state_currents = const.e * np.sum(spin.Rates[0] - spin.Rates[1], axis=1)
    A_expected = (
        (state_currents[1] - state_currents[0])
        * (p_off[0] - p_off[1])
        / 2
        * s
        / (1 + s)
    )  # A
    assert A_expected != 0, "The chosen system must have a nonzero ESR contrast."

    freqs, signal, I_off = ESR.calc_FreqSweep_ESRrate(
        spin, freq_array=freq_array, B1=B1, T2=T2, norm=False
    )
    np.testing.assert_array_equal(freqs, freq_array)
    assert np.all(np.isfinite(signal)), "The ESR signal contains NaN or inf."

    _, _, popt, _ = hf.fit_fano(freqs, signal)
    A_fit, f_fit, FWHM_fit, C_fit = popt

    np.testing.assert_allclose(
        A_fit, A_expected, rtol=1e-5, atol=0, err_msg="Incorrect ESR amplitude."
    )
    np.testing.assert_allclose(
        FWHM_fit,
        FWHM_expected,
        rtol=1e-5,
        atol=0,
        err_msg="Incorrect ESR FWHM.",
    )
    np.testing.assert_allclose(
        f_fit,
        f_expected,
        rtol=0,
        atol=1e-7,
        err_msg="Incorrect ESR resonance frequency.",
    )

    # Check the complete line as well, not only the fitted parameters.
    expected_signal = A_expected / (1 + 4 * ((freqs - f_expected) / FWHM_expected) ** 2)
    np.testing.assert_allclose(
        signal,
        expected_signal,
        rtol=1e-8,
        atol=abs(A_expected) * 1e-10,
    )
    np.testing.assert_allclose(C_fit, 0, rtol=0, atol=abs(A_expected) * 1e-5)

    I_off_expected = (
        np.dot(p_off, state_currents) + spin.b0 * spin.G_st * spin.V_DC * 1e-3
    )
    np.testing.assert_allclose(I_off, I_off_expected, rtol=1e-10, atol=0)
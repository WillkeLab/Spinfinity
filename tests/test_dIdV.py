import numpy as np
import scipy.constants as const

from spinfinity import dIdV, SpinSys, bath, hamil


"""Scientific checks for second-order spin-1/2 IETS."""

import numpy as np
import scipy.constants as const
from scipy.integrate import quad
from scipy.special import expit

from spinfinity import SpinSys, dIdV, hamil


def _reference_broadened_step(x):
    """Evaluate the broadened step by independent Fermi-function quadrature.

    Parameters
    ----------
    x : float
        Dimensionless excitation-energy argument.

    Returns
    -------
    float
        Dimensionless differential tunneling phase-space factor.

    Notes
    -----
    With f(t) = expit(-t), integrate f(t) * [1-f(t)] * [1-f(t-x)].
    This does not use the closed expression implemented in Spinfinity.
    """
    return quad(
        lambda t: expit(-t) * expit(t) * expit(t - x),
        -np.inf,
        np.inf,
        epsabs=1e-12,
        epsrel=1e-12,
    )[0]


def test_iets_spin_half_zero_temperature():
    """Verify exact zero-temperature thresholds and conductance plateaus."""
    spin = SpinSys([0.5])
    spin.B = [0.0, 0.0, 5.0]
    spin.gfactorvector = [2.0]
    spin.T = 0.0
    spin.G_st = 1e-9
    spin.U = np.array([0.0])
    spin.b0 = 0.0
    spin.TipPolarization = [0.0, 0.0, 0.0]
    spin.SamplePolarization = [0.0, 0.0, 0.0]
    hamil.calc_EigEnergies(spin, rebuildH=True)

    mu_B = const.physical_constants["Bohr magneton"][0]
    gap = 2 * mu_B * 5.0 / const.e * 1e3  # meV
    np.testing.assert_allclose(spin.E_All, [0.0, gap], rtol=1e-12, atol=1e-14)

    # Numerical values of energies in meV and excitation voltages in mV match.
    voltages = gap * np.array([-2.0, -1.01, -0.99, 0.0, 0.99, 1.01, 2.0])
    V, conductance = dIdV.calc_IETS_2nd(spin, V_array=voltages, norm=False)
    expected = spin.G_st * np.array([3.0, 3.0, 1.0, 1.0, 1.0, 3.0, 3.0])

    np.testing.assert_array_equal(V, voltages)
    np.testing.assert_allclose(conductance, expected, rtol=1e-12, atol=0)

    # Use the actual level difference here to exercise the exact half-step,
    # without ambiguity from floating-point rounding of the SI reference.
    thresholds = np.array([-spin.E_All[1], spin.E_All[1]])
    _, at_threshold = dIdV.calc_IETS_2nd(spin, V_array=thresholds, norm=False)
    np.testing.assert_allclose(at_threshold, 2 * spin.G_st, rtol=1e-12, atol=0)


def test_tunneling_broadened_step_reference_and_limits():
    """Verify the broadened step against quadrature, symmetry, and limits."""
    arguments = np.array(
        [
            -100.0,
            -50.0,
            -1.0,
            -1e-3,
            -1e-8,
            0.0,
            1e-8,
            1e-3,
            1.0,
            50.0,
            100.0,
        ]
    )
    expected = np.array([_reference_broadened_step(x) for x in arguments])
    calculated = dIdV.tunnelingBroadenedStepFunction(arguments)

    np.testing.assert_allclose(calculated, expected, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(
        calculated + dIdV.tunnelingBroadenedStepFunction(-arguments),
        1.0,
        rtol=0,
        atol=1e-11,
    )
    np.testing.assert_array_equal(
        dIdV.tunnelingBroadenedStepFunction(np.array([-np.inf, 0.0, np.inf])),
        [1.0, 0.5, 0.0],
    )
    assert dIdV.tunnelingBroadenedStepFunction(0.0) == 0.5

    values = dIdV.tunnelingBroadenedStepFunction(np.linspace(-100.0, 100.0, 4001))
    assert np.all(np.isfinite(values))
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert np.all(np.diff(values) <= 1e-13)


def test_iets_spin_half_finite_temperature_reference():
    """Compare thermal IETS with independent populations and quadrature."""
    spin = SpinSys([0.5])
    spin.B = [0.0, 0.0, 5.0]
    spin.gfactorvector = [2.0]
    spin.G_st = 1e-9
    spin.U = np.array([0.0])
    spin.b0 = 0.0
    spin.TipPolarization = [0.0, 0.0, 0.0]
    spin.SamplePolarization = [0.0, 0.0, 0.0]
    hamil.calc_EigEnergies(spin, rebuildH=True)

    mu_B = const.physical_constants["Bohr magneton"][0]
    gap = 2 * mu_B * 5.0 / const.e * 1e3  # meV
    voltages = np.array([-2.0, -0.8, -gap, -0.2, 0.0, 0.2, gap, 0.8, 2.0])

    for temperature in (0.5, 1.0, 10.0, 300.0):
        spin.T = temperature
        thermal_energy = const.Boltzmann * temperature / const.e * 1e3
        p_excited = expit(-gap / thermal_energy)

        expected = []
        for voltage in voltages:
            excitation = _reference_broadened_step(
                (gap - voltage) / thermal_energy
            ) + _reference_broadened_step((gap + voltage) / thermal_energy)
            relaxation = _reference_broadened_step(
                (-gap - voltage) / thermal_energy
            ) + _reference_broadened_step((-gap + voltage) / thermal_energy)
            expected.append(
                spin.G_st
                * (1 + 2 * ((1 - p_excited) * excitation + p_excited * relaxation))
            )

        V, conductance = dIdV.calc_IETS_2nd(spin, V_array=voltages, norm=False)
        np.testing.assert_array_equal(V, voltages)
        np.testing.assert_allclose(
            conductance,
            expected,
            rtol=1e-9,
            atol=spin.G_st * 1e-11,
            err_msg=f"Incorrect thermal spectrum at T={temperature} K.",
        )


def test_iets_thermal_routes_agree_and_converge():
    """Verify agreement and grid convergence of the two thermal IETS routes."""
    spin = SpinSys([0.5])
    spin.B = [0.0, 0.0, 5.0]
    spin.T = 1.0
    spin.G_st = 1e-9
    spin.G_ss = [1e-8]
    spin.G_tt = 0
    spin.b0 = 0.0
    spin.SamplePolarization = [0.0, 0.0, 0.0]
    hamil.calc_EigEnergies(spin, rebuildH=True)

    for polarization, potential in ((0.0, 0.0), (0.6, 0.0), (-0.6, 0.25)):
        spin.TipPolarization = [0.0, 0.0, polarization]
        spin.U = np.array([potential])
        errors = []

        for points in (401, 801):
            voltages = np.linspace(-2.0, 2.0, points)
            _, thermal = dIdV.calc_IETS_2nd(spin, V_array=voltages, norm=False)
            V, rate_based, populations = dIdV.calc_IETS_2ndR(
                spin,
                V_array=voltages,
                AllowPumping=False,
                norm=False,
            )

            np.testing.assert_array_equal(V, voltages)
            assert np.all(np.isfinite(rate_based))
            np.testing.assert_allclose(populations.sum(axis=0), 1.0, rtol=0, atol=1e-12)
            errors.append(np.max(np.abs(rate_based[1:-1] - thermal[1:-1])) / spin.G_st)

        # Central differences have a second-order discretization error.
        assert errors[1] < 0.35 * errors[0], errors
        assert errors[1] < 2e-4, errors


def test_iets_spin_half_pumped_population_reference():
    """Verify analytical zero-temperature populations with substrate relaxation.

    Notes
    -----
    For unpolarized reservoirs, U=b0=0 and disabled tip-tip scattering,
    p_exc = (abs(eV)-gap)/(2*abs(eV)+(Gss/Gst)*gap) above threshold.
    Below threshold the excited-state population vanishes.
    """
    spin = SpinSys([0.5])
    spin.B = [0.0, 0.0, 5.0]
    spin.gfactorvector = [2.0]
    spin.T = 0.0
    spin.G_st = 1e-9
    spin.G_tt = 0
    spin.U = np.array([0.0])
    spin.b0 = 0.0
    spin.TipPolarization = [0.0, 0.0, 0.0]
    spin.SamplePolarization = [0.0, 0.0, 0.0]
    hamil.calc_EigEnergies(spin, rebuildH=True)

    mu_B = const.physical_constants["Bohr magneton"][0]
    gap = 2 * mu_B * 5.0 / const.e * 1e3  # meV
    voltages = np.linspace(-2.0, 2.0, 401)
    available_energy = np.abs(voltages)  # meV, for voltages expressed in mV

    for ratio in (0.01, 1.0, 100.0):
        spin.G_ss = [ratio * spin.G_st]
        expected_excited = np.where(
            available_energy <= gap,
            0.0,
            (available_energy - gap) / (2 * available_energy + ratio * gap),
        )
        expected = np.vstack((1 - expected_excited, expected_excited))

        V, conductance, populations = dIdV.calc_IETS_2ndR(
            spin,
            V_array=voltages,
            AllowPumping=True,
            norm=False,
        )

        np.testing.assert_array_equal(V, voltages)
        assert np.all(np.isfinite(conductance))
        assert np.all(np.isfinite(populations))
        assert np.all(populations >= -1e-12)
        np.testing.assert_allclose(populations.sum(axis=0), 1.0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            populations,
            expected,
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"Incorrect pumped populations for Gss/Gst={ratio}.",
        )

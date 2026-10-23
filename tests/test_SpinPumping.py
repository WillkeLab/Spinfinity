"""Scientific checks for stationary electrical spin pumping."""

import numpy as np
import scipy.constants as const

from spinfinity import SpinPumping as sp
from spinfinity import SpinSys, bath, hamil


def test_spin_pumping_high_bias_limit():
    """
    Verify the polarization limit for a single spin and a coupled pair.

    Notes
    -----
    Magnetizations are normalized as 2*<Sz>. Fields are collinear, bias
    is in mV, and all conductances are in S. Weak substrate coupling
    and disabled tip-tip scattering isolate the tunneling-dominated limit.
    The absolute tolerance includes the finite-bias correction at 100 mV.
    """
    for spins in ([0.5], [0.5, 0.5]):
        spin = SpinSys(spins)
        spin.B = [0.0, 0.0, 0.5]
        spin.BTip = [0.0, 0.0, 0.05]
        spin.gfactorvector = [2.0] * spin.NSpins
        spin.ReadoutSpin = 0
        spin.tip = "f"
        spin.T = 0.2
        spin.SamplePolarization = [0.0, 0.0, 0.0]
        spin.G_st = 0.5e-9
        spin.G_ss = [0.01 * spin.G_st] * spin.NSpins
        spin.G_tt = 0
        spin.U[:] = 0.0
        spin.b0 = 0.0
        spin.DipoleVector[:] = 0.0

        if spin.NSpins == 2:
            exchange = np.array([0.4, 0.4, 0.2]) * spin.GHztomeVConversion
            hamil.assign_ExchangeInteraction(spin, [(0, 1, exchange)])

        hamil.calc_EigEnergies(spin, rebuildH=True)

        for eta in (-0.4, 0.0, 0.4):
            spin.TipPolarization = [0.0, 0.0, eta]

            for voltage in (-100.0, 100.0):
                spin.V_DC = voltage
                bath.calc_Rates(spin)
                bath.calc_Populations(spin, AllowPumping=True)
                measured = 2.0 * bath.calc_Magnetization(
                    spin, type="z", AllowPumping=True
                )

                np.testing.assert_allclose(
                    measured,
                    np.sign(voltage) * eta,
                    rtol=0.0,
                    atol=1e-3,
                    err_msg=(
                        f"Incorrect high-bias limit: N={spin.NSpins}, "
                        f"eta={eta}, V={voltage} mV."
                    ),
                )


def test_spin_pumping_zero_temperature_analytical_approximation():
    """
    Compare remote magnetization with the paper's T=0 approximation.

    Notes
    -----
    Uses the voltage-based analytical approximation of Greule et al.,
    Electrical Spin Pumping in Exchange-coupled Molecules,
    https://arxiv.org/abs/2605.06450.

    Independent SI constants define the reference. Voltage is converted
    from mV to V and reference energies are in J. No parameters are fitted.
    T=0 does not remove the weak-mixing and dominant-pumping approximations.
    The tolerance is 0.01 in normalized magnetization, not machine precision.
    """
    spin = SpinSys([0.5, 0.5])
    spin.B = [0.0, 0.0, 0.5]
    spin.BTip = [0.0, 0.0, 0.05]
    spin.gfactorvector = [2.0, 2.0]
    spin.ReadoutSpin = 0
    spin.tip = "f"
    spin.T = 0.0
    spin.SamplePolarization = [0.0, 0.0, 0.0]
    spin.G_st = 0.5e-9
    spin.G_ss = [2.5e-9, 2.5e-9]
    spin.G_tt = 1
    spin.U[:] = 0.0
    spin.b0 = 0.0
    spin.DipoleVector[:] = 0.0

    J_perp_GHz = 0.4
    exchange = np.array([J_perp_GHz, J_perp_GHz, 0.2]) * spin.GHztomeVConversion
    hamil.assign_ExchangeInteraction(spin, [(0, 1, exchange)])
    hamil.calc_EigEnergies(spin, rebuildH=True)

    mu_B = const.physical_constants["Bohr magneton"][0]
    delta = spin.gfactorvector[0] * mu_B * spin.BTip[2]
    J_perp = const.h * J_perp_GHz * 1e9
    zeeman_energy = spin.gfactorvector[1] * mu_B * spin.B[2]

    mixing = (
        J_perp**2 / (4.0 * (delta**2 + J_perp**2))
        + 0.5
        - delta / (2.0 * np.sqrt(delta**2 + J_perp**2))
    )
    relaxation_current = spin.G_ss[1] * zeeman_energy / (2.0 * const.e)

    for eta in (-0.4, 0.4):
        spin.TipPolarization = [0.0, 0.0, eta]

        for voltage in (-100.0, -50.0, -20.0, -10.0, 10.0, 20.0, 50.0, 100.0):
            spin.V_DC = voltage
            bath.calc_Rates(spin)
            bath.calc_Populations(spin, AllowPumping=True)
            measured = (
                2.0 * bath.calc_Magnetization(spin, type="z", AllowPumping=True)[1]
            )

            # The paper's approximation uses G_st*V, not calc_TunnelCurrent.
            nominal_current = spin.G_st * voltage * 1e-3
            expected = (eta * nominal_current * mixing + relaxation_current) / (
                abs(nominal_current) * mixing + relaxation_current
            )

            np.testing.assert_allclose(
                measured,
                expected,
                rtol=0.0,
                atol=1e-2,
                err_msg=f"Incorrect approximation at eta={eta}, V={voltage} mV.",
            )


def test_uncoupled_remote_spin_remains_thermal():
    """
    Verify that an uncoupled remote spin stays in thermal equilibrium.

    Notes
    -----
    Both transverse and longitudinal exchange are zero. The remote spin
    couples only to the unpolarized substrate. Its normalized magnetization
    must equal tanh(g*mu_B*B/(2*k_B*T)), independently of bias and tip
    polarization. The reference uses SI constants and temperature in K.
    """
    spin = SpinSys([0.5, 0.5])
    spin.B = [0.0, 0.0, 0.5]
    spin.BTip = [0.0, 0.0, 0.05]
    spin.gfactorvector = [2.0, 2.0]
    spin.ReadoutSpin = 0
    spin.tip = "f"
    spin.T = 0.2
    spin.SamplePolarization = [0.0, 0.0, 0.0]
    spin.G_st = 0.5e-9
    spin.G_ss = [2.5e-9, 2.5e-9]
    spin.G_tt = 1
    spin.U[:] = 0.0
    spin.b0 = 0.0
    spin.Jvector[:] = 0.0
    spin.DipoleVector[:] = 0.0
    hamil.calc_EigEnergies(spin, rebuildH=True)

    mu_B = const.physical_constants["Bohr magneton"][0]
    expected = np.tanh(
        spin.gfactorvector[1] * mu_B * spin.B[2] / (2.0 * const.Boltzmann * spin.T)
    )

    for eta in (-0.4, 0.0, 0.4):
        spin.TipPolarization = [0.0, 0.0, eta]

        for voltage in (-100.0, 0.0, 100.0):
            spin.V_DC = voltage
            bath.calc_Rates(spin)
            bath.calc_Populations(spin, AllowPumping=True)
            measured = (
                2.0 * bath.calc_Magnetization(spin, type="z", AllowPumping=True)[1]
            )

            np.testing.assert_allclose(
                measured,
                expected,
                rtol=0.0,
                atol=1e-11,
                err_msg=f"Remote spin was pumped at eta={eta}, V={voltage} mV.",
            )


def test_split_populations_matches_partial_trace():
    """
    Compare local population projections with independent partial traces.

    Notes
    -----
    The two spins have unequal stationary magnetizations, so swapping
    subsystem A and B is detectable. A transverse tip field produces
    complex eigenvectors. Both laboratory-z and tilted subsystem bases
    are checked. split_Populations returns probabilities in the reference
    subsystem's eigenvector order, not necessarily laboratory-z order.
    """
    spin = SpinSys([0.5, 0.5])
    spin.B = [0.0, 0.0, 0.5]
    spin.BTip = [0.01, 0.015, 0.05]
    spin.gfactorvector = [2.0, 2.0]
    spin.ReadoutSpin = 0
    spin.tip = "f"
    spin.T = 0.2
    spin.TipPolarization = [0.0, 0.0, 0.4]
    spin.SamplePolarization = [0.0, 0.0, 0.0]
    spin.G_st = 0.5e-9
    spin.G_ss = [2.5e-9, 2.5e-9]
    spin.G_tt = 1
    spin.U[:] = 0.0
    spin.b0 = 0.0
    spin.DipoleVector[:] = 0.0

    exchange = np.array([0.4, 0.4, 0.2]) * spin.GHztomeVConversion
    hamil.assign_ExchangeInteraction(spin, [(0, 1, exchange)])
    hamil.calc_EigEnergies(spin, rebuildH=True)

    subspin = SpinSys([0.5])

    for reference_field in ([0.0, 0.0, 0.5], [0.3, 0.4, 0.5]):
        subspin.B = reference_field
        hamil.calc_EigEnergies(subspin, rebuildH=True)

        for voltage in (-20.0, 20.0):
            spin.V_DC = voltage
            populations_A, populations_B = sp.split_Populations(spin, subspin)

            # Reconstruct the density matrix in the product-spin basis.
            rho = spin.eigVectors @ np.diag(spin.Populations) @ spin.eigVectors.conj().T
            rho_tensor = rho.reshape(2, 2, 2, 2)  # A_row, B_row, A_col, B_col
            rho_A = np.trace(rho_tensor, axis1=1, axis2=3)
            rho_B = np.trace(rho_tensor, axis1=0, axis2=2)

            basis = subspin.eigVectors
            expected_A = np.diag(basis.conj().T @ rho_A @ basis).real
            expected_B = np.diag(basis.conj().T @ rho_B @ basis).real

            assert abs(expected_A[0] - expected_B[0]) > 1e-3, (
                "The reference case must distinguish subsystem A from B."
            )

            for measured, expected in (
                (populations_A, expected_A),
                (populations_B, expected_B),
            ):
                np.testing.assert_allclose(measured, expected, rtol=0.0, atol=1e-12)
                np.testing.assert_allclose(measured.sum(), 1.0, rtol=0.0, atol=1e-12)
                assert np.all(measured >= -1e-12), "Negative local probability."
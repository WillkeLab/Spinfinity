import numpy as np
import scipy.constants as const

from spinfinity import SpinSys, hamil


def test_spin_half_in_magnetic_field():
    """A spin-1/2 at 1 T should have the expected Zeeman splitting."""
    spinsys = SpinSys([0.5])
    spinsys.B = [0.0, 0.0, 1.0]  # T

    hamil.calc_EigEnergies(spinsys)

    # A Hamiltonian must be Hermitian.
    np.testing.assert_allclose(
        spinsys.H,
        spinsys.H.conj().T,
        atol=1e-12,
    )

    # Independent SI reference for g=2 and B=1 T.
    expected_gap = (
        2.0 * const.physical_constants["Bohr magneton"][0]
        / const.e * 1e3
    )

    np.testing.assert_allclose(
        spinsys.E_All,
        [0.0, expected_gap],
        rtol=1e-10,
        atol=1e-14,
    )

    # Eigenvectors must be orthonormal.
    identity = np.eye(2)
    np.testing.assert_allclose(
        spinsys.eigVectors.conj().T @ spinsys.eigVectors,
        identity,
        atol=1e-12,
    )


def test_dipolar_coupling_two_spins():
    """Compare the two-spin dipolar Hamiltonian with its analytical matrix."""
    spin = SpinSys([0.5, 0.5])
    spin.gfactorvector = [2.0, 2.0]
    spin.latticeLengthXY = np.array([0.25, 0.25, 0.25])  # nm
    spin.AtomPositions = np.array([
        [0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ])

    hamil.assign_DipolarCoupling(spin, [(0, 1, "calc")])
    hamil.build_Hamiltonian(spin)

    # Independent SI reference: g0 = g1 = 2, separation = 1 nm.
    mu_B = const.physical_constants["Bohr magneton"][0]
    D0 = const.mu_0 * 4 * mu_B**2 / (4 * np.pi * (1e-9)**3)
    D0 *= 1e3 / const.e  # J -> meV

    # Product basis: |up up>, |up down>, |down up>, |down down>.
    H_expected = D0 / 4 * np.array([
        [1.0, 0.0, 0.0, -3.0],
        [0.0, -1.0, -1.0, 0.0],
        [0.0, -1.0, -1.0, 0.0],
        [-3.0, 0.0, 0.0, 1.0],
    ])

    np.testing.assert_allclose(
        spin.H, H_expected, rtol=1e-8, atol=1e-14
    )


def test_ferrimagnet_zero_field_ground_doublet():
    """Verify the analytical zero-field ground doublet of a mixed-spin dimer.

    Notes
    -----
    Check zero anisotropy and finite axial anisotropy. Compare projectors
    because phases and basis choices within the degenerate doublet are arbitrary.
    The product basis orders the spin-1 site before the spin-1/2 site.
    """
    J_meV = 14.0

    for D_meV in (0.0, 1.9):
        spin = SpinSys([1.0, 0.5])
        spin.B = [0.0, 0.0, 0.0]
        spin.BTip = [0.0, 0.0, 0.0]
        spin.Dvector = np.array([D_meV, 0.0])

        hamil.assign_ExchangeInteraction(
            spin, [(0, 1, (J_meV, J_meV, J_meV))]
        )
        hamil.calc_EigEnergies(spin, rebuildH=True)

        # Exact zero-field coefficients for isotropic exchange.
        denominator = np.sqrt(
            (J_meV - 2 * D_meV) ** 2 + 8 * J_meV**2
        )
        weight = (1 + (J_meV - 2 * D_meV) / denominator) / 2
        a = np.sqrt(1 - weight)
        b = np.sqrt(weight)

        # Rows: |1,+1/2>, |1,-1/2>, |0,+1/2>,
        #       |0,-1/2>, |-1,+1/2>, |-1,-1/2>.
        expected_states = np.array([
            [0.0, 0.0],
            [-b, 0.0],
            [a, 0.0],
            [0.0, -a],
            [0.0, b],
            [0.0, 0.0],
        ])

        ground_states = spin.eigVectors[:, :2]
        expected_projector = expected_states @ expected_states.conj().T
        calculated_projector = ground_states @ ground_states.conj().T

        np.testing.assert_allclose(
            spin.E_All[:2], 0.0, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            calculated_projector,
            expected_projector,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Incorrect ground-doublet contributions at D={D_meV} meV.",
        )


def test_exchange_coupled_spin_half_eigenenergies_greule():
    """Compute the eigenenergies of two weakly exchange-coupled spin-1/2.

    Notes
    -----
    Use collinear external and tip fields and equal g-factors.
    Compare both absolute and ground-state-referenced energies in meV.
    """
    g = 2.0
    B = 0.5  # T

    # Independent SI constants and conversion to meV.
    mu_B = const.physical_constants["Bohr magneton"][0] / const.e * 1e3
    GHz_to_meV = const.h * 1e12 / const.e
    J_perp = 0.4 * GHz_to_meV
    J_parallel = 0.2 * GHz_to_meV

    for Btip_z in (-30e-3, 0.0, 0.1e-3, 30e-3):
        spin = SpinSys([0.5, 0.5])
        spin.gfactorvector = [g, g]
        spin.B = [0.0, 0.0, B]
        spin.BTip = [0.0, 0.0, Btip_z]
        spin.ReadoutSpin = 0
        spin.tip = "f"

        hamil.assign_ExchangeInteraction(
            spin, [(0, 1, (J_perp, J_perp, J_parallel))]
        )
        hamil.calc_EigEnergies(spin, rebuildH=True)

        # Analytical eigenenergies, all expressed in meV.
        delta = g * mu_B * Btip_z
        splitting = np.sqrt(delta**2 + J_perp**2)
        expected = np.sort(
            np.array([
                J_parallel / 4 - g * mu_B * B - delta / 2,
                -J_parallel / 4 - splitting / 2,
                -J_parallel / 4 + splitting / 2,
                J_parallel / 4 + g * mu_B * B + delta / 2,
            ])
        )

        np.testing.assert_allclose(
            spin.E_sort,
            expected,
            rtol=1e-10,
            atol=1e-14,
            err_msg=f"Incorrect absolute energies at Btip_z={Btip_z} T.",
        )
        np.testing.assert_allclose(
            spin.E_All,
            expected - expected[0],
            rtol=1e-10,
            atol=1e-14,
            err_msg=f"Incorrect excitation energies at Btip_z={Btip_z} T.",
        )
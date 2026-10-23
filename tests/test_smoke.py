"""Smoke tests for small public Spinfinity workflows.

These tests check execution, output shapes and basic invariants, not detailed
agreement with analytical models. Fields are in T, voltages in mV, frequencies
in GHz, evolution times in ns, currents in A and conductances in S.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
from qutip import Qobj

from spinfinity import CC, ESR, SpinSys, bath, dIdV, hamil, lind


@pytest.fixture
def spin():
    """Create a fresh spin-half tunnel-junction setup.

    Returns
    -------
    SpinSys
        Configured system without cached Hamiltonian, rates or populations.
    """
    spin = SpinSys([0.5])
    spin.B = [0.0, 0.0, 0.5]
    spin.BTip = [1e-9, 0.0, 0.01]
    spin.T = 1.0
    spin.V_DC = 1.0
    spin.TipPolarization = [0.0, 0.0, 0.4]
    spin.G_st = 0.1e-9
    spin.G_ss = [1e-9]
    spin.G_tt = 1
    spin.U = np.array([0.1])
    spin.b0 = 0.1
    return spin


@pytest.fixture(autouse=True)
def hide_plots(monkeypatch):
    """Disable figure display without mocking any calculations.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture that restores the original display method after each test.
    """
    monkeypatch.setattr(go.Figure, "show", lambda *args, **kwargs: None)


def test_smoke_hamiltonian():
    """Build, diagonalize and visualize a coupled electronic spin system."""
    spin = SpinSys([1.0, 0.5])
    spin.B = [0.0, 0.0, 0.5]
    spin.Dvector[0] = 0.02
    hamil.assign_ExchangeInteraction(spin, [(0, 1, [0.01, 0.01, 0.02])])
    hamil.calc_EigStates(spin, rebuildH=True)

    assert spin.H.shape == (6, 6)
    assert np.all(np.isfinite(spin.E_All))
    np.testing.assert_allclose(spin.H, spin.H.conj().T, atol=1e-14)

    output = hamil.plot_EigenMatrix(spin)
    assert isinstance(output["fig"], go.Figure)
    assert len(output["fig"].data) == 1
    assert output["m"].shape == (6, 6)
    np.testing.assert_allclose(output["m"].sum(axis=1), 1.0, atol=1e-12)


def test_smoke_bath(spin):
    """Run rates, stationary populations, current and state lifetimes.

    Parameters
    ----------
    spin : SpinSys
        Fresh spin-half setup supplied by the fixture.
    """
    hamil.calc_EigEnergies(spin, rebuildH=True)
    bath.calc_Rates(spin)
    bath.calc_Populations(spin, AllowPumping=True)
    bath.calc_Lifetimes(spin)

    assert spin.Rates.shape == (4, 2, 2)
    assert np.all(np.isfinite(spin.Rates))
    assert np.all(spin.Rates >= 0.0)
    assert spin.Populations.shape == (2,)
    assert np.all(spin.Populations >= -1e-12)
    np.testing.assert_allclose(spin.Populations.sum(), 1.0, atol=1e-12)
    assert np.all(np.isfinite(spin.T_state_tot))
    assert np.all(spin.T_state_tot > 0.0)

    for pumping in (False, True):
        current = bath.calc_TunnelCurrent(spin, AllowPumping=pumping)
        assert np.ndim(current) == 0
        assert np.isfinite(current)


@pytest.mark.parametrize("order", ["2nd", "2ndR", "3rd", "3rdR"])
def test_smoke_iets(spin, order):
    """Calculate and plot each implemented IETS mode on a short voltage grid.

    Parameters
    ----------
    spin : SpinSys
        Fresh spin-half setup supplied by the fixture.
    order : str
        IETS mode selected by pytest.
    """
    spin.J = np.array([-0.005])
    spin.J_tip = -0.002
    voltage = np.linspace(-0.2, 0.2, 11)
    output = dIdV.plot_IETS(spin, V_array=voltage, order=order, norm=False)

    np.testing.assert_array_equal(output["V_array"], voltage)
    assert output["dIdVsignal"].shape == voltage.shape
    assert np.all(np.isfinite(output["dIdVsignal"]))
    assert isinstance(output["fig"], go.Figure)
    assert len(output["fig"].data) == 1
    np.testing.assert_allclose(output["fig"].data[0].y, output["dIdVsignal"])


@pytest.mark.parametrize("mode", ["phenomenological", "rates"])
def test_smoke_esr(spin, mode):
    """Calculate and plot both ESR sweep implementations.

    Parameters
    ----------
    spin : SpinSys
        Fresh spin-half setup supplied by the fixture.
    mode : str
        ESR implementation selected by pytest.
    """
    frequency = np.linspace(13.0, 15.0, 31)
    if mode == "rates":
        output = ESR.plot_FreqSweep_ESRrate(
            spin, freq_array=frequency, B1=1e-3, T2=20.0, norm=False
        )
    else:
        output = ESR.plot_FreqSweep(
            spin, freq_array=frequency, AllowPumping=False, norm=False
        )

    np.testing.assert_array_equal(output["Freq_array"], frequency)
    assert output["ESRsignal"].shape == frequency.shape
    assert np.all(np.isfinite(output["ESRsignal"]))
    assert np.max(np.abs(output["ESRsignal"])) > 0.0
    assert isinstance(output["fig"], go.Figure)
    assert len(output["fig"].data) >= 1
    np.testing.assert_allclose(output["fig"].data[0].y, output["ESRsignal"])


def test_smoke_lindblad(spin):
    """Combine collapse operators, evolve a density matrix and plot the result.

    Parameters
    ----------
    spin : SpinSys
        Fresh spin-half setup supplied by the fixture.
    """
    hamil.calc_EigEnergies(spin, rebuildH=True)
    spin.collapse_ops = lind.calc_collapseOperators(spin)
    spin.collapse_ops += lind.calc_elasticCollapseOperators(spin)
    spin.collapse_ops.append(lind.calc_dephasingOperator(spin, 0, T_phi=100.0))
    rho = lind.initialize_thermal(spin)
    times = np.linspace(0.0, 10.0, 11)

    assert spin.collapse_ops
    assert all(op.shape == (2, 2) for op in spin.collapse_ops)
    assert all(np.all(np.isfinite(op.full())) for op in spin.collapse_ops)

    result = lind.calc_TimeEvolution(spin, psi_0=rho, time_array=times)
    np.testing.assert_array_equal(result.times, times)
    assert np.asarray(result.expect).shape == (1, len(times))
    assert np.all(np.isfinite(result.expect))
    assert len(result.states) == len(times)
    for state in result.states:
        np.testing.assert_allclose(state.tr(), 1.0, atol=1e-8)
        assert state.isherm

    output = lind.plot_TimeEvolution(spin)
    assert isinstance(output["fig"], go.Figure)
    np.testing.assert_allclose(output["fig"].data[0].y, result.expect[0])


def test_smoke_coherent_control(spin):
    """Run a short Rabi sequence with explicitly assigned dissipation.

    Parameters
    ----------
    spin : SpinSys
        Fresh spin-half setup supplied by the fixture.
    """
    hamil.calc_EigEnergies(spin, rebuildH=True)
    spin.collapse_ops = lind.calc_collapseOperators(spin)
    spin.collapse_ops.append(lind.calc_dephasingOperator(spin, 0, T_phi=100.0))
    lind.initialize_thermal(spin)
    spin.f_RF = spin.E_All_inGHz[1] - spin.E_All_inGHz[0]
    spin.RabiRate = 0.02
    times = np.linspace(0.0, 20.0, 11)

    returned_times, result = CC.calc_RabiOscillation(
        spin, times, e_ops=[Qobj(spin.Sz[0])]
    )

    np.testing.assert_array_equal(returned_times, times)
    assert np.asarray(result.expect).shape == (1, len(times))
    assert np.all(np.isfinite(result.expect))


def test_smoke_save_load(spin, tmp_path):
    """Save and reload a system using only a temporary test directory.

    Parameters
    ----------
    spin : SpinSys
        Fresh spin-half setup supplied by the fixture.
    tmp_path : pathlib.Path
        Temporary directory managed by pytest.
    """
    hamil.calc_EigEnergies(spin, rebuildH=True)
    saved_path = spin.save(filename="spin.pkl", filepath=str(tmp_path))
    assert Path(saved_path).is_file()
    loaded = SpinSys.load(filename="spin.pkl", filepath=str(tmp_path))

    assert loaded.Spins == spin.Spins
    np.testing.assert_array_equal(loaded.B, spin.B)
    np.testing.assert_array_equal(loaded.BTip, spin.BTip)
    np.testing.assert_array_equal(loaded.H, spin.H)
    hamil.calc_EigEnergies(loaded, rebuildH=True)
    np.testing.assert_allclose(loaded.E_All, spin.E_All, atol=1e-14)
    assert loaded.get_parameters_dict()["System"]["NSpins"] == 1
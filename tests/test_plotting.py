"""Check display controls without opening figures or changing the physics."""

from unittest.mock import patch

import numpy as np
import plotly.graph_objects as go
import pytest
from qutip import Bloch, Qobj

from spinfinity import CC, ESR, SpinSys, bath, dIdV, hamil, lind, plotting
from spinfinity import SpinPumping as pumping


@pytest.mark.parametrize(
    ("plot_name", "kwargs", "preferred_width"),
    [
        ("plot_EigenMatrix", {}, 700),
        ("plot_EnergyVsS", {}, 700),
        ("plot_ZeemanDiagramm", {"B_array": np.linspace(0.1, 0.6, 3)}, 600),
        (
            "plot_TipZeemanDiagramm",
            {"BTip_array": np.linspace(0.0, 0.1, 3)},
            600,
        ),
    ],
)
@pytest.mark.parametrize("show", [None, "fixed", "responsive", "no"])
def test_hamiltonian_plot_display_modes(
    monkeypatch, plot_name, kwargs, preferred_width, show
):
    """
    Check each plot's display mode and unchanged numerical results.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture restoring the global default after the test.
    plot_name : str
        Spin Hamiltonian plotting function to exercise.
    kwargs : dict
        Plot-specific arguments.
    preferred_width : int
        Expected fixed width in pixels.
    show : str or None
        Per-call display mode.
    """
    monkeypatch.setattr(plotting, "_DISPLAY_MODE", "fixed")
    spin = SpinSys([0.5])
    spin.B = [0.0, 0.0, 0.5]
    spin.BTip = [0.0, 0.0, 0.01]
    hamil.calc_EigStates(spin, rebuildH=True)
    field_initial = np.array(spin.B)
    tip_initial = np.array(spin.BTip)
    energies_initial = spin.E_All.copy()
    plot_function = getattr(hamil, plot_name)

    with patch.object(go.Figure, "show") as display:
        reference = plot_function(spin, **kwargs, show="no")
        display.assert_not_called()
        output = plot_function(spin, **kwargs, show=show)
        responsive = show == "responsive"
        if show == "no":
            display.assert_not_called()
        else:
            display.assert_called_once_with(config={"responsive": responsive})

    fig = output["fig"]
    assert isinstance(fig, go.Figure)
    assert fig.layout.height == 450
    assert fig.layout.autosize is responsive
    assert fig.layout.width == (None if responsive else preferred_width)
    assert output.keys() == reference.keys()
    for key in output:
        if key != "fig":
            np.testing.assert_allclose(output[key], reference[key])
    np.testing.assert_allclose(spin.B, field_initial)
    np.testing.assert_allclose(spin.BTip, tip_initial)
    np.testing.assert_allclose(spin.E_All, energies_initial)


def test_global_display_mode_and_overrides(monkeypatch):
    """
    Check that subsequent calls inherit the setting and can override it.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture restoring the global default after the test.
    """
    monkeypatch.setattr(plotting, "_DISPLAY_MODE", "fixed")
    spin = SpinSys([0.5])
    spin.B = [0.0, 0.0, 0.5]
    hamil.calc_EigStates(spin, rebuildH=True)

    with patch.object(go.Figure, "show") as display:
        for mode in ("responsive", "no", "fixed"):
            plotting.set_display_mode(mode)
            display.reset_mock()
            output = hamil.plot_EigenMatrix(spin)
            assert output["fig"].layout.autosize is (mode == "responsive")
            if mode == "no":
                display.assert_not_called()
            else:
                display.assert_called_once_with(
                    config={"responsive": mode == "responsive"}
                )

            for override in ("fixed", "responsive", "no"):
                display.reset_mock()
                output = hamil.plot_EigenMatrix(spin, show=override)
                assert output["fig"].layout.autosize is (override == "responsive")
                if override == "no":
                    display.assert_not_called()
                else:
                    display.assert_called_once_with(
                        config={"responsive": override == "responsive"}
                    )
                assert mode == plotting._DISPLAY_MODE


@pytest.mark.parametrize("show", [None, "fixed", "responsive", "no"])
def test_plot_all_forwards_display_mode(monkeypatch, show):
    """
    Check the combined plot forwards the mode and retains its return value.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture restoring the global default after the test.
    show : str or None
        Display mode forwarded to the individual plotting functions.
    """
    monkeypatch.setattr(plotting, "_DISPLAY_MODE", "no")
    spin = SpinSys([0.5])
    spin.B = [0.0, 0.0, 0.5]
    hamil.calc_EigStates(spin, rebuildH=True)

    with patch.object(go.Figure, "show", autospec=True) as display:
        assert hamil.plot_All(spin, show=show) is None
        if show in (None, "no"):
            display.assert_not_called()
        else:
            assert display.call_count == 3
            for call, width in zip(
                display.call_args_list, (700, 700, 600), strict=True
            ):
                fig = call.args[0]
                assert call.kwargs == {"config": {"responsive": show == "responsive"}}
                assert fig.layout.width == (None if show == "responsive" else width)


def test_invalid_display_modes(monkeypatch):
    """
    Reject unsupported modes without displaying or changing the default.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture restoring the global default after the test.
    """
    monkeypatch.setattr(plotting, "_DISPLAY_MODE", "fixed")
    spin = SpinSys([0.5])
    spin.B = [0.0, 0.0, 0.5]
    hamil.calc_EigStates(spin, rebuildH=True)

    with patch.object(go.Figure, "show") as display:
        for invalid in ("capped", "invalid", None):
            with pytest.raises(ValueError, match="mode must"):
                plotting.set_display_mode(invalid)
        with pytest.raises(ValueError, match="show must"):
            hamil.plot_EigenMatrix(spin, show="capped")
        display.assert_not_called()
    assert plotting._DISPLAY_MODE == "fixed"


def _plotting_spin(spins):
    """
    Build a small finite-rate system for plotting workflows.

    Parameters
    ----------
    spins : list of float
        Local spin quantum numbers.

    Returns
    -------
    spin : SpinSys
        Spin system with computed eigenstates, rates, and populations.
    """
    spin = SpinSys(spins)
    spin.B = [0.0, 0.0, 0.5]
    spin.BTip = [0.01, 0.0, 0.01]
    spin.T = 0.5
    spin.V_DC = 1.0
    spin.G_st = 1e-10
    spin.G_ss = [1e-9] * len(spins)
    spin.G_tt = 1
    spin.TipPolarization = [0.0, 0.0, 0.4]
    spin.U = np.full(len(spins), 0.1)
    hamil.calc_EigStates(spin, rebuildH=True)
    bath.calc_Rates(spin)
    bath.calc_Populations(spin)
    return spin


@pytest.mark.parametrize("mode", ["fixed", "responsive", "no"])
def test_package_plot_workflows_display_modes(monkeypatch, mode):
    """
    Exercise all plotting modules and their display branches without a GUI.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture restoring the global default after the test.
    mode : str
        Global display mode used by the plotting workflows.
    """
    monkeypatch.setattr(plotting, "_DISPLAY_MODE", mode)
    responsive = mode == "responsive"

    def check(function, *args, width=800, **kwargs):
        with (
            patch.object(go.Figure, "show", autospec=True) as display,
            patch.object(Bloch, "show", autospec=True) as bloch_display,
        ):
            output = function(*args, **kwargs)
        if mode == "no":
            display.assert_not_called()
            bloch_display.assert_not_called()
        else:
            assert display.call_count >= 1
            for call in display.call_args_list:
                fig = call.args[0]
                assert fig.layout.autosize is responsive
                assert fig.layout.width == (None if responsive else width)
                assert fig.layout.height == 450
                assert call.kwargs == {"config": {"responsive": responsive}}
            if function.__module__ == CC.__name__:
                bloch_display.assert_called_once()
        if output is not None and "fig" in output:
            assert output["fig"].layout.autosize is responsive
            assert output["fig"].layout.width == (None if responsive else width)
            assert output["fig"].layout.height == 450

    spin = _plotting_spin([0.5])
    check(bath.plot_Populations, spin, width=700)
    for count in (None, 1, 0):
        check(bath.plot_RateContributions, spin, N=count, width=1200)
        check(bath.plot_RatesTo, spin, state=0, N=count, width=1000)
        check(bath.plot_RatesFrom, spin, state=0, N=count, width=1000)

    frequencies = np.linspace(13.0, 15.0, 11)
    check(ESR.plot_FreqSweep, spin, freq_array=frequencies, norm=False)
    check(ESR.plot_FreqSweep_ESRrate, spin, freq_array=frequencies, T2=20, norm=False)
    check(ESR.plot_ESRPopulations, spin, freq=14, T2=20)
    check(ESR.plot_ESRPopulations, spin, freq=14, T2=20, filter=0.001)
    check(
        ESR.plot_Tipfieldmap,
        spin,
        freq_array=frequencies,
        BTip_array=np.array([0.01, 0.02]),
        Angle=45,
        norm=False,
    )
    for all_spins in (False, True):
        check(
            dIdV.plot_IETS,
            spin,
            V_array=np.linspace(-0.2, 0.2, 11),
            plotAllSpins=all_spins,
            norm=False,
        )

    spin = _plotting_spin([0.5])
    spin.collapse_ops = lind.calc_collapseOperators(spin)
    initial_state = (Qobj(spin.eigVectors[:, 0]) + Qobj(spin.eigVectors[:, 1])).unit()
    lind.calc_TimeEvolution(
        spin, psi_0=initial_state, time_array=np.linspace(0, 1, 101), Saxis="x"
    )
    check(lind.plot_TimeEvolution, spin)
    check(
        lind.plot_DensityMatrix,
        spin,
        time_type="index",
        time_index=[0, 1, 2, 3],
        width=1000,
    )
    check(lind.plot_TunnelCurrent_time, spin)
    check(lind.plot_FourierFreqs, spin)
    spin.timeEvolution_Tipfield_result = np.vstack(
        [spin.timeEvolution_result.expect[0]] * 2
    )
    spin.timeEvolution_Tipfield_time_array = np.asarray(spin.timeEvolution_result.times)
    spin.timeEvolution_Tipfield_BTip_array = np.array([0.01, 0.02])
    spin.timeEvolution_Tipfield_Saxis = "x"
    spin.timeEvolution_Tipfield_Baxis = "z"
    check(lind.plot_TimeEvolution_Tipfield, spin)

    spin = _plotting_spin([0.5, 0.5])
    hamil.assign_ExchangeInteraction(spin, [(0, 1, [0.001, 0.001, 0.002])])
    hamil.calc_EigStates(spin, rebuildH=True)
    bath.calc_Rates(spin)
    subspin = _plotting_spin([0.5])
    voltages = np.array([-0.2, 0.0, 0.2])
    check(pumping.plot_Magnetization_Voltage, spin, V_array=voltages, spins=[0, 1])
    check(pumping.plot_Populations_Voltage, spin, V_array=voltages)
    check(pumping.plot_Rates_Voltage, spin, V_array=voltages, width=900)
    check(pumping.plot_SplitMagnetization_Voltage, spin, subspin, V_array=voltages)
    check(
        pumping.plot_SplitPopulations_Voltage,
        spin,
        subspin,
        V_array=voltages,
        width=1000,
    )
    check(pumping.plot_SplitPopulations, spin, subspin, width=900)

    spin = _plotting_spin([0.5])
    spin.collapse_ops = lind.calc_collapseOperators(spin)
    spin.psi0 = Qobj(spin.eigVectors[:, 0])
    spin.f_RF = float(spin.E_All_inGHz[1])
    spin.RabiRate = 0.02
    check(CC.plot_RamseyShot, spin, tau=3, tsteps=101)
    check(CC.plot_HahnEchoShot, spin, tau=3, tsteps=101)
    check(CC.plot_HahnEchoShot_pluspihalf, spin, tau=3, tsteps=101)
    check(CC.plot_3pulseESEEMShot_pluspihalf, spin, T=2, tau=3, tsteps=101)


@pytest.mark.parametrize("mode", ["fixed", "responsive", "no"])
def test_custom_figure_display_modes(monkeypatch, mode):
    """
    Check the public display helper used by custom tutorial figures.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture restoring the global default after the test.
    mode : str
        Global display mode.
    """
    monkeypatch.setattr(plotting, "_DISPLAY_MODE", mode)
    fig = go.Figure()
    with patch.object(go.Figure, "show") as display:
        assert plotting.show_figure(fig, preferred_width=900) is None
        if mode == "no":
            display.assert_not_called()
        else:
            display.assert_called_once_with(config={"responsive": mode == "responsive"})
    assert fig.layout.autosize is (mode == "responsive")
    assert fig.layout.width == (None if mode == "responsive" else 900)

    with patch.object(go.Figure, "show") as display:
        plotting.show_figure(fig, show="no")
        display.assert_not_called()
    assert mode == plotting._DISPLAY_MODE

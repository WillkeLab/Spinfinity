"""Check ESR transition reporting independently of spectra and figure display."""

from unittest.mock import patch

import numpy as np
import plotly.graph_objects as go
import pytest

from spinfinity import ESR, SpinSys


@pytest.fixture
def spin():
    """
    Configure a spin-half system with a finite phenomenological ESR signal.

    Returns
    -------
    SpinSys
        Spin system with fields in T and temperature in K.
    """
    spin = SpinSys([0.5])
    spin.B = [0.0, 0.0, 0.5]
    spin.BTip = [0.01, 0.0, 0.01]
    spin.T = 0.5
    return spin


@pytest.mark.parametrize("output_flag", [None, False, True])
def test_calc_freqsweep_output(spin, capsys, output_flag):
    """
    Check transition reporting and an unchanged returned spectrum.

    Parameters
    ----------
    spin : SpinSys
        Spin-half setup supplied by the fixture.
    capsys : pytest.CaptureFixture
        Fixture capturing console output.
    output_flag : bool or None
        Explicit flag, or None to exercise the default.
    """
    frequencies = np.linspace(13.0, 15.0, 101)
    reference = ESR.calc_FreqSweep(
        spin, freq_array=frequencies, norm=False, output=False
    )
    assert capsys.readouterr().out == ""

    kwargs = {} if output_flag is None else {"output": output_flag}
    result = ESR.calc_FreqSweep(spin, freq_array=frequencies, norm=False, **kwargs)
    report = capsys.readouterr().out
    if output_flag is False:
        assert report == ""
    else:
        assert "Found 1 potential transitions:" in report
        assert "|0> -> |1>; f = " in report
        assert "GHz" in report

    np.testing.assert_array_equal(result[0], reference[0])
    np.testing.assert_array_equal(result[1], reference[1])
    assert np.all(np.isfinite(result[1]))
    assert np.max(result[1]) > 0.0


@pytest.mark.parametrize("output_flag", [None, False, True])
def test_plot_freqsweep_output(spin, capsys, output_flag):
    """
    Check flag forwarding without displaying the figure or changing results.

    Parameters
    ----------
    spin : SpinSys
        Spin-half setup supplied by the fixture.
    capsys : pytest.CaptureFixture
        Fixture capturing console output.
    output_flag : bool or None
        Explicit flag, or None to exercise the quiet wrapper default.
    """
    frequencies = np.linspace(13.0, 15.0, 101)
    reference = ESR.calc_FreqSweep(
        spin, freq_array=frequencies, norm=False, output=False
    )
    assert capsys.readouterr().out == ""

    kwargs = {} if output_flag is None else {"output": output_flag}
    with patch.object(go.Figure, "show") as display:
        result = ESR.plot_FreqSweep(
            spin, freq_array=frequencies, norm=False, show="no", **kwargs
        )
        display.assert_not_called()
    report = capsys.readouterr().out
    if output_flag is True:
        assert "Found 1 potential transitions:" in report
    else:
        assert report == ""

    np.testing.assert_array_equal(result["Freq_array"], reference[0])
    np.testing.assert_array_equal(result["ESRsignal"], reference[1])


def test_tipfieldmap_suppresses_transition_reports(spin, capsys):
    """
    Check that a tip-field map does not print a transition list for each row.

    Parameters
    ----------
    spin : SpinSys
        Spin-half setup supplied by the fixture.
    capsys : pytest.CaptureFixture
        Fixture capturing console output.
    """
    frequencies = np.linspace(13.0, 15.0, 101)
    tip_fields = np.array([0.01, 0.02])
    initial_tip = np.array(spin.BTip)
    with patch.object(go.Figure, "show") as display:
        result = ESR.plot_Tipfieldmap(
            spin,
            freq_array=frequencies,
            BTip_array=tip_fields,
            norm=False,
            show="no",
        )
        display.assert_not_called()

    assert capsys.readouterr().out == ""
    assert result["ESRsignal"].shape == (len(tip_fields), len(frequencies))
    assert np.all(np.isfinite(result["ESRsignal"]))
    np.testing.assert_array_equal(spin.BTip, initial_tip)

"""
Minimal, extensible GUI for configuring `SpinSys` objects and rendering
visualizations.

The layout is intentionally simple:
    - A parameter form to configure the spin system.
    - A selector for available visualization routines.
    - Dynamic controls for visualization-specific arguments.
    - A matplotlib canvas that embeds the figure produced by the selected routine.

This module is designed to be a starting point that can be extended with
additional parameters or visualizations by editing the `BASE_PARAMETER_SPECS`
or `VISUALIZATION_OPTIONS` lists below.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from spinfinity.SpinSys import SpinSys
import spinfinity.SpinVisualization as vis


# --------------------------------------------------------------------------- #
# Helper data structures
# --------------------------------------------------------------------------- #


@dataclass
class ParameterSpec:
    """Declarative description of a single form field."""

    name: str
    label: str
    field_type: str = "text"  # text, float, int, choice, bool
    default: Any = ""
    tooltip: Optional[str] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    decimals: int = 4
    choices: Optional[List[str]] = None


@dataclass
class VisualizationOption:
    """Metadata for a visualization entry."""

    key: str
    label: str
    func: Callable[..., Any]
    parameters: List[ParameterSpec] = field(default_factory=list)
    fixed_kwargs: Dict[str, Any] = field(default_factory=dict)
    arg_builder: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    description: Optional[str] = None


# --------------------------------------------------------------------------- #
# Parsing utilities
# --------------------------------------------------------------------------- #


def _parse_numeric_sequence(
    value: Any,
    *,
    expected_len: Optional[int] = None,
    default: Optional[List[float]] = None,
    allow_empty: bool = False,
) -> List[float]:
    """
    Convert user input into a list of floats.

    Examples of accepted inputs:
        "1, 1.5, 2"
        "0.5"
        [1, 2, 3]
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        sequence = [float(v) for v in value]
    else:
        text = str(value).strip()
        if not text:
            if default is not None:
                sequence = [float(v) for v in default]
            elif allow_empty:
                sequence = []
            else:
                raise ValueError("Expected at least one numeric value.")
        else:
            tokens = [token.strip() for token in text.replace(";", ",").split(",") if token.strip()]
            if not tokens:
                if default is not None:
                    sequence = [float(v) for v in default]
                elif allow_empty:
                    sequence = []
                else:
                    raise ValueError("Expected at least one numeric value.")
            else:
                try:
                    sequence = [float(token) for token in tokens]
                except ValueError as exc:
                    raise ValueError(f"Could not parse numeric sequence from '{text}'.") from exc

    if expected_len is not None:
        if len(sequence) == expected_len:
            return sequence
        if len(sequence) == 1:
            sequence = sequence * expected_len
        else:
            raise ValueError(f"Expected {expected_len} values, received {len(sequence)}.")

    return sequence


def _parse_integer(value: Any) -> int:
    """Parse an integer value and raise a friendly error if it fails."""
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not parse integer value from '{value}'.") from exc


# --------------------------------------------------------------------------- #
# Spin system builder
# --------------------------------------------------------------------------- #


def build_spin_system(params: Dict[str, Any]) -> SpinSys:
    """
    Instantiate and populate a `SpinSys` object from form inputs.
    """
    spins = _parse_numeric_sequence(params["spins"], allow_empty=False)
    if not spins:
        raise ValueError("At least one spin value is required.")

    spinsys = SpinSys(spins)

    # Core parameters
    spinsys.T = float(params["temperature"])
    spinsys.ReadoutSpin = _parse_integer(params["readout_spin"])
    if not 0 <= spinsys.ReadoutSpin < spinsys.NSpins:
        raise ValueError(f"Readout spin index must be between 0 and {spinsys.NSpins - 1}.")

    spinsys.B = _parse_numeric_sequence(params["B"], expected_len=3, default=[0.0, 0.0, 0.0])
    spinsys.BTip = _parse_numeric_sequence(
        params["tip_field"], expected_len=3, default=[0.0, 0.0, 0.0]
    )
    spinsys.tip = params["tip_type"]

    # Anisotropy
    d_values = _parse_numeric_sequence(
        params["D"], expected_len=spinsys.NSpins, default=[0.0]
    )
    e_values = _parse_numeric_sequence(
        params["E"], expected_len=spinsys.NSpins, default=[0.0]
    )
    spinsys.Dvector = np.array(d_values, dtype=float)
    spinsys.Evector = np.array(e_values, dtype=float)

    # Exchange interaction
    pair_count = spinsys.NSpins * (spinsys.NSpins - 1) // 2
    if pair_count > 0:
        j_values_raw = params.get("Jvector", "")
        if str(j_values_raw).strip():
            j_values = _parse_numeric_sequence(j_values_raw, allow_empty=False)
            if len(j_values) != 3 * pair_count:
                raise ValueError(
                    f"Exchange interaction expects {pair_count} triplets (3 values each). "
                    f"Received {len(j_values)} values."
                )
        else:
            j_values = [0.0] * (3 * pair_count)
        spinsys.Jvector = np.array(j_values, dtype=float).reshape(pair_count, 3)

    # Onsite potentials U
    u_values = _parse_numeric_sequence(
        params.get("U", 0.0), expected_len=spinsys.NSpins, default=[0.0]
    )
    spinsys.U = np.array(u_values, dtype=float)

    # Kondo parameter J
    kondo_values = _parse_numeric_sequence(
        params.get("KondoJ", 0.0), expected_len=spinsys.NSpins, default=[0.0]
    )
    spinsys.J = np.array(kondo_values, dtype=float)

    # Tunnelling / polarization parameters
    spinsys.V_DC = float(params["V_DC"])
    spinsys.TipPolarization = _parse_numeric_sequence(
        params["tip_polarization"], expected_len=3, default=[0.0, 0.0, 0.0]
    )
    spinsys.SamplePolarization = _parse_numeric_sequence(
        params["sample_polarization"], expected_len=3, default=[0.0, 0.0, 0.0]
    )

    return spinsys


# --------------------------------------------------------------------------- #
# Notebook export utilities
# --------------------------------------------------------------------------- #


def _format_python_literal(value: Any) -> str:
    """Return a Python literal string for simple data structures."""
    if isinstance(value, np.ndarray):
        return _format_python_literal(value.tolist())
    if isinstance(value, (list, tuple)):
        inner = ", ".join(_format_python_literal(item) for item in value)
        bracket_open, bracket_close = ("[", "]") if isinstance(value, list) else ("(", ")")
        if isinstance(value, tuple) and len(value) == 1:
            return f"{bracket_open}{inner},)"
        return f"{bracket_open}{inner}{bracket_close}"
    if isinstance(value, dict):
        items = ", ".join(
            f"{_format_python_literal(key)}: {_format_python_literal(val)}" for key, val in value.items()
        )
        return "{" + items + "}"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    return repr(value)


def _array_assignment(name: str, values: List[Any], dtype: str = "float") -> str:
    return f"{name} = np.array({_format_python_literal(values)}, dtype={dtype})"


def generate_notebook(spinsys: SpinSys, option: VisualizationOption, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Create a Jupyter notebook structure that recreates the current configuration."""
    spins = list(spinsys.Spins)
    d_vector = spinsys.Dvector.tolist() if getattr(spinsys, "Dvector", None) is not None else []
    e_vector = spinsys.Evector.tolist() if getattr(spinsys, "Evector", None) is not None else []
    j_vector = None
    if getattr(spinsys, "Jvector", None) is not None and spinsys.Jvector.size:
        j_vector = spinsys.Jvector.tolist()
    u_vector = spinsys.U.tolist() if getattr(spinsys, "U", None) is not None else []
    kondo_vector = spinsys.J.tolist() if getattr(spinsys, "J", None) is not None else []

    base_setup_lines = [
        f"spins = {_format_python_literal(spins)}",
        "spinsys = SpinSys(spins)",
        f"spinsys.T = {float(spinsys.T)}",
        f"spinsys.ReadoutSpin = {int(spinsys.ReadoutSpin)}",
        f"spinsys.B = {_format_python_literal(list(spinsys.B))}",
        f"spinsys.BTip = {_format_python_literal(list(spinsys.BTip))}",
        f"spinsys.tip = {repr(spinsys.tip)}",
    ]

    if d_vector:
        base_setup_lines.append(f"{_array_assignment('spinsys.Dvector', d_vector)}")
    if e_vector:
        base_setup_lines.append(f"{_array_assignment('spinsys.Evector', e_vector)}")
    if j_vector is not None:
        base_setup_lines.append(f"{_array_assignment('spinsys.Jvector', j_vector)}")
    if u_vector:
        base_setup_lines.append(f"{_array_assignment('spinsys.U', u_vector)}")
    if kondo_vector:
        base_setup_lines.append(f"{_array_assignment('spinsys.J', kondo_vector)}")

    base_setup_lines.extend(
        [
            f"spinsys.V_DC = {float(spinsys.V_DC)}",
            f"spinsys.TipPolarization = {_format_python_literal(list(spinsys.TipPolarization))}",
            f"spinsys.SamplePolarization = {_format_python_literal(list(spinsys.SamplePolarization))}",
        ]
    )

    kwargs_literal = _format_python_literal(kwargs) if kwargs else "{}"
    viz_call_lines = [
        f"# Generate visualization: {option.label}",
        f"kwargs = {kwargs_literal}",
        f"result = vis.{option.func.__name__}(spinsys, **kwargs)",
        "plt.show()",
    ]

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Spin Simulation Export\n",
                    "\n",
                    "This notebook reproduces the spin configuration and visualization exported from the GUI.\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "from spinsimulation.SpinSys import SpinSys\n",
                    "import spinsimulation.SpinVisualization as vis\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [line + "\n" for line in base_setup_lines],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [line + "\n" for line in viz_call_lines],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": sys.version.split()[0],
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    return notebook


# --------------------------------------------------------------------------- #
# Form widgets
# --------------------------------------------------------------------------- #


class ParameterPanel(QWidget):
    """
    Generic panel that renders inputs from a list of `ParameterSpec`.
    """

    def __init__(self, specs: List[ParameterSpec], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._specs = specs
        self._widgets: Dict[str, QWidget] = {}
        self._form = QFormLayout()
        self._form.setLabelAlignment(Qt.AlignRight)
        self.setLayout(self._form)
        self._build()

    def _build(self) -> None:
        for spec in self._specs:
            widget = self._create_widget(spec)
            if spec.tooltip:
                widget.setToolTip(spec.tooltip)
            self._widgets[spec.name] = widget
            self._form.addRow(spec.label, widget)

    def _create_widget(self, spec: ParameterSpec) -> QWidget:
        ftype = spec.field_type
        if ftype == "float":
            widget = QDoubleSpinBox()
            widget.setDecimals(spec.decimals)
            widget.setSingleStep(spec.step or 0.1)
            if spec.minimum is not None:
                widget.setMinimum(spec.minimum)
            if spec.maximum is not None:
                widget.setMaximum(spec.maximum)
            widget.setValue(float(spec.default or 0.0))
            return widget
        if ftype == "int":
            widget = QSpinBox()
            if spec.minimum is not None:
                widget.setMinimum(int(spec.minimum))
            if spec.maximum is not None:
                widget.setMaximum(int(spec.maximum))
            widget.setValue(int(spec.default or 0))
            return widget
        if ftype == "choice":
            widget = QComboBox()
            choices = spec.choices or []
            widget.addItems(choices)
            if spec.default in choices:
                widget.setCurrentText(spec.default)
            return widget
        if ftype == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(spec.default))
            return widget
        # default to text entry
        widget = QLineEdit()
        widget.setText(str(spec.default) if spec.default is not None else "")
        return widget

    def update_specs(self, specs: List[ParameterSpec]) -> None:
        """
        Replace the form contents with a new specification list.
        """
        # clear existing fields
        while self._form.count():
            item = self._form.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._specs = specs
        self._widgets.clear()
        self._build()

    def values(self) -> Dict[str, Any]:
        """
        Return current values with basic Python types.
        """
        collected: Dict[str, Any] = {}
        for spec in self._specs:
            widget = self._widgets[spec.name]
            ftype = spec.field_type
            if ftype == "float":
                collected[spec.name] = float(widget.value())
            elif ftype == "int":
                collected[spec.name] = int(widget.value())
            elif ftype == "choice":
                collected[spec.name] = str(widget.currentText())
            elif ftype == "bool":
                collected[spec.name] = bool(widget.isChecked())
            else:
                collected[spec.name] = str(widget.text()).strip()
        return collected


# --------------------------------------------------------------------------- #
# Plot display widget
# --------------------------------------------------------------------------- #


class PlotDisplay(QWidget):
    """
    Container that hosts a matplotlib `FigureCanvas`.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)
        self._canvas: Optional[FigureCanvas] = None

    def show_figure(self, figure: Figure) -> None:
        if self._canvas is not None:
            self._layout.removeWidget(self._canvas)
            self._canvas.setParent(None)
            self._canvas.deleteLater()
        self._canvas = FigureCanvas(figure)
        self._layout.addWidget(self._canvas)
        self._canvas.draw()


# --------------------------------------------------------------------------- #
# GUI main window
# --------------------------------------------------------------------------- #


BASE_PARAMETER_SPECS: List[ParameterSpec] = [
    ParameterSpec(
        name="spins",
        label="Spins (comma separated)",
        field_type="text",
        default="1",
        tooltip="Example: 1 or 1,0.5 for two spins.",
    ),
    ParameterSpec(
        name="temperature",
        label="Temperature [K]",
        field_type="float",
        default=1.0,
        minimum=0.01,
        maximum=500.0,
        step=0.1,
    ),
    ParameterSpec(
        name="B",
        label="Magnetic field Bx,By,Bz [T]",
        field_type="text",
        default="0,0,0",
        tooltip="Three values in Tesla.",
    ),
    ParameterSpec(
        name="tip_field",
        label="Tip field Btip x,y,z [T]",
        field_type="text",
        default="0,0,0",
    ),
    ParameterSpec(
        name="tip_type",
        label="Tip type",
        field_type="choice",
        default="f",
        choices=["f", "a"],
    ),
    ParameterSpec(
        name="readout_spin",
        label="Readout spin index",
        field_type="int",
        default=0,
        minimum=0,
        maximum=20,
    ),
    ParameterSpec(
        name="D",
        label="Anisotropy D (meV)",
        field_type="text",
        default="0",
        tooltip="One value (replicated) or one per spin.",
    ),
    ParameterSpec(
        name="E",
        label="Anisotropy E (meV)",
        field_type="text",
        default="0",
        tooltip="One value (replicated) or one per spin.",
    ),
    ParameterSpec(
        name="Jvector",
        label="Exchange Jx,Jy,Jz (per pair)",
        field_type="text",
        default="",
        tooltip="Triplets per spin pair in order (0-1, 0-2, ..., 1-2, ...). Example: 0.1,0.1,0.1; 0.0,0.0,0.0",
    ),
    ParameterSpec(
        name="U",
        label="Onsite potential U",
        field_type="text",
        default="0",
        tooltip="One value (replicated) or one per spin.",
    ),
    ParameterSpec(
        name="KondoJ",
        label="Kondo J parameter",
        field_type="text",
        default="0",
        tooltip="One value (replicated) or one per spin.",
    ),
    ParameterSpec(
        name="V_DC",
        label="Bias voltage V_DC [V]",
        field_type="float",
        default=0.0,
        minimum=-2.0,
        maximum=2.0,
        step=0.01,
    ),
    ParameterSpec(
        name="tip_polarization",
        label="Tip polarization Px,Py,Pz",
        field_type="text",
        default="0,0,0",
    ),
    ParameterSpec(
        name="sample_polarization",
        label="Sample polarization Px,Py,Pz",
        field_type="text",
        default="0,0,0",
    ),
]


def _build_zeeman_kwargs(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "Brange": _parse_numeric_sequence(raw["Brange"], expected_len=2, default=[0.0, 1.0]),
        "N": int(raw["N"]),
        "energyscale": raw["energyscale"],
    }


def _build_energy_kwargs(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": raw["axis"], "energyscale": raw["energyscale"]}


def _build_iets_kwargs(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "Vrange": _parse_numeric_sequence(raw["Vrange"], expected_len=2, default=[-30.0, 30.0]),
        "N": int(raw["N"]),
        "norm": bool(raw["norm"]),
        "order": raw["order"],
        "plotAllSpins": bool(raw["plotAllSpins"]),
    }


def _build_esr_freq_kwargs(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "FreqRange": _parse_numeric_sequence(raw["FreqRange"], expected_len=2, default=[10.0, 20.0]),
        "AllowPumping": bool(raw["AllowPumping"]),
        "N": int(raw["N"]),
        "lw": float(raw["lw"]),
        "norm": bool(raw["norm"]),
        "output": bool(raw["output"]),
    }


def _build_esr_tip_kwargs(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "FreqRange": _parse_numeric_sequence(raw["FreqRange"], expected_len=2, default=[10.0, 20.0]),
        "N_Freq": int(raw["N_Freq"]),
        "lw": float(raw["lw"]),
        "norm": bool(raw["norm"]),
        "BTipRange": _parse_numeric_sequence(raw["BTipRange"], expected_len=2, default=[-0.05, 0.05]),
        "N_B": int(raw["N_B"]),
        "AllowPumping": bool(raw["AllowPumping"]),
        "Angle": float(raw["Angle"]),
    }


VISUALIZATION_OPTIONS: List[VisualizationOption] = [
    VisualizationOption(
        key="eigen_prob",
        label="Eigenstate matrix (probability)",
        func=vis.plotEigenMatrix,
        fixed_kwargs={"plot_type": "p"},
        description="Heatmap of eigenstates vs. basis states using probabilities.",
    ),
    VisualizationOption(
        key="eigen_amp",
        label="Eigenstate matrix (amplitude)",
        func=vis.plotEigenMatrix,
        fixed_kwargs={"plot_type": "a"},
        description="Heatmap of eigenstates vs. basis states using amplitudes.",
    ),
    VisualizationOption(
        key="zeeman",
        label="Zeeman diagram",
        func=vis.plotZeemanDiagramm,
        parameters=[
            ParameterSpec(
                name="Brange",
                label="Field range [T]",
                field_type="text",
                default="0,1",
                tooltip="Start, end magnetic field in Tesla.",
            ),
            ParameterSpec(
                name="N",
                label="Sweep points",
                field_type="int",
                default=200,
                minimum=10,
                maximum=5000,
            ),
            ParameterSpec(
                name="energyscale",
                label="Energy scale",
                field_type="choice",
                default="meV",
                choices=["meV", "GHz"],
            ),
        ],
        arg_builder=_build_zeeman_kwargs,
    ),
    VisualizationOption(
        key="energy_vs_s",
        label="Energy vs ⟨S⟩",
        func=vis.plotEnergyVsS,
        parameters=[
            ParameterSpec(
                name="axis",
                label="Spin axis",
                field_type="choice",
                default="z",
                choices=["x", "y", "z"],
            ),
            ParameterSpec(
                name="energyscale",
                label="Energy scale",
                field_type="choice",
                default="meV",
                choices=["meV", "GHz"],
            ),
        ],
        arg_builder=_build_energy_kwargs,
    ),
    VisualizationOption(
        key="iets",
        label="IETS spectrum",
        func=vis.plotIETS,
        parameters=[
            ParameterSpec(
                name="Vrange",
                label="Bias range [mV]",
                field_type="text",
                default="-30,30",
                tooltip="Start, end voltage in millivolts.",
            ),
            ParameterSpec(
                name="N",
                label="Voltage points",
                field_type="int",
                default=400,
                minimum=10,
                maximum=5000,
            ),
            ParameterSpec(
                name="norm",
                label="Normalize",
                field_type="bool",
                default=False,
            ),
            ParameterSpec(
                name="order",
                label="Order",
                field_type="choice",
                default="2nd",
                choices=["2nd", "2ndR", "3rd", "3rdR"],
            ),
            ParameterSpec(
                name="plotAllSpins",
                label="All spins",
                field_type="bool",
                default=False,
            ),
        ],
        arg_builder=_build_iets_kwargs,
        description="Differential conductance spectrum using SpinVisualization.plotIETS.",
    ),
    VisualizationOption(
        key="esr_freq",
        label="ESR frequency sweep",
        func=vis.plotESR_FreqSweep_Benjamin,
        parameters=[
            ParameterSpec(
                name="FreqRange",
                label="Frequency range [GHz]",
                field_type="text",
                default="10,20",
                tooltip="Start, end frequency in GHz.",
            ),
            ParameterSpec(
                name="AllowPumping",
                label="Allow pumping",
                field_type="bool",
                default=False,
            ),
            ParameterSpec(
                name="N",
                label="Frequency points",
                field_type="int",
                default=1000,
                minimum=10,
                maximum=10000,
            ),
            ParameterSpec(
                name="lw",
                label="Linewidth",
                field_type="float",
                default=0.02,
                minimum=0.0,
                maximum=5.0,
                step=0.01,
            ),
            ParameterSpec(
                name="norm",
                label="Normalize",
                field_type="bool",
                default=True,
            ),
            ParameterSpec(
                name="output",
                label="Return transitions",
                field_type="bool",
                default=False,
                tooltip="Enable to see resonance table printed in the notebook.",
            ),
        ],
        arg_builder=_build_esr_freq_kwargs,
        description="Electron spin resonance frequency sweep using SpinVisualization.plotESR_FreqSweep_Benjamin.",
    ),
    VisualizationOption(
        key="esr_tip",
        label="ESR tip field sweep",
        func=vis.plotESR_Tipfieldsweep,
        parameters=[
            ParameterSpec(
                name="FreqRange",
                label="Frequency range [GHz]",
                field_type="text",
                default="10,20",
            ),
            ParameterSpec(
                name="N_Freq",
                label="Frequency points",
                field_type="int",
                default=500,
                minimum=10,
                maximum=2000,
            ),
            ParameterSpec(
                name="lw",
                label="Linewidth",
                field_type="float",
                default=0.02,
                minimum=0.0,
                maximum=5.0,
                step=0.01,
            ),
            ParameterSpec(
                name="norm",
                label="Normalize",
                field_type="bool",
                default=True,
            ),
            ParameterSpec(
                name="BTipRange",
                label="Tip field range [T]",
                field_type="text",
                default="-0.05,0.05",
                tooltip="Start, end tip field sweep range in Tesla.",
            ),
            ParameterSpec(
                name="N_B",
                label="Tip field points",
                field_type="int",
                default=100,
                minimum=10,
                maximum=2000,
            ),
            ParameterSpec(
                name="AllowPumping",
                label="Allow pumping",
                field_type="bool",
                default=False,
            ),
            ParameterSpec(
                name="Angle",
                label="Tip angle [deg]",
                field_type="float",
                default=0.0,
                minimum=-180.0,
                maximum=180.0,
                step=1.0,
            ),
        ],
        arg_builder=_build_esr_tip_kwargs,
        description="ESR intensity versus tip magnetic field using SpinVisualization.plotESR_Tipfieldsweep.",
    ),
]


class SpinVisualizationWindow(QMainWindow):
    """
    Main application window.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spin Simulation Viewer")
        self.resize(1100, 700)

        # Central widget with horizontal split
        central = QWidget()
        root_layout = QHBoxLayout()
        root_layout.setContentsMargins(10, 10, 10, 10)
        central.setLayout(root_layout)
        self.setCentralWidget(central)

        # Left: spin system configuration
        self.spin_params_panel = ParameterPanel(BASE_PARAMETER_SPECS)
        spin_group = QGroupBox("Spin system parameters")
        spin_layout = QVBoxLayout()
        spin_layout.addWidget(self.spin_params_panel)
        spin_group.setLayout(spin_layout)
        root_layout.addWidget(spin_group, stretch=1)

        # Right: visualization controls and plot
        right_column = QVBoxLayout()

        controls_row = QHBoxLayout()
        controls_row.addWidget(QLabel("Visualization:"))
        self.visualization_selector = QComboBox()
        for option in VISUALIZATION_OPTIONS:
            self.visualization_selector.addItem(option.label, option.key)
        controls_row.addWidget(self.visualization_selector, stretch=1)
        self.render_button = QPushButton("Render")
        controls_row.addWidget(self.render_button)
        self.export_button = QPushButton("Export Notebook")
        controls_row.addWidget(self.export_button)
        controls_container = QWidget()
        controls_container.setLayout(controls_row)
        right_column.addWidget(controls_container)

        self.current_option = VISUALIZATION_OPTIONS[0]

        self.viz_params_panel = ParameterPanel(self.current_option.parameters)
        viz_group = QGroupBox("Visualization parameters")
        viz_layout = QVBoxLayout()
        viz_layout.addWidget(self.viz_params_panel)
        if self.current_option.description:
            description = QLabel(self.current_option.description)
            description.setWordWrap(True)
            viz_layout.addWidget(description)
        viz_group.setLayout(viz_layout)
        right_column.addWidget(viz_group)

        self.plot_display = PlotDisplay()
        right_column.addWidget(self.plot_display, stretch=1)

        right_container = QWidget()
        right_container.setLayout(right_column)
        root_layout.addWidget(right_container, stretch=2)

        # Signals
        self.visualization_selector.currentIndexChanged.connect(self._on_visualization_changed)
        self.render_button.clicked.connect(self._on_render_requested)
        self.export_button.clicked.connect(self._on_export_requested)

    # ---- Callbacks ----------------------------------------------------- #

    def _on_visualization_changed(self, index: int) -> None:
        option = VISUALIZATION_OPTIONS[index]
        self.current_option = option
        self.viz_params_panel.update_specs(option.parameters)
        # Update description if provided
        group_box = self.viz_params_panel.parentWidget()
        if isinstance(group_box, QGroupBox):
            # Remove trailing description label if present
            layout = group_box.layout()
            if layout is not None:
                # Remove any description labels beyond the panel itself
                while layout.count() > 1:
                    item = layout.takeAt(1)
                    widget = item.widget()
                    if widget is not None:
                        widget.deleteLater()
                if option.description:
                    description = QLabel(option.description)
                    description.setWordWrap(True)
                    layout.addWidget(description)

    def _on_render_requested(self) -> None:
        try:
            base_values = self.spin_params_panel.values()
            spinsys = build_spin_system(base_values)
            option = self.current_option
            viz_values = self.viz_params_panel.values()
            kwargs = option.arg_builder(viz_values) if option.arg_builder else viz_values
            kwargs = {**kwargs, **option.fixed_kwargs}
            figure = self._run_visualization(spinsys, option.func, kwargs)
            if figure is None:
                raise RuntimeError("The selected visualization did not produce a matplotlib figure.")
            self.plot_display.show_figure(figure)
        except Exception as exc:  # pylint: disable=broad-except
            QMessageBox.critical(self, "Simulation error", str(exc))

    # ---- Visualization helpers ---------------------------------------- #

    def _run_visualization(
        self, spinsys: SpinSys, func: Callable[..., Any], kwargs: Dict[str, Any]
    ) -> Optional[Figure]:
        """
        Execute a SpinVisualization function and extract the resulting figure.

        Some SpinVisualization functions return a figure directly, while others
        rely on the pyplot state machine. We capture new figures created during
        the call and return the most recent one.
        """
        previous_figures = set(plt.get_fignums())
        original_show = plt.show
        plt.show = lambda *args, **unused: None  # disable blocking windows
        try:
            result = func(spinsys, **kwargs)
        finally:
            plt.show = original_show

        figure = self._extract_figure(result)
        if figure is not None:
            return figure

        new_figures = [num for num in plt.get_fignums() if num not in previous_figures]
        if not new_figures:
            return None
        # Use the most recently created figure
        latest_num = new_figures[-1]
        figure = plt.figure(latest_num)
        # Close any other stray figures
        for num in new_figures[:-1]:
            plt.close(num)
        return figure

    @staticmethod
    def _extract_figure(result: Any) -> Optional[Figure]:
        if isinstance(result, Figure):
            return result
        if isinstance(result, tuple):
            for item in result:
                if isinstance(item, Figure):
                    return item
        return None

    # ---- Export helpers ------------------------------------------------ #

    def _on_export_requested(self) -> None:
        option = self.current_option
        try:
            base_values = self.spin_params_panel.values()
            spinsys = build_spin_system(base_values)
            viz_values = self.viz_params_panel.values()
            kwargs = option.arg_builder(viz_values) if option.arg_builder else viz_values
            kwargs = {**kwargs, **option.fixed_kwargs}
        except Exception as exc:  # pylint: disable=broad-except
            QMessageBox.critical(self, "Invalid configuration", str(exc))
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Notebook",
            "spin_simulation_export.ipynb",
            "Jupyter Notebook (*.ipynb)",
        )
        if not path:
            return

        try:
            notebook = generate_notebook(spinsys, option, kwargs)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(notebook, handle, indent=2)
            QMessageBox.information(self, "Export complete", f"Notebook saved to:\n{path}")
        except Exception as exc:  # pylint: disable=broad-except
            QMessageBox.critical(self, "Export failed", str(exc))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    app = QApplication(sys.argv)
    window = SpinVisualizationWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

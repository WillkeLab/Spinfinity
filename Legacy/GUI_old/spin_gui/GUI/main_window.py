import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QSpinBox, QLabel, QPushButton, QFileDialog, QGroupBox, QFormLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import json

# --- Adapter placeholder ---
def simulate_spectrum(params):
    # TODO: Replace with real call to spinsimulation
    x = np.linspace(-10, 10, 200)
    y = 0.4 + 0.1 * np.tanh((x + 1)) - 0.25 * np.tanh((x - 1))
    return x, y

class SimulationThread(QThread):
    result_ready = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        x, y = simulate_spectrum(self.params)
        self.result_ready.emit(x, y)

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(5, 4))
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self.ax.set_xlabel('eV (meV)')
        self.ax.set_ylabel('dI/dV (a.u.)')
        self.ax.set_title('dI/dV Spectrum')

    def plot(self, x, y):
        self.ax.clear()
        self.ax.plot(x, y, 'k-')
        self.ax.set_xlabel('eV (meV)')
        self.ax.set_ylabel('dI/dV (a.u.)')
        self.ax.set_title('dI/dV Spectrum')
        self.ax.grid(True)
        self.draw()

class ControlPanel(QWidget):
    param_changed = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.params = {
            'S': 0.5,
            'D': 0.0,
            'E': 0.0,
            'J': -0.04,
            'T': 1.0,
            'Bz': 0.0,
            'Bx': 0.0,
            'By': 0.0,
        }
        layout = QVBoxLayout()
        form = QFormLayout()
        self.widgets = {}
        for key, val in self.params.items():
            if isinstance(val, float):
                slider = QSlider(Qt.Horizontal)
                slider.setMinimum(-100)
                slider.setMaximum(100)
                slider.setValue(int(val * 100))
                slider.valueChanged.connect(self.emit_params)
                spin = QSpinBox()
                spin.setMinimum(-100)
                spin.setMaximum(100)
                spin.setValue(int(val * 100))
                spin.valueChanged.connect(lambda v, k=key: self.slider_spin_sync(k, v))
                self.widgets[key] = (slider, spin)
                row = QHBoxLayout()
                row.addWidget(slider)
                row.addWidget(spin)
                form.addRow(QLabel(key), row)
        layout.addLayout(form)
        # Save/Load buttons
        btn_row = QHBoxLayout()
        self.save_btn = QPushButton('Save Preset')
        self.load_btn = QPushButton('Load Preset')
        self.save_btn.clicked.connect(self.save_preset)
        self.load_btn.clicked.connect(self.load_preset)
        btn_row.addWidget(self.save_btn)
        btn_row.addWidget(self.load_btn)
        layout.addLayout(btn_row)
        self.setLayout(layout)

    def slider_spin_sync(self, key, value):
        slider, spin = self.widgets[key]
        slider.setValue(value)
        self.emit_params()

    def emit_params(self):
        for key, (slider, spin) in self.widgets.items():
            val = slider.value() / 100.0
            spin.setValue(slider.value())
            self.params[key] = val
        self.param_changed.emit(self.params.copy())

    def save_preset(self):
        fname, _ = QFileDialog.getSaveFileName(self, 'Save Preset', '', 'JSON Files (*.json)')
        if fname:
            with open(fname, 'w') as f:
                json.dump(self.params, f)

    def load_preset(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Load Preset', '', 'JSON Files (*.json)')
        if fname:
            with open(fname, 'r') as f:
                self.params = json.load(f)
            for key, val in self.params.items():
                if key in self.widgets:
                    slider, spin = self.widgets[key]
                    slider.setValue(int(val * 100))
                    spin.setValue(int(val * 100))
            self.emit_params()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Spin Hamiltonian Simulator')
        self.resize(1000, 600)
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        # Left: Control panel
        self.control_panel = ControlPanel()
        self.control_panel.param_changed.connect(self.run_simulation)
        main_layout.addWidget(self.control_panel, 1)
        # Right: Plot area
        self.plot_canvas = PlotCanvas()
        main_layout.addWidget(self.plot_canvas, 3)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        # Initial simulation
        self.run_simulation(self.control_panel.params)

    def run_simulation(self, params):
        self.thread = SimulationThread(params)
        self.thread.result_ready.connect(self.update_plot)
        self.thread.start()

    def update_plot(self, x, y):
        self.plot_canvas.plot(x, y)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()




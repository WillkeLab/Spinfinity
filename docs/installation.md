# Installation

Spinfinity requires **Python 3.11 or newer**. Its numerical and plotting
dependencies are installed automatically.

We recommend using a separate Python environment for Spinfinity.

## Download the project

Clone the repository:

```bash
git clone https://github.com/WillkeLab/Spinfinity.git
cd Spinfinity
```

Alternatively, download the repository using **Code → Download ZIP** on
[GitHub](https://github.com/WillkeLab/Spinfinity), extract it, and open a
terminal in the extracted folder.

If you already have a local checkout, use that folder instead.

Run the remaining installation commands from the project root—the folder
containing `pyproject.toml`.

## Create an environment

Choose either a virtual environment or a Conda environment. You do not
need both.

### Virtual environment

On Windows, using PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If your Python installation uses `python` instead of `py`, use
`python -m venv .venv` to create the environment.

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Activate this environment whenever you open a new terminal to work with
Spinfinity.

### Conda environment

If you already use Conda, you can create a dedicated environment instead:

```bash
conda create -n spinfinity python=3.12 pip
conda activate spinfinity
```

Run these commands in a terminal where Conda is available, such as
Anaconda Prompt or Miniconda Prompt.

## Install Spinfinity

With your environment activated, run:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

The `-e` option installs Spinfinity in editable mode: changes to the local
Python source are used without reinstalling the package. Restart an
already-running notebook kernel after changing imported code.

If you change dependencies in `pyproject.toml`, run the installation
command again.

For a non-editable installation, use `python -m pip install .` instead.

## Verify the installation

Check that Spinfinity can be imported and inspect its location:

```bash
python -c "import spinfinity; print(spinfinity.__file__)"
```

For an editable installation, the printed path should point to your
local checkout.

Then run this example in Python or a notebook:

```python
from spinfinity import SpinSys, hamil

spin = SpinSys([0.5])
spin.B = [0.0, 0.0, 0.5]  # T
spin.BTip = [0.0, 0.0, 0.0]  # T
spin.gfactorvector = [2.0]

hamil.calc_EigEnergies(spin, rebuildH=True)

print("Relative eigenenergies (meV):", spin.E_All)
```

The two relative energies should be approximately
`[0.0, 0.05788382]` meV.

Continue with the
[first spin Hamiltonian tutorial](tutorials/01_first_spin_hamiltonian.ipynb).

## Run notebooks

For notebooks in VS Code, install Python kernel support in the same
environment:

```bash
python -m pip install ipykernel
```

Install the VS Code Python and Jupyter extensions. Open a notebook and
use **Select Kernel** to choose your Spinfinity environment.

If you prefer JupyterLab:

```bash
python -m pip install jupyterlab
python -m jupyterlab
```

## Build the documentation

This step is optional. From the project root, install the documentation
dependencies and Python kernel support:

```bash
python -m pip install -e ".[docs]" ipykernel
```

Start the local documentation server:

```bash
python -m mkdocs serve
```

Open the address printed in the terminal, normally
`http://127.0.0.1:8000`. Stop the server with `Ctrl+C`.

To generate the static website without starting a server:

```bash
python -m mkdocs build
```

The website is generated in `site/`. Tutorial notebooks are executed
during the build, so building takes longer than rendering Markdown alone.
The authors recommend to install the "spinfinity" package in a python
environment. Create and activate a Python environment with Python 3.10 or newer, 
then install the package from the project root:

```bash
conda install jupyter pip
cd folderpath 
pip install -e .  
```

The Spinfinty project comes with a documentation using the mkdocs package.
Install it to your environment via:

```bash
cd folderpath 
pip install .[docs]  
```

To generate the local documentation website run:

```bash
mkdocs serve
```
You can then access the documentation via the URL provided. 
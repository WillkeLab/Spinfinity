# Spinfinity Conventions And Style Guide

This document defines the working conventions for contributors and coding agents
editing the `spinfinity` Python package. The goal is consistency with the
existing codebase while steadily improving readability, API clarity, and
documentation quality.

## Scope

- Apply these rules to files in `spinfinity/`, examples, and package docs.
- Prefer small, localized changes over broad style-only rewrites.
- Do not refactor unrelated legacy code unless it is necessary for the task.

## Package Character

`spinfinity` is a numerics-heavy scientific Python package for quantum spin
systems in tunnel-junction and ESR-style simulations. The package is organized
around a mutable `SpinSys` state container plus module-level calculation and
plotting functions.

The existing public API is function-oriented, not class-hierarchical. Preserve
that shape unless a larger redesign is explicitly requested.

## Docstring Convention


## Docstring Convention

### Function docstrings

```python
"""
One-line description of the function.

Multi-line description of the function.

Parameters
----------
a : type
    Short description, including unit and default value if relevant.

Returns
-------
b : type
    Short description.

Raises
------
ValueError
    Description of when the error is raised.

Notes
-----
The function updates the following attributes of `spinsys`:

- ...
"""
```

### Moduele Dosctrings
All Modules should follow this docstring except SpinSys
```python
"""
One-line description of the module.

Multi-line description of the module.

Functions
---------
function_name
    One-line description.

Dependencies
------------
- Uses `spinfinity...` for eigenvalue and eigenvector calculations.
"""


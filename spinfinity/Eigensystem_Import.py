"""Parse eigenenergies and eigenstates from CASSCF/RASSI output."""

import re
from fractions import Fraction
from pathlib import Path

import numpy as np


class Eigensystem_Import:
    """
    Parser for the CASSCF/RASSI output around.

      'CALCULATION OF CRYSTAL-FIELD PARAMETERS OF THE GROUND ATOMIC MULTIPLET J =  x/y.'
    and
      'DECOMPOSITION OF THE RASSI WAVE FUNCTIONS ... J = x/y'

    It extracts:
      - eigenenergies (from the 'ENERGY SPLITTING ... J =  x/y' table,
        RASSI INITIAL column, in cm^-1)
      - eigenstates in the |J, M> basis (M, complex coeff, weight%)

    Parameters
    ----------
    text : str
        Full content of the log file.
    j_label : str, optional
        The J value as it appears in the output, e.g. "7/2", "5/2", "15/2".
        Default is "7/2".
    """

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def __init__(self, text: str, j_label: str = "7/2"):
        self.text = text
        self.j_label = j_label  # e.g. "7/2"
        self._energies = None  # {state_index: energy_cm1}
        # {state_index: {"M": [...],"coeffs": [...],"weights": [...],"energy": float}}
        self._states = None
        self._stevens = None  #

    @classmethod
    def from_file(cls, filename: str, j_label: str = "7/2"):
        """
        Create parser from a .log file.

        Parameters
        ----------
        filename : str
            Path to the log file.
        j_label : str, optional
            The J value as it appears in the output, e.g. "7/2", "5/2", "15/2".
            Default is "7/2".

        Returns
        -------
        Eigensystem_Import
            An instance of the parser with the file content loaded.
        """
        txt = Path(filename).read_text(errors="ignore")
        return cls(txt, j_label=j_label)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_M_label(m_raw: str):
        """
        Convert a string like '-7/2>' or ' 3/2>' into a float (-3.5, 1.5, ...).

        Parameters
        ----------
        m_raw : str
            The raw string representing the M quantum number, possibly with whitespace
            and a trailing '>'.

        Returns
        -------
        float or str
            The parsed float value of M, or the original stripped string
            if parsing fails.
        """
        s = m_raw.strip().rstrip(">")
        s = s.replace(" ", "")
        try:
            return float(Fraction(s))
        except Exception:
            try:
                return float(s)
            except Exception:
                return m_raw.strip()

    # ------------------------------------------------------------------
    # public: energies
    # ------------------------------------------------------------------
    def get_energies(self):
        """
        Return a dict: {state_index: energy_cm1}.

        Energies are taken from the table:

          ENERGY SPLITTING INDUCED BY CUMMULATIVE INDIVIDUAL-RANK OPERATORS (in cm-1).
          ...
           J =  x/y |      RASSI    |     ONLY      |     ONLY      |     ONLY      |
                    |     INITIAL   |      O2       |     O2+O4     |   O2+O4+O6    |
          ...
           w.f.  3  |   62.42978561 |   ...

        We read the "RASSI INITIAL" column (2nd numeric field after the first '|').

        Returns
        -------
        dict
            Mapping of state index (1-based) to energy in cm^-1.

        Raises
        ------
        RuntimeError
            If the ENERGY SPLITTING table or the specific J block cannot be found,
            or if no energies can be parsed.
        """
        if self._energies is not None:
            return self._energies

        text = self.text
        energies = {}

        # Find the ENERGY SPLITTING section
        m_head = re.search(
            r"ENERGY SPLITTING INDUCED BY CUMMULATIVE INDIVIDUAL-RANK "
            r"OPERATORS \(in cm-1\)\.",
            text,
        )
        if not m_head:
            raise RuntimeError(
                "Could not find 'ENERGY SPLITTING INDUCED BY CUMMULATIVE "
                "INDIVIDUAL-RANK OPERATORS' in file."
            )

        # Work on a chunk after this header
        section = text[m_head.end() : m_head.end() + 8000]
        lines = section.splitlines()

        in_this_J = False
        in_data = False

        j_pattern = rf"J\s*=\s*{re.escape(self.j_label)}"

        for line in lines:
            # Look for the specific J block, e.g. "J =  7/2 |      RASSI"
            if re.search(j_pattern, line) and "RASSI" in line:
                in_this_J = True
                in_data = False
                continue

            if not in_this_J:
                continue

            # Look for dashed separator lines; data starts after the first one,
            # and we stop at the next.
            if set(line.strip()) <= set("-| "):
                if not in_data:
                    in_data = True
                    continue
                else:
                    if energies:
                        break
                    else:
                        continue

            if not in_data:
                # still in header rows
                continue

            striped = line.strip()
            if not striped:
                if energies:
                    break
                continue

            if not striped.startswith("w.f."):
                continue

            # split by '|' to get columns
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                continue

            # parts[0]: "w.f.  3"
            # parts[1]: "62.42978561"  (RASSI INITIAL)
            m_idx = re.search(r"w\.f\.\s*(\d+)", parts[0])
            if not m_idx:
                continue
            idx = int(m_idx.group(1))

            try:
                e_cm1 = float(parts[1])
            except ValueError:
                continue

            energies[idx] = e_cm1

        if not energies:
            raise RuntimeError(
                f"Could not parse any w.f. energies from the J = {self.j_label} "
                "ENERGY SPLITTING table."
            )

        self._energies = energies
        return energies

    # ------------------------------------------------------------------
    # public: composition of eigenstates
    # ------------------------------------------------------------------
    def get_states(self):
        """
        Return a dict.

          {
            state_index: {
                "M":        [M1, M2, ...],        # float
                "coeffs":   [c1, c2, ...],        # complex
                "weights":  [w1, w2, ...],        # float (percent)
                "energy":   float or None         # cm^-1, from RASSI INITIAL column
            },
            ...
          }

        Returns
        -------
        dict
            Mapping of state index (1-based) to a dictionary containing:
            - "M": list of magnetic quantum numbers (floats)
            - "coeffs": list of complex coefficients
            - "weights": list of weights (percentages, floats)
            - "energy": energy in cm^-1 (float) or None if not found

        Raises
        ------
        RuntimeError
            If the DECOMPOSITION OF THE RASSI WAVE FUNCTIONS section for the specified
            J cannot be found, or if no states can be parsed.
        """
        if self._states is not None:
            return self._states

        lines = self.text.splitlines()
        states = {}

        j_pattern = rf"J\s*=\s*{re.escape(self.j_label)}"

        # find start of decomposition block
        start_idx = None
        for i, line in enumerate(lines):
            if (
                "DECOMPOSITION OF THE RASSI WAVE FUNCTIONS" in line
                and "LOWEST ATOMIC MULTIPLET" in line
                and re.search(j_pattern, line)
            ):
                start_idx = i
                break

        if start_idx is None:
            raise RuntimeError(
                "Could not find 'DECOMPOSITION OF THE RASSI WAVE FUNCTIONS ... "
                f"J = {self.j_label}' in file."
            )

        i = start_idx
        while i < len(lines):
            line = lines[i]

            # end when we hit some later section
            if "CUMULATIVE WEIGHT OF INDIVIDUAL-RANK OPERATORS" in line:
                break

            # header of each pair of states:
            #  | JM > | ab initio state  1 | ab initio state  2 |
            if "| JM >" in line and "ab initio state" in line:
                ids = [int(x) for x in re.findall(r"ab initio state\s+(\d+)", line)]
                for st in ids:
                    states.setdefault(st, {"M": [], "coeffs": [], "weights": []})

                # skip dashed line(s)
                j = i + 1
                while j < len(lines) and set(lines[j].strip()) <= set("-| "):
                    j += 1

                # parse rows with JM, coeff, weight for these states
                while j < len(lines):
                    row = lines[j]
                    # stop if dashed line, blank, or new header
                    if (
                        (not row.strip())
                        or (set(row.strip()) <= set("-| "))
                        or ("| JM >" in row and "ab initio state" in row)
                    ):
                        break

                    if "%" not in row:
                        j += 1
                        continue

                    parts = row.split("|")
                    if len(parts) < 3:
                        j += 1
                        continue

                    M_val = self._parse_M_label(parts[1])

                    # for each state in this header line
                    for s_idx, st in enumerate(ids):
                        base = 2 + 2 * s_idx  # coef column, then weight column
                        if base + 1 >= len(parts):
                            continue

                        coef_field = parts[base].strip()
                        weight_field = parts[base + 1].strip()

                        # parse complex coefficient
                        coef_core = coef_field.replace("*I", "").strip()
                        if coef_core:
                            toks = coef_core.split()
                            if len(toks) >= 2:
                                r = float(toks[0])
                                im = float(toks[1])
                            elif len(toks) == 1:
                                r = float(toks[0])
                                im = 0.0
                            else:
                                r = im = 0.0
                            coef = complex(r, im)
                        else:
                            coef = 0.0 + 0.0j

                        # parse weight (percentage)
                        w_str = weight_field.replace("%", "").strip()
                        w = float(w_str) if w_str else 0.0

                        states[st]["M"].append(M_val)
                        states[st]["coeffs"].append(coef)
                        states[st]["weights"].append(w)

                    j += 1

                i = j
            else:
                i += 1

        # attach energies (RASSI INITIAL) to each state
        energies = self.get_energies()
        for st, data in states.items():
            data["energy"] = energies.get(st, None)

        self._states = states
        return self._states

    @staticmethod
    def format_states(data):
        """
        Convert a dictionary of quantum states into an array of formatted strings.

        Parameters
        ----------
        data : dict
            A dictionary where each key corresponds to a state index (1-based), and
            each value is another dictionary with the following fields:

            - 'M' : list of float
                The magnetic quantum number values.
            - 'coeffs' : list of complex
                Complex coefficients corresponding to each M.
            - 'weights' : list of float
                (Unused in formatting; included for completeness.)
            - 'energy' : float
                Energy value of the state.

        Returns
        -------
        numpy.ndarray of object
            Each element is a formatted string of the form:

            ``"|n> (E = energy) = +a.bbb+ c.dddj|M1> + ... "``

            where:
            - `n` is a 0-based state index,
            - coefficients are formatted with sign and 3 decimal places,
            - only non-zero coefficients are included.

        Notes
        -----
        - The input dictionary is assumed to be ordered by integer keys.
        - Dictionary key `1` corresponds to state `|0>`, key `2` → `|1>`, etc.
        - Modify the formatting precision by changing the `:.3f` specifier.

        Examples
        --------
        >>> arr = format_states(data)
        >>> print(arr[0])
        |0> (E = 0.0) = +1.000+0.000j|0.5>
        """
        formatted = []

        for idx, state in data.items():
            energy = state["energy"]
            M_vals = state["M"]
            coeffs = state["coeffs"]

            terms = []
            for M, c in zip(M_vals, coeffs, strict=True):
                if abs(c) > 0:  # include all non-zero coefficients
                    # Format coefficient with sign, 3 decimals
                    real = f"{c.real:+.3f}"
                    imag = f"{c.imag:+.3f}"
                    coeff_str = f"{real}{imag}j"
                    terms.append(f"{coeff_str}|{M}>")

            # Join all terms in a single string
            rhs = " + ".join(terms)

            # State index: shift idx→0-based (so state "1" → |0>)
            label = f"|{idx - 1}> (E = {energy}) = {rhs}"

            formatted.append(label)

        return np.array(formatted, dtype=object)

    @staticmethod
    def extract_eigenvectors(data):
        """
        Return eigenvectors as columns, ordered by increasing state label.

        Rows follow descending M: +J, ..., -J, matching Spinfinity.
        Each state must contain the same M values; input row order may differ.
        Coefficients are used as provided, without normalization.

        Parameters
        ----------
        data : dict
            A dictionary where each key corresponds to a state index (1-based), and
            each value is another dictionary with the following fields:

            - 'M' : list of float
                The magnetic quantum number values.
            - 'coeffs' : list of complex
                Complex coefficients corresponding to each M.

        Returns
        -------
        mat : numpy.ndarray
            A 2D array of shape (len(M_basis), len(keys)), where each column
            corresponds to a state and each row corresponds to a magnetic quantum
            number M, ordered from +J to -J.

        Raises
        ------
        ValueError
            If the M basis is not consistent across all states in the input data.
        """
        keys = sorted(data)
        M_basis = sorted(data[keys[0]]["M"], reverse=True)
        mat = np.zeros((len(M_basis), len(keys)), dtype=complex)

        for col, key in enumerate(keys):
            state = data[key]
            coefficients = dict(zip(state["M"], state["coeffs"], strict=True))

            if set(coefficients) != set(M_basis):
                raise ValueError("All states must contain the same M basis.")

            mat[:, col] = [coefficients[M] for M in M_basis]

        return mat

    @staticmethod
    def convert_to_meV(energy_dict):
        """
        Convert energies from cm^-1 to meV, ordered by increasing state label.

        The ordering matches the columns returned by extract_eigenvectors.

        Parameters
        ----------
        energy_dict : dict
            A dictionary mapping state index (1-based) to energy in cm^-1.

        Returns
        -------
        numpy.ndarray
            A 1D array of energies in meV, ordered by increasing state index.
        """
        factor = 0.1239842
        keys = sorted(energy_dict)

        return np.array(
            [energy_dict[key] * factor for key in keys],
            dtype=float,
        )

    def get_stevens_parameters(self):
        """
        Parse the "Extended Stevens Operators (ESO)" table with header:.

            k |  q  |    (K)^2    |         B(k,q)

        inside the section:
            'CALCULATION OF CRYSTAL-FIELD PARAMETERS OF THE GROUND ATOMIC MULTIPLET
            J = x/y.'

        Returns
        -------
        dict
            Dictionary mapping "k,q" -> float, e.g. {"2,-2": -1.23e-3, "6,6": ...}

        Raises
        ------
        RuntimeError
            If the Stevens table cannot be found or parsed for the specified J.
        """
        if self._stevens is not None:
            return self._stevens

        text = self.text
        j_pat = re.escape(self.j_label)

        # 1) Find the correct J block
        m0 = re.search(
            rf"CALCULATION OF CRYSTAL-FIELD PARAMETERS OF THE GROUND "
            rf"ATOMIC MULTIPLET\s+J\s*=\s*{j_pat}\.",
            text,
        )
        if not m0:
            raise RuntimeError(
                "Could not find 'CALCULATION OF CRYSTAL-FIELD PARAMETERS ... "
                f"J = {self.j_label}.'"
            )

        # 2) Work on a chunk after that header
        section = text[m0.end() : m0.end() + 60000]  # generous window

        # 3) Find the ESO B(k,q) table header (the one with (K)^2)
        #    We anchor on the exact header line content,
        #       then parse until the closing dashed line.
        header_re = re.compile(
            r"^\s*k\s*\|\s*q\s*\|\s*\(?K\)?\^2\s*\|\s*B\(k,q\)\s*$",
            re.IGNORECASE | re.MULTILINE,
        )

        mh = header_re.search(section)
        if not mh:
            # some MOLCAS variants have extra spaces: try a looser header match
            mh = re.search(
                r"^\s*k\s*\|\s*q\s*\|.*\(\s*K\s*\)\^2.*B\(k,q\)",
                section,
                re.IGNORECASE | re.MULTILINE,
            )

        if not mh:
            raise RuntimeError(
                "Could not find the extended Stevens table header "
                f"(with (K)^2 and B(k,q)) for J = {self.j_label}."
            )

        # Parse from header line onward
        table_text = section[mh.start() :]
        stevens = self._parse_stevens_bkq_table(table_text)

        if not stevens:
            raise RuntimeError(
                "Found the Stevens table header, but could not parse any "
                f"B(k,q) rows for J = {self.j_label}."
            )

        self._stevens = stevens
        return stevens

    @staticmethod
    def _parse_stevens_bkq_table(table_text: str):
        """
        Parse the extended Stevens table blocks.

        often separated by dashed lines,
        returning dict {"k,q": float} for all ranks (2,4,6,...).

        Parameters
        ----------
        table_text : str
            The text of the Stevens table starting from the header line.

        Returns
        -------
        dict
            Mapping of "k,q" strings to their corresponding B(k,q) float values.
        """
        lines = table_text.splitlines()

        # Data lines look like:
        #   2 |  -2 |  1.50 | -1.99628921154922E-03 |
        row_re = re.compile(
            r"^\s*(\d+)\s*\|\s*([+-]?\d+)\s*\|\s*([+-]?\d+(?:\.\d+)?)?\s*\|\s*([+-]?\d+(?:\.\d+)?E[+-]\d+)\s*\|?\s*$",
            re.IGNORECASE,
        )

        # Stop when we clearly leave the B(k,q) table and enter something else
        stop_markers = (
            "A(k,q)",  # often the next table
            "WYBOURNE",  # other CF conventions
            "CRYSTAL-FIELD HAMILTONIAN",
            "EIGENVALUES",
            "WAVE FUNCTION",
            "CALCULATION OF CRYSTAL-FIELD PARAMETERS",  # next J block
        )

        out = {}
        in_data = False
        saw_any_row = False

        for ln in lines:
            s = ln.strip()

            # detect leaving this table after we already parsed something
            if saw_any_row:
                upper = s.upper()
                if any(m.upper() in upper for m in stop_markers):
                    break

            # Header separator: first dashed line under the header starts data mode
            if not in_data:
                if s and set(s) <= set("-|"):
                    in_data = True
                continue

            # Inside the table: dashed separators can appear between blocks
            # -> SKIP, don't break
            if s and set(s) <= set("-|"):
                continue

            if not s:
                continue

            m = row_re.match(ln)
            if not m:
                # allow some non-row noise lines without aborting parsing
                continue

            k = int(m.group(1))
            q = int(m.group(2))
            val = float(m.group(4))

            out[f"{k},{q}"] = val
            saw_any_row = True

        return out

    def get_stevens_parameters_meV(self, factor: float = 0.1239842):
        """
        Return extended Stevens parameters converted to meV.

        Parameters
        ----------
        factor : float, optional
            Conversion factor from cm^-1 to meV.
            Default: 0.1239842 meV / cm^-1

        Returns
        -------
        dict
            Dictionary {"k,q": B_kq_meV}
        """
        Bkq = self.get_stevens_parameters()

        return {kq: val * factor for kq, val in Bkq.items()}

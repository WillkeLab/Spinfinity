import re
from fractions import Fraction
from pathlib import Path
import numpy as np

class Eigensystem_Import:
    """
    Parser for the CASSCF/RASSI output around

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
        self._energies = None      # {state_index: energy_cm1}
        self._states = None        # {state_index: {"M": [...], "coeffs": [...], "weights": [...], "energy": float}}
        self._stevens = None  # 

    @classmethod
    def from_file(cls, filename: str, j_label: str = "7/2"):
        """Create parser from a .log file."""
        txt = Path(filename).read_text(errors="ignore")
        return cls(txt, j_label=j_label)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_M_label(m_raw: str):
        """Convert a string like '-7/2>' or ' 3/2>' into a float (-3.5, 1.5, ...)."""
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
        Return a dict: {state_index: energy_cm1}

        Energies are taken from the table:

          ENERGY SPLITTING INDUCED BY CUMMULATIVE INDIVIDUAL-RANK OPERATORS (in cm-1).
          ...
           J =  x/y |      RASSI    |     ONLY      |     ONLY      |     ONLY      |
                    |     INITIAL   |      O2       |     O2+O4     |   O2+O4+O6    |
          ...
           w.f.  3  |   62.42978561 |   ...

        We read the "RASSI INITIAL" column (2nd numeric field after the first '|').
        """
        if self._energies is not None:
            return self._energies

        text = self.text
        energies = {}

        # Find the ENERGY SPLITTING section
        m_head = re.search(
            r"ENERGY SPLITTING INDUCED BY CUMMULATIVE INDIVIDUAL-RANK OPERATORS \(in cm-1\)\.",
            text
        )
        if not m_head:
            raise RuntimeError(
                "Could not find 'ENERGY SPLITTING INDUCED BY CUMMULATIVE INDIVIDUAL-RANK OPERATORS' in file."
            )

        # Work on a chunk after this header
        section = text[m_head.end(): m_head.end() + 8000]
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
                f"Could not parse any w.f. energies from the J = {self.j_label} ENERGY SPLITTING table."
            )

        self._energies = energies
        return energies

    # ------------------------------------------------------------------
    # public: composition of eigenstates
    # ------------------------------------------------------------------
    def get_states(self):
        """
        Return a dict:
          {
            state_index: {
                "M":        [M1, M2, ...],        # float
                "coeffs":   [c1, c2, ...],        # complex
                "weights":  [w1, w2, ...],        # float (percent)
                "energy":   float or None         # cm^-1, from RASSI INITIAL column
            },
            ...
          }
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
                f"Could not find 'DECOMPOSITION OF THE RASSI WAVE FUNCTIONS ... J = {self.j_label}' in file."
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
                    if (not row.strip()) or (set(row.strip()) <= set("-| ")) \
                       or ("| JM >" in row and "ab initio state" in row):
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
            for M, c in zip(M_vals, coeffs):
                if abs(c) > 0:  # include all non-zero coefficients
                    # Format coefficient with sign, 3 decimals
                    real = f"{c.real:+.3f}"
                    imag = f"{c.imag:+.3f}"
                    coeff_str = f"{real}{imag}j"
                    terms.append(f"{coeff_str}|{M}>")

            # Join all terms in a single string
            rhs = " + ".join(terms)

            # State index: shift idx→0-based (so state "1" → |0>)
            label = f"|{idx-1}> (E = {energy}) = {rhs}"

            formatted.append(label)

        return np.array(formatted, dtype=object)

    def extract_eigenvectors(data):
        """
        Convert a dictionary of quantum states into a matrix of eigenvectors.

        Parameters
        ----------
        data : dict
            A dictionary where each key corresponds to a state index (1-based), and
            each value contains:
            
            - 'M' : list of float
                Magnetic quantum numbers (defines ordering of basis states).
            - 'coeffs' : list of complex
                Complex expansion coefficients for this state.

        Returns
        -------
        numpy.ndarray
            A 2D array of shape (N_basis, N_states) where each column is one
            eigenvector, ordered consistently with `format_states()`:
            
            - Dictionary key `1` → column 0  
            - Dictionary key `2` → column 1  
            - etc.
            
            Coefficients inside each column follow the order of the 'M' list.

        Notes
        -----
        - No normalization is performed; the coefficients are used as provided.
        - All states must share the same M-basis ordering.

        Examples
        --------
        >>> vecs = extract_eigenvectors(data)
        >>> print(vecs.shape)
        (8, 8)

        >>> print(vecs[:, 0])  # first eigenvector (state |0>)
        """
        
        # Ensure consistent ordering by sorting integer keys
        keys = sorted(data.keys())

        # Number of basis states from length of the 'M' list
        N_basis = len(data[keys[0]]["M"])
        N_states = len(keys)

        # Allocate output matrix
        mat = np.zeros((N_basis, N_states), dtype=complex)

        # Fill columns
        for col, key in enumerate(keys):
            coeffs = data[key]["coeffs"]
            mat[:, col] = np.array(coeffs, dtype=complex)

        return mat.T

    def convert_to_meV(energy_dict):
        """
        Convert energy values from a dictionary into meV.

        Parameters
        ----------
        energy_dict : dict
            Dictionary of energies, where the values are numeric and
            will be converted to meV by multiplying with the factor 0.1239842.

        Returns
        -------
        numpy.ndarray
            1D array containing the converted energies in meV, in the
            same order as the dictionary's values.

        Notes
        -----
        The conversion uses the constant factor:
            1 unit = 0.1239842 meV

        Examples
        --------
        >>> energies = {1: 0.0, 2: 0.0, 3: 62.42978561}
        >>> convert_to_meV(energies)
        array([0.        , 0.        , 7.7360607])
        """
        factor = 0.1239842
        energies_meV = [value * factor for value in energy_dict.values()]
        return np.array(energies_meV)

    def get_stevens_parameters(self):
        """
        Parse the "Extended Stevens Operators (ESO)" table with header:

            k |  q  |    (K)^2    |         B(k,q)

        inside the section:
            'CALCULATION OF CRYSTAL-FIELD PARAMETERS OF THE GROUND ATOMIC MULTIPLET J = x/y.'

        Returns
        -------
        dict
            Dictionary mapping "k,q" -> float, e.g. {"2,-2": -1.23e-3, "6,6": ...}
        """
        if self._stevens is not None:
            return self._stevens

        text = self.text
        j_pat = re.escape(self.j_label)

        # 1) Find the correct J block
        m0 = re.search(
            rf"CALCULATION OF CRYSTAL-FIELD PARAMETERS OF THE GROUND ATOMIC MULTIPLET\s+J\s*=\s*{j_pat}\.",
            text
        )
        if not m0:
            raise RuntimeError(
                f"Could not find 'CALCULATION OF CRYSTAL-FIELD PARAMETERS ... J = {self.j_label}.'"
            )

        # 2) Work on a chunk after that header
        section = text[m0.end(): m0.end() + 60000]  # generous window

        # 3) Find the ESO B(k,q) table header (the one with (K)^2)
        #    We anchor on the exact header line content, then parse until the closing dashed line.
        header_re = re.compile(
            r"^\s*k\s*\|\s*q\s*\|\s*\(?K\)?\^2\s*\|\s*B\(k,q\)\s*$",
            re.IGNORECASE | re.MULTILINE
        )

        mh = header_re.search(section)
        if not mh:
            # some MOLCAS variants have extra spaces: try a looser header match
            mh = re.search(r"^\s*k\s*\|\s*q\s*\|.*\(\s*K\s*\)\^2.*B\(k,q\)", section, re.IGNORECASE | re.MULTILINE)

        if not mh:
            raise RuntimeError(
                f"Could not find the extended Stevens table header (with (K)^2 and B(k,q)) for J = {self.j_label}."
            )

        # Parse from header line onward
        table_text = section[mh.start():]
        stevens = self._parse_stevens_bkq_table(table_text)

        if not stevens:
            raise RuntimeError(
                f"Found the Stevens table header, but could not parse any B(k,q) rows for J = {self.j_label}."
            )

        self._stevens = stevens
        return stevens

    @staticmethod
    def _parse_stevens_bkq_table(table_text: str):
        """
        Parse the extended Stevens table blocks (often separated by dashed lines),
        returning dict {"k,q": float} for all ranks (2,4,6,...).
        """
        lines = table_text.splitlines()

        # Data lines look like:
        #   2 |  -2 |  1.50 | -1.99628921154922E-03 |
        row_re = re.compile(
            r"^\s*(\d+)\s*\|\s*([+-]?\d+)\s*\|\s*([+-]?\d+(?:\.\d+)?)?\s*\|\s*([+-]?\d+(?:\.\d+)?E[+-]\d+)\s*\|?\s*$",
            re.IGNORECASE
        )

        # Stop when we clearly leave the B(k,q) table and enter something else
        stop_markers = (
            "A(k,q)",                 # often the next table
            "WYBOURNE",               # other CF conventions
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

            # Inside the table: dashed separators can appear between blocks -> SKIP, don't break
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

            return {
                kq: val * factor
                for kq, val in Bkq.items()
            }
import re
from fractions import Fraction
from pathlib import Path

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

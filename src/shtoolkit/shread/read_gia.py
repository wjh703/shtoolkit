import re
from pathlib import Path

import numpy as np


def read_gia_model(filepath: str | Path, lmax: int, model: str) -> np.ndarray:
    gia = np.zeros((2, lmax + 1, lmax + 1), dtype=np.float64)

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    if model in ["ICE6G-D", "ICE6G-C"]:
        if model == "ICE6G-D":
            regex_deg_0_to_2 = (
                r"GRACE Approximation for degrees 0 to 2\s+((?:\s*\d+\s+\d+\s+[+\-\d\.E]+\s+[+\-\d\.E]+\s*)+)"
            )
            regex_deg_greater_2 = (
                r"GRACE Approximation/Absolute Sea-level Values for degrees > 2\s+"
                r"((?:\s*\d+\s+\d+\s+[+\-\d\.E]+\s+[+\-\d\.E]+\s*)+)"
            )
        else:
            regex_deg_0_to_2 = (
                r"GRACE Coefficients for degrees 0 to 2\s+((?:\s*\d+\s+\d+\s+[+\-\d\.E]+\s+[+\-\d\.E]+\s*)+)"
            )
            regex_deg_greater_2 = (
                r"GRACE/GEOID Velocity Coefficients for degrees > 2"
                r"((?:\s*\d+\s+\d+\s+[+\-\d\.E]+\s+[+\-\d\.E]+\s*)+)"
            )

        deg_0_to_2 = re.findall(regex_deg_0_to_2, content)
        deg_greater_2 = re.findall(regex_deg_greater_2, content)

        if deg_0_to_2 and deg_greater_2:
            deg_full = deg_0_to_2[0] + deg_greater_2[0]
        else:
            msg = f"Do not match all cilms ranging from degree 0 to the {lmax}"
            raise ValueError(msg)

        for line in deg_full.split("\n"):
            ls = line.split()
            if not ls:
                continue
            l, m = int(ls[0]), int(ls[1])
            if m > lmax:
                break
            if l > lmax:
                continue
            gia[:, l, m] = float(ls[2]), float(ls[3])
    elif model == "C18":
        regex_deg_full = r"\s*(\d+)\s+([+-]?\d+)\s+([+-]?\d+\.\d+e[+-]?\d+)\s*"
        deg_full = re.findall(regex_deg_full, content)
        if not deg_full:
            msg = "Do not match any cilms in C18 file"
            raise ValueError(msg)

        for ls in deg_full:
            l, m = int(ls[0]), int(ls[1])
            if abs(m) > lmax:
                break
            if l > lmax:
                continue
            if m < 0:
                gia[1, l, abs(m)] = float(ls[2])
            else:
                gia[0, l, m] = float(ls[2])
    else:
        msg = f"Invalid GIA model '{model}'."
        raise ValueError(msg)

    return gia

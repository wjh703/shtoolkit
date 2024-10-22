import re
from pathlib import Path

import numpy as np


def read_gia_model(filepath: str | Path, lmax: int, model: str = "P18") -> np.ndarray:
    """read GIA model, only support ICE6G-D"""
    gia = np.zeros((2, lmax + 1, lmax + 1), dtype=np.float64)

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    if model == "ICE6G-D":
        regex_deg_0_to_2 = (
            r"GRACE Approximation for degrees 0 to 2\s+((?:\s*\d+\s+\d+\s+[+\-\d\.E]+\s+[+\-\d\.E]+\s*)+)"
        )
        regex_deg_greater_2 = (
            r"GRACE Approximation/Absolute Sea-level Values for degrees > 2\s+"
            r"((?:\s*\d+\s+\d+\s+[+\-\d\.E]+\s+[+\-\d\.E]+\s*)+)"
        )
        deg_0_to_2 = re.findall(regex_deg_0_to_2, content)
        deg_greater_2 = re.findall(regex_deg_greater_2, content)
        if deg_0_to_2 and deg_greater_2:
            deg_full = deg_0_to_2[0] + deg_greater_2[0]
        else:
            msg = "Do not match all cilms ranging from degree 0 to the maximum"
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
    else:
        msg = f"Invalid value of GIA model <{model}>."
        raise ValueError(msg)
    
    return gia

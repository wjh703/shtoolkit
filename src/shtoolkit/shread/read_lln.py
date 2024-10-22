from pathlib import Path

import numpy as np

from ..shtype import LoadLoveNumDict


def read_load_love_num(filepath: str | Path, lmax: int, frame: str = "CF") -> LoadLoveNumDict:
    h_el = np.zeros(lmax + 1)
    l_el = np.zeros(lmax + 1)
    k_el = np.zeros(lmax + 1)

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if "h_asymptotic" in line:
                break
        for line in f:
            ls = line.lower().strip().split()
            l = int(ls[0])
            if l > lmax:
                break
            h_el[l] = float(ls[1])
            n = 1 if l == 0 else l
            l_el[l] = float(ls[2]) / n
            k_el[l] = float(ls[3]) / n

    if frame == "CF":
        h_el[1] = -0.25823532
        l_el[1] = 0.12911766
        k_el[1] = 0.02607313
    lln: LoadLoveNumDict = {"h_el": h_el, "l_el": l_el, "k_el": k_el}
    return lln

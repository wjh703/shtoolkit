import re
from pathlib import Path

import numpy as np

from .. import shtime


def read_technical_note_c20_c30(
    filepath: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """read SLR degree 2/3 zonal gravitional coefficients from CSR(TN11) or GSFC(TN14)"""
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    technical_note_valid = r"TN11E|TN-14"
    tn_match = re.search(technical_note_valid, filepath.stem)
    if tn_match:
        tn_name = tn_match.group()
    else:
        msg = "Invalid technical note of C20/C30, expected TN-14 or TN11E"
        raise ValueError(msg)

    with open(filepath, "r") as f:
        content = f.read()

    data_regex = r"\d{5}\.\d\s+(\S+)\s+(\S+)\s+(?:\S+\s+)(\S+)\s+(\S+)\s+(?:\S+\s+)(\S+)\s+(?:\S+\s+)(\S+)"
    data_match = re.findall(data_regex, content)
    start, c20, c20_sigma, c30, c30_sigma, end = map(lambda x: np.array(x, dtype=float), zip(*data_match, strict=False))
    epochs = (start + end) / 2

    return epochs, c20, c20_sigma * 1e-10, c30, c30_sigma * 1e-10, tn_name


def read_technical_note_deg1(filepath: str | Path):
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    with open(filepath, "r") as f:
        content = f.read()

    data_regex = (
        r"GRCOF2\s+\d+\s+\d+\s+([+\-\d\.Ee]+)\s+[+\-\d\.Ee]+\s+([+\-\d\.Ee]+)\s+[+\-\d\.Ee]+\s+(\d{8})\.\d{4}\s+(\d{8})\.\d{4}\n"
        r"GRCOF2\s+\d+\s+\d+\s+([+\-\d\.Ee]+)\s+([+\-\d\.Ee]+)\s+([+\-\d\.Ee]+)\s+([+\-\d\.Ee]+)\s+\d{8}\.\d{4}\s+\d{8}\.\d{4}"
    )

    data_matches = re.findall(data_regex, content)
    deg1 = np.zeros((len(data_matches), 3))
    deg1_std = np.zeros((len(data_matches), 3))
    epochs = np.zeros(len(data_matches))

    for i, match in enumerate(data_matches):
        c10, c10_std, start_time, end_time = match[:4]
        c11, s11, c11_std, s11_std = match[4:]
        deg1[i] = c10, c11, s11
        deg1_std[i] = c10_std, c11_std, s11_std

        start = shtime.date_to_decimal_year(start_time)
        end = shtime.date_to_decimal_year(end_time)
        epoch = (start + end) / 2
        epochs[i] = epoch

    return epochs, deg1, deg1_std


def read_gsfc_c20_long_term(filepath: str | Path):
    with open(filepath, "r") as f:
        content = f.read()
    "1976.4454 -4.8416978090E-04      -3.7222       0.5119 -4.8416965988E-04      -2.5120       0.3656"
    """
        Column  1: Year and fraction of year of solution mid-point
        Column  2: TSVD C20
        Column  3: TSVD C20 - mean C20 (1.0E-10)
        Column  4: TSVD C20 Sigma (1.0E-10)
        Column  5: TSVD MM C20
        Column  6: TSVD MM TSVD C20 - mean C20 (1.0E-10)
        Column  7: TSVD MM TSVD C20 Sigma (1.0E-10)
        Column  8: AOD1B 28-day mean (1.0E-10)
    """

    regex = r"\s+".join([r"(-?\d+\.\d+(?:[eE][+-]?\d+)?)"] * 8)

    regex = (
        r"(\d+\.\d+)\s+(\-?\d+\.\d+E\-\d+)\s+(\-?\d+\.\d+)\s+(\-?\d+\.\d+)\s+"
        r"(\-?\d+\.\d+E\-\d+)\s+(\-?\d+\.\d+)\s+(\-?\d+\.\d+)\s+(\-?\d+\.\d+)"
    )

    epochs, c20_tsvd, c20_tsvd_delta, c20_tsvd_sigma, c20_mm, c20_mm_delta, c20_mm_sigma, aod1b = map(
        lambda x: np.array(x, dtype=float), zip(*re.findall(regex, content), strict=True)
    )

    c20_tsvd_delta *= 1e-10
    c20_tsvd_sigma *= 1e-10
    c20_mm_delta *= 1e-10
    c20_mm_sigma *= 1e-10
    aod1b *= 1e-10

    return epochs, c20_tsvd, c20_tsvd_delta, c20_tsvd_sigma, c20_mm, c20_mm_delta, c20_mm_sigma, aod1b

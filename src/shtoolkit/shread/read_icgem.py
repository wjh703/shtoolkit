import re
from pathlib import Path

import numpy as np

from .. import shtime


def read_icgem(filepath: str | Path, lmax: int | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    """read GRACE/GRACE-FO gravitional coefficients in icgem format, including GSM, GAC, GAB, GAA"""
    if isinstance(filepath, str):
        filepath = Path(filepath)

    epoch = _parse_epoch_from_filename(filepath.name)
    cilm, ecilm = _read_file(filepath, lmax)

    return epoch, cilm, ecilm


def read_non_icgem(filepath: str | Path, lmax: int | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    centers_valid = r"GRGS|CNESG"
    if isinstance(filepath, str):
        filepath = Path(filepath)

    filename = filepath.name
    center_match = re.findall(centers_valid, filename)
    if center_match:
        center = center_match[0]
    else:
        msg = "Do not match any valid center in the icgem filename."
        raise ValueError(msg)

    if center == "GRGS" or "CNESG":
        epoch_pattern = r"_(\d{4})(\d{3})-(\d{4})(\d{3})_"
        epoch_match = re.search(epoch_pattern, filename)
        if epoch_match:
            start_year, start_day, end_year, end_day = map(float, epoch_match.groups())
            if end_year > start_year:
                end_day += 365.25
            epoch = start_year + ((start_day + end_day) / 2 - 1) / 365.25 + 0.00136
        else:
            msg = f"Do not match any valid epoch in the icgem filename ({center})"
            raise ValueError(msg)

        cilm, ecilm = _read_file(filepath, lmax)

    return epoch, cilm, ecilm  # type: ignore


def read_static_gravity_field(filepath: str | Path, lmax: int | None = None):
    """read static gravity field in icgem format"""
    if isinstance(filepath, str):
        filepath = Path(filepath)
    cilm_static, _ = _read_file(filepath, lmax)
    return cilm_static


def _read_file(filepath: Path, lmax: int | None):
    encoding_to_try = ("utf-8", "iso-8859-1")
    for encoding in encoding_to_try:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                for line in f:
                    if "end_of_head" in line or "Tide convention" in line:
                        break

                    if ("max_degree" in line or "SHM" in line) and lmax is None:
                        lmax = int(re.findall(r"\d+", line)[0])

                if not isinstance(lmax, int):
                    raise TypeError(f"lmax does not specify as int object, got {type(lmax)}")

                cilm = np.zeros((2, lmax + 1, lmax + 1))
                ecilm = np.zeros((2, lmax + 1, lmax + 1))
                for line in f:
                    ls = line.lower().strip().lower().replace("d", "e").split()
                    if "gfc" not in ls:
                        continue
                    l, m = int(ls[1]), int(ls[2])
                    if m > lmax:
                        break
                    if l > lmax:
                        continue
                    cilm[:, l, m] = float(ls[3]), float(ls[4])
                    ecilm[:, l, m] = float(ls[5]), float(ls[6])
        except UnicodeDecodeError:
            continue

    return cilm, ecilm


def _parse_epoch_from_filename(filename: str):
    """Parse the epoch from the filename of an icgem file"""
    centers_valid = (
        r"UTCSR|GFZOP|JPLEM|COSTG|GRGS|AIUB|ITSG|HUST|Tongji|"
        r"IGG-SLR-HYBRID|IGG_SLRS|HLSST_SLR|Tongji-LEO|IGG-SLR-HYBRID"
    )
    center_match = re.findall(centers_valid, filename)
    if center_match:
        center = center_match[0]
    else:
        msg = "Do not match any valid center in the icgem filename."
        raise ValueError(msg)

    if center in ("UTCSR", "GFZOP", "JPLEM", "COSTG"):
        epoch_pattern = r"_(\d{4})(\d{3})-(\d{4})(\d{3})_"
        epoch_match = re.search(epoch_pattern, filename)
        if epoch_match:
            start_year, start_day, end_year, end_day = map(float, epoch_match.groups())
            if end_year > start_year:
                end_day += 365.25
            epoch = start_year + ((start_day + end_day) / 2 - 1) / 365.25 + 0.00136
        else:
            msg = f"Do not match any valid epoch in the icgem filename ({center})"
            raise ValueError(msg)
    elif center in ("AIUB", "ITSG", "Tongji", "IGG-SLR-HYBRID", "IGG_SLRS", "IGG-SLR-HYBRID", "Tongji-LEO"):
        pattern = r"(?<=_)\d{4}|(?<=-)\d{2}"
        timestamp = re.findall(pattern, filename)
        if len(timestamp) == 1:
            epoch_str = "20" + timestamp[0]
        else:
            epoch_str = "".join(timestamp)
        epoch = shtime.year_month_to_decimal_year(epoch_str)
    elif center == "HUST":
        pattern = r"\d{6}"
        timestamp = re.findall(pattern, filename)
        if timestamp:
            epoch_str = timestamp[0]
        else:
            raise ValueError("no time in file.stem")
        epoch = shtime.year_month_to_decimal_year(epoch_str)
    elif center == "HLSST_SLR":
        pattern = r"_(\d{4})_(\d{2})"
        timestamp = re.findall(pattern, filename)
        if timestamp:
            epoch_str = "".join(timestamp[0])
            epoch = shtime.year_month_to_decimal_year(epoch_str)
        else:
            msg = f"Do not match any valid epoch in the icgem filename ({center})"
            raise ValueError(msg)

    return epoch

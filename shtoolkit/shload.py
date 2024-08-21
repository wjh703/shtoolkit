import re
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd
from pyshtools.shio import SHCilmToVector, SHVectorToCilm

from . import shtime
from .shtype import LoadLoveNumDict


__all__ = [
    "read_load_love_num",
    "read_icgem",
    "read_non_icgem",
    "read_slr_5x5",
    "read_technical_note_c20_c30",
    "read_technical_note_deg1",
]


def read_load_love_num(
    filepath: str | Path, lmax: int, frame: str = "CF"
) -> LoadLoveNumDict:
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


def read_icgem(
    filepath: str | Path, lmax: int | None = None
) -> tuple[float, np.ndarray, np.ndarray]:
    """read GRACE/GRACE-FO gravitional coefficients in icgem format, including GSM, GAC, GAB, GAA"""
    centers = r"UTCSR|GFZOP|JPLEM|COSTG|GRGS|AIUB|ITSG|HUST|Tongji"
    if isinstance(filepath, str):
        filepath = Path(filepath)

    filename = filepath.name
    center = re.findall(centers, filename)[0]

    if center in ["UTCSR", "GFZOP", "JPLEM"]:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if "end_of_head" in line:
                    break

                if "time_period_of_data" in line:
                    pattern = r"\d{8}"
                    matches = re.findall(pattern, line)[:2]
                    start = shtime.date_to_decimal_year(matches[0])
                    end = shtime.date_to_decimal_year(matches[1])
                    epoch = (start + end) / 2

                if "max_degree" in line:
                    if lmax is None:
                        pattern = r"\d+"
                        lmax = int(re.findall(pattern, line)[0])

            clm, eclm = read_cilm(f, lmax)  # type: ignore

    elif center in ["AIUB", "ITSG", "Tongji"]:
        pattern = r"(?<=_)\d{4}|(?<=-)\d{2}"
        timestamp = re.findall(pattern, filename)
        if len(timestamp) == 1:
            epoch_str = "20" + timestamp[0]
        else:
            epoch_str = "".join(timestamp)
        epoch = shtime.year_month_to_decimal_year(epoch_str)

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if "end_of_head" in line:
                    break

                if "max_degree" in line:
                    if lmax is None:
                        pattern = r"\d+"
                        lmax = int(re.findall(pattern, line)[0])

            clm, eclm = read_cilm(f, lmax)  # type: ignore

    elif center == "COSTG":
        pattern = r"(\d{4})(\d{3})-(\d{4})(\d{3})"
        timestamp = re.findall(pattern, filename)[0]
        start_year = int(timestamp[0])
        start_day = float(timestamp[1])
        end_year = int(timestamp[2])
        end_day = float(timestamp[3])
        if end_year > start_year:
            end_day += 365.25
        mid = ((start_day + end_day) / 2 - 1) / 365.25
        epoch = start_year + mid

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if "end_of_head" in line:
                    break

                if "max_degree" in line:
                    if lmax is None:
                        pattern = r"\d+"
                        lmax = int(re.findall(pattern, line)[0])
            if not lmax:
                raise ValueError("lmax is None")
            clm, eclm = read_cilm(f, lmax)

    elif center == "HUST":
        pattern = r"\d{6}"
        timestamp = re.findall(pattern, filename)

        if timestamp:
            epoch_str = timestamp[0]
        else:
            raise ValueError("no time in file.stem")

        epoch = shtime.year_month_to_decimal_year(epoch_str)

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if "end_of_head" in line:
                    break

                if "max_degree" in line:
                    if lmax is None:
                        pattern = r"\d+"
                        lmax = int(re.findall(pattern, line)[0])
            if not lmax:
                raise ValueError("lmax is None")
            clm, eclm = read_cilm(f, lmax)

    else:
        raise ValueError(f"check that filename contains {centers} ")

    return epoch, clm, eclm


def read_non_icgem(
    filepath: str | Path, lmax: int | None = None
) -> tuple[float, np.ndarray, np.ndarray]:
    centers = r"GRGS|CNESG"
    if isinstance(filepath, str):
        filepath = Path(filepath)

    filename = filepath.name
    center = re.findall(centers, filename)[0]

    if center == "GRGS" or "CNESG":

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if "Tide convention" in line:
                    break

                if "SHM" in line:
                    if lmax is None:
                        pattern = r"\d+"
                        lmax = int(re.findall(pattern, line)[0])

            if not lmax:
                raise ValueError("lmax is None")
            clm = np.zeros((2, lmax + 1, lmax + 1), dtype=np.float64)
            eclm = np.zeros((2, lmax + 1, lmax + 1), dtype=np.float64)
            epoch = None

            for line in f:
                if epoch is None:
                    pattern = r"\b(\d{8})\b"
                    start, end = re.findall(pattern, line)
                    start = shtime.date_to_decimal_year(start)
                    end = shtime.date_to_decimal_year(end)
                    epoch = (start + end) / 2

                ls = line.lower().strip().split()
                l, m = int(ls[1]), int(ls[2])
                if m > lmax:
                    break
                if l > lmax:
                    continue
                clm[:, l, m] = float(ls[3]), float(ls[4])
                eclm[:, l, m] = float(ls[5]), float(ls[6])

    return epoch, clm, eclm  # type: ignore


def read_cilm(f: TextIO, lmax: int):
    clm = np.zeros((2, lmax + 1, lmax + 1), dtype=np.float64)
    eclm = np.zeros((2, lmax + 1, lmax + 1), dtype=np.float64)
    for line in f:
        ls = line.lower().strip().split()
        l, m = int(ls[1]), int(ls[2])
        if m > lmax:
            break
        if l > lmax:
            continue
        clm[:, l, m] = float(ls[3]), float(ls[4])
        eclm[:, l, m] = float(ls[5]), float(ls[6])
    return clm, eclm


def read_technical_note_c20_c30(
    filepath: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """read SLR degree 2/3 zonal gravitional coefficients from CSR(TN11) or GSFC(TN14)"""
    centers = r"CSR|GSFC"
    with open(filepath, "r") as f:
        for line in f:
            center = re.findall(centers, line)[0] + " SLR"
            break
        for line in f:
            if "Product" in line:
                break
        c20, c30 = [], []
        c20_sigma, c30_sigma = [], []
        dtime = []
        for line in f:
            ls = line.lower().strip().split()
            epoch = (float(ls[9]) + float(ls[1])) / 2
            dtime.append(epoch)
            c20.append(float(ls[2]))
            c30.append(float(ls[5]))
            c20_sigma.append(float(ls[4]) * 1e-10)
            c30_sigma.append(float(ls[7]) * 1e-10)
    return (
        np.array(dtime),
        np.array(c20),
        np.array(c20_sigma),
        np.array(c30),
        np.array(c30_sigma),
        center,
    )


def read_technical_note_deg1(
    filepath: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("end of head"):
                break
        c10, c11, s11 = [], [], []
        c10_sigma, c11_sigma, s11_sigma = [], [], []
        epochs = []
        for line in f:
            ls = line.lower().strip().split()
            l, m = int(ls[1]), int(ls[2])
            start = shtime.date_to_decimal_year(int(ls[-2].split(".")[0]))
            stop = shtime.date_to_decimal_year(int(ls[-1].split(".")[0]))
            epoch = (start + stop) / 2
            if epoch not in epochs:
                epochs.append(epoch)
            if (l, m) == (1, 0):
                c10.append(float(ls[3]))
                c10_sigma.append(float(ls[5]))
            else:
                c11.append(float(ls[3]))
                c11_sigma.append(float(ls[5]))
                s11.append(float(ls[4]))
                s11_sigma.append(float(ls[6]))
    deg1 = np.c_[c10, c11, s11]
    deg1_sigma = np.c_[c10_sigma, c11_sigma, s11_sigma]
    return np.array(epochs), deg1, deg1_sigma


def read_gia_model(filepath: str | Path, lmax: int, model: str = "P18") -> np.ndarray:
    """read GIA model, only support ICE6G-D"""
    f = open(filepath, "r", encoding="utf-8")
    gia = np.zeros((2, lmax + 1, lmax + 1), dtype=np.float64)
    if model == "ICE6G-D":
        with f:
            for line in f:
                if "GRACE" in line:
                    break

            for line in f:
                if "Absolute" in line or line.startswith("\n"):
                    break
                ls = line.lower().strip().split()
                l, m = int(ls[0]), int(ls[1])
                if m > lmax:
                    break
                if l > lmax:
                    continue
                gia[:, l, m] = float(ls[2]), float(ls[3])

            for line in f:
                if "GRACE" in line:
                    break

            for line in f:
                ls = line.lower().strip().split()
                l, m = int(ls[0]), int(ls[1])
                if m > lmax:
                    break
                if l > lmax:
                    continue
                gia[:, l, m] = float(ls[2]), float(ls[3])

    elif model == "C18":
        with f:
            for line in f:
                if "(yr^-1)" in line:
                    break

            for line in f:
                if "Absolute" in line or line.startswith("\n"):
                    break
                ls = line.lower().strip().split()
                l, m = int(ls[0]), int(ls[1])
                if abs(m) > lmax:
                    break
                if l > lmax:
                    continue
                if m < 0:
                    gia[1, l, abs(m)] = float(ls[2])
                else:
                    gia[0, l, m] = float(ls[2])

    elif model == "ICE6G-C":
        with f:
            for line in f:
                if "GRACE" in line:
                    break

            for line in f:
                if "GEOID" in line or line.startswith("\n"):
                    break
                ls = line.lower().strip().split()
                l, m = int(ls[0]), int(ls[1])
                if m > lmax:
                    break
                if l > lmax:
                    continue
                gia[:, l, m] = float(ls[2]), float(ls[3])

            for line in f:
                if "GRACE/GEOID" in line:
                    break

            for line in f:
                ls = line.lower().strip().split()
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


def read_slr_5x5(filepath: str | Path) -> tuple[np.ndarray, np.ndarray]:
    def read_gsfc_5x5(ff: TextIO):
        cilm = np.zeros((2, 7, 7))
        for _ in range(19):
            line = next(ff).strip().split()
            l, m, c, s = int(line[0]), int(line[1]), float(line[2]), float(line[3])
            cilm[0, l, m] = c
            cilm[1, l, m] = s
        return cilm

    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("Product"):
                break
        cilms = []
        epochs = []
        for line in f:
            ls = line.strip().split()
            epochs.append(float(ls[1]))
            cilms.append(read_gsfc_5x5(f))

    vector = np.array([SHCilmToVector(c) for c in np.asarray(cilms)])

    vector_df = (
        pd.DataFrame(np.hstack((np.asarray(epochs)[np.newaxis], vector)))
        .rolling(4)
        .mean()
        .dropna()
        .to_numpy()
    )
    epochs_28d = vector_df[:, 0]
    vector_28d = vector_df[:, 1:]
    cilms_28d = np.array([SHVectorToCilm(v) for v in vector_28d])

    return epochs_28d, cilms_28d

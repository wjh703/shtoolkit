from datetime import datetime

import numpy as np
import pandas as pd

__all__ = [
    "time_uniform",
    "grace_time",
    "generate_time_series",
    "generate_time_series_day",
    "year_month_to_decimal_year",
    "date_to_decimal_year",
]

GRACEMissing = [
    "200206",
    "200207",
    "200306",
    "201101",
    "201106",
    "201205",
    "201210",
    "201303",
    "201308",
    "201309",
    "201402",
    "201407",
    "201412",
    "201506",
    "201510",
    "201511",
    "201604",
    "201609",
    "201610",
    "201702",
]
GFOMissing = ["201808", "201809"]
GapBetweenGRACEandGFO = pd.period_range("201707", "201805", freq="M").strftime("%Y%m").to_list()


def time_uniform(
    inputime: np.ndarray, data: np.ndarray, trange: tuple[str, str] = ("200204", "202212")
) -> tuple[np.ndarray, np.ndarray]:

    startime, endtime = trange
    ut = generate_time_series(startime, endtime)
    outime = np.copy(ut)
    if data.ndim == 1:
        gdata = np.full(len(ut), np.nan)
    else:
        gdata = np.full((len(ut), data.shape[-1]), np.nan)
    # gdata = np.zeros(len(ut))
    for i, t in enumerate(inputime):
        residual = np.abs(ut - t)
        if any(residual < 0.050):
            argmin = np.argmin(residual)
            gdata[argmin] = data[i]
            ut[argmin] = np.inf
    # print(f"numbers of gap are {np.isnan(gdata).sum(axis=0)}")
    return outime, gdata


def grace_time(startime: str = "200204", endtime: str = "202212") -> np.ndarray:
    """
    return decimal year during GRACE/GFO data span
    """
    uniform_time = pd.period_range(startime, endtime, freq="M").strftime("%Y%m").to_list()
    diff_time = set(uniform_time) - set(GRACEMissing) - set(GFOMissing) - set(GapBetweenGRACEandGFO)
    return np.array([year_month_to_decimal_year(i) for i in sorted(diff_time)])


def generate_time_series(startime: str, endtime: str) -> np.ndarray:
    """
    return center dates in decimal year every month between startime and endtime
    """
    time_series = pd.period_range(startime, endtime, freq="M").strftime("%Y%m").values
    return np.array([year_month_to_decimal_year(ym) for ym in time_series], dtype=np.float64)


def generate_time_series_day(startime: str, endtime: str) -> np.ndarray:
    """
    return center dates in decimal year every month between startime and endtime
    """
    time_series = pd.period_range(startime, endtime, freq="D").strftime("%Y%m%d").values
    return np.array([date_to_decimal_year(ym) for ym in time_series], dtype=np.float64)


def year_month_to_decimal_year(year_month: str | int) -> float:
    if isinstance(year_month, int):
        year_month = str(year_month)
    date = datetime.strptime(year_month, "%Y%m")
    mon_num = date.month
    year = date.year
    decimal_year = year + (mon_num - 1) / 12 + 1 / 24
    return decimal_year


def date_to_decimal_year(year_month_day: str | int) -> float:
    if isinstance(year_month_day, int):
        year_month_day = str(year_month_day)
    if year_month_day[-2:] == "00":
        year_month_day = year_month_day.replace("00", "01")
    date = datetime.strptime(year_month_day, "%Y%m%d").timetuple()
    day_num = date.tm_yday
    year = date.tm_year
    decimal_year = year + (day_num - 1) / 365.25 + 1 / (365.25 * 2)
    return decimal_year

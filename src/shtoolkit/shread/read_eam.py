import datetime
import pathlib
import re

import numpy as np
import pandas as pd

from ..shtime import year_month_to_decimal_year


def read_eam(filefolder, time_interval=("19760101", "20241231"), freq="D", retfreq="ME"):
    filepaths = pathlib.Path(filefolder).iterdir()
    data = []
    for filepath in filepaths:
        stem = filepath.stem
        if "HAM" in stem:
            pattern = (
                r"(\d{4} \d{2} \d{2}) \d{2}[ \t]+\d+\.\d+[ \t]+"
                r"(-?\d+\.\d+[eE][-+]?\d+)[ \t]+"
                r"(-?\d+\.\d+[eE][-+]?\d+)[ \t]+"
                r"(-?\d+\.\d+[eE][-+]?\d+)[ \t]+"
                r"(-?\d+\.\d+[eE][-+]?\d+)[ \t]+"
                r"(-?\d+\.\d+[eE][-+]?\d+)[ \t]+"
                r"(-?\d+\.\d+[eE][-+]?\d+)\n"
            )
            epoch_format = "%Y %m %d"
        elif "SLAM" in stem:
            pattern = (
                r"(\d{4} \d{2} \d{2}) \d{2}[ \t]+\d+\.\d+[ \t]+"
                r"(-?\d+\.\d+[eE][-+]?\d+)[ \t]+"
                r"(-?\d+\.\d+[eE][-+]?\d+)[ \t]+"
                r"(-?\d+\.\d+[eE][-+]?\d+)\n"
            )
            epoch_format = "%Y %m %d"
        elif "AAM" in stem or "OAM" in stem:
            pattern = (
                r"(\d{4} \d{2} \d{2} \d{2})[ \t]+\d+\.\d+[ \t]+"
                r"(-?\d+\.\d+[eE][-+]?\d+)[ \t]+"
                r"(-?\d+\.\d+[eE][-+]?\d+)[ \t]+"
                r"(-?\d+\.\d+[eE][-+]?\d+)[ \t]+"
                r"(-?\d+\.\d+[eE][-+]?\d+)[ \t]+"
                r"(-?\d+\.\d+[eE][-+]?\d+)[ \t]+"
                r"(-?\d+\.\d+[eE][-+]?\d+)\n"
            )
            epoch_format = "%Y %m %d %H"
        else:
            raise ValueError("Invalid file name")

        with open(filepath, "r") as f:
            content = f.read()
            data.extend(re.findall(pattern, content))

    epochs, datasets = np.split(np.asarray(data), [1], axis=1)
    epochs_datetime = [datetime.datetime.strptime(epoch, epoch_format) for epoch in epochs.ravel()]

    eam_pd = pd.DataFrame(datasets.astype(float), index=epochs_datetime).resample(retfreq).mean()

    epochs = np.array([year_month_to_decimal_year(str(i.year * 100 + i.month)) for i in eam_pd.index])
    return np.hstack((epochs.reshape(-1, 1), eam_pd.to_numpy()))

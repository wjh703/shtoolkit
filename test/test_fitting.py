import timeit

import numpy as np

from shtoolkit import shtime

if __name__ == "__main__":
    deg1 = np.loadtxt(
        "D:/wjh_code/My_code/my_code_data/output/真实结果/CSR/GSM_like/buffer_100km_FM.txt",
        delimiter=",",
    )
    deg1_time = np.copy(deg1[:, 0])
    deg1 -= deg1[:100].mean(axis=0)

    a = shtime.cosine_fitting(deg1_time, deg1[:, 2])
    b = shtime.lstsq_seasonal_trend.cosine_fitting(deg1_time, deg1[:, 2])
    # print(list(map(lambda x, y : (x == y).all(), a, b)))

    callable1 = lambda: shtime.cosine_fitting(deg1_time, deg1[:, 2])
    callable2 = lambda: shtime.lstsq_seasonal_trend.cosine_fitting(deg1_time, deg1[:, 2])
    print(timeit.timeit(callable1, number=1000))
    print(timeit.timeit(callable2, number=1000))

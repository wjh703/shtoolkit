import numpy as np


def helmert(A1: np.ndarray, P1: np.ndarray, L1: np.ndarray, A2: np.ndarray, P2: np.ndarray, L2: np.ndarray):
    n1 = len(L1)
    n2 = len(L2)
    N1 = A1.T * P1 @ A1
    W1 = A1.T * P1 @ L1
    while True:
        # for _ in range(1):
        N2 = A2.T * P2 @ A2
        N = N1 + N2
        invN = np.linalg.inv(N)

        W2 = A2.T * P2 @ L2
        W = W1 + W2

        x = invN @ W
        # x = np.linalg.solve(N, W)

        V1 = A1 @ x - L1
        V2 = A2 @ x - L2

        residual1 = V1.T * P1 @ V1
        residual2 = V2.T * P2 @ V2
        R = np.array([residual1, residual2])

        s1 = n1 - 2 * np.trace(invN @ N1) + np.trace(invN @ N1) ** 2
        s2 = np.trace(invN @ N1 @ invN @ N2)
        s3 = s2
        s4 = n2 - 2 * np.trace(invN @ N2) + np.trace(invN @ N2) ** 2

        S = np.array([[s1, s2], [s3, s4]])

        theta = np.linalg.solve(S, R)
        rerr = np.abs(theta[1] - theta[0]) / (theta[0] + theta[1]) * 2
        if rerr < 0.1:
            # if np.allclose(theta[0], theta[1], 0.05):
            break
        else:
            P2 *= theta[0] / theta[1]
    return x


def vce(A1: np.ndarray, P1: np.ndarray, L1: np.ndarray, A2: np.ndarray, P2: np.ndarray, L2: np.ndarray):
    n1, n2 = len(L1), len(L2)
    
    var1, var2 = 1, 1

    while True:
        N1 = (A1.T * P1 @ A1) / var1
        W1 = (A1.T * P1 @ L1) / var1

        N2 = (A2.T * P2 @ A2) / var2
        W2 = (A2.T * P2 @ L2) / var2

        N = N1 + N2
        W = W1 + W2
        invN = np.linalg.inv(N)

        x = invN @ W

        v1 = A1 @ x - L1
        v2 = A2 @ x - L2

        r1 = np.trace(invN @ N1)
        r2 = np.trace(invN @ N2)

        var1_new = (v1.T * P1 @ v1) / (n1 - r1)
        var2_new = (v2.T * P2 @ v2) / (n2 - r2)

        var = np.array([var1, var2])
        var_new = np.array([var1_new, var2_new])

        if np.allclose(var, var_new, rtol=0.05):
            break
        else:
            var1, var2 = var1_new, var2_new
    
    return x, var


def vce3(
    A1: np.ndarray,
    P1: np.ndarray,
    L1: np.ndarray,
    A2: np.ndarray,
    P2: np.ndarray,
    L2: np.ndarray,
    A3: np.ndarray,
    P3: np.ndarray,
    L3: np.ndarray,
):
    n1, n2, n3 = len(L1), len(L2), len(L1)

    # rms1 = np.sqrt((L1**2).sum())
    # rms2 = np.sqrt((L2**2).sum())
    # var1, var2 = rms1 / rms1, rms1 / rms2
    var1, var2, var3 = 1, 1, 1
    while True:
        N1 = (A1.T * P1 @ A1) / var1
        W1 = (A1.T * P1 @ L1) / var1

        N2 = (A2.T * P2 @ A2) / var2
        W2 = (A2.T * P2 @ L2) / var2

        N3 = (A3.T * P3 @ A3) / var3
        W3 = (A3.T * P3 @ L3) / var3

        N = N1 + N2 + N3
        W = W1 + W2 + W3

        invN = np.linalg.inv(N)

        x = invN @ W

        v1 = A1 @ x - L1
        v2 = A2 @ x - L2
        v3 = A3 @ x - L3

        r1 = np.trace(invN @ N1)
        r2 = np.trace(invN @ N2)
        r3 = np.trace(invN @ N3)

        var1_new = (v1.T * P1 @ v1) / (n1 - r1)
        var2_new = (v2.T * P2 @ v2) / (n2 - r2)
        var3_new = (v3.T * P3 @ v3) / (n3 - r3)

        var = np.array([var1, var2, var3])
        var_new = np.array([var1_new, var2_new, var3_new])
        # if any(var_new < 0):
        #     breakpoint()
        if np.allclose(var, var_new, rtol=0.02):
            break
        else:
            var1, var2, var3 = var1_new, var2_new, var3_new
    return x, var

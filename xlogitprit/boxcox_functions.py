import numpy as np


def boxcox_transformation(X_matrix, lmdas):
    """returns boxcox transformed matrix

    Args:
        X_matrix: array-like
            matrix to apply boxcox transformation on
        lmdas: array-like
            lambda parameters used in boxcox transformation

    Returns:
        bxcx_X: array-like
            matrix after boxcox transformation
    """
    X_matrix[X_matrix == 0] = 1e-100  # avoids errors causes by log(0)
    if not (X_matrix > 0).all():
        raise Exception("All elements must be positive")
    bxcx_X = np.zeros_like(X_matrix)
    X_matrix = X_matrix.astype("float64")
    bxcx_X = bxcx_X.astype("float64")
    for i in range(len(lmdas)):
        if lmdas[i] == 0:
            bxcx_X[:, :, i] = np.log(X_matrix[:, :, i])
        else:
            # derivative of ((x^λ)-1)/λ
            bxcx_X[:, :, i] = (np.power(X_matrix[:, :, i], lmdas[i])-1)/lmdas[i]

    return bxcx_X


def boxcox_param_deriv(X_matrix, lmdas):
    """estimate derivate of boxcox transformation parameter (lambda)

    Args:
        X_matrix: array-like
            matrix to apply boxcox transformation on
        lmdas: array-like
            lambda parameters used in boxcox transformation

    Returns:
        der_bxcx_X: array-like
            estimated derivate of boxcox transformed matrix
    """
    X_matrix[X_matrix == 0] = 1e-100  # avoids errors causes by log(0)
    der_bxcx_X = np.zeros_like(X_matrix)
    X_matrix = X_matrix.astype("float64")
    der_bxcx_X = der_bxcx_X.astype("float64")
    for i in range(len(lmdas)):
        i -= 1
        if lmdas[i] == 0:
            # derivative of log(x)
            der_bxcx_X[:, :, i] = ((np.power(np.log(X_matrix[:, :, i])), 2))/2
        else:
            der_bxcx_X[:, :, i] = (
                (lmdas[i]*(np.power(X_matrix[:, :, i], lmdas[i])) *
                 np.log(X_matrix[:, :, i]) -
                 (np.power(X_matrix[:, :, i], lmdas[i]))+1) /
                (np.power(lmdas[i], 2)))

    return der_bxcx_X


def boxcox_transformation_mixed(X_matrix, lmdas):
    """returns boxcox transformed matrix

    Args:
        X_matrix: array-like
            matrix to apply boxcox transformation on
        lmdas: array-like
            lambda parameters used in boxcox transformation

    Returns:
        bxcx_X: array-like
            matrix after boxcox transformation
    """
    X_matrix[X_matrix == 0] = 1e-100  # avoids errors causes by log(0)
    if not (X_matrix > 0).all():
        raise Exception("All elements must be positive")
    bxcx_X = np.zeros_like(X_matrix)
    X_matrix = X_matrix.astype("float64")
    bxcx_X = bxcx_X.astype("float64")

    for i in range(len(lmdas)):
        if lmdas[i] == 0:
            bxcx_X[:, :, :, i] = np.log(X_matrix[:, :, :, i])
        else:
            bxcx_X[:, :, :, i] = np.nan_to_num(np.power(X_matrix[:, :, :, i], lmdas[i])-1) / \
                    lmdas[i]
    return bxcx_X


def boxcox_param_deriv_mixed(X_matrix, lmdas):
    """estimate derivate of boxcox transformation parameter (lambda)

    Args:
        X_matrix: array-like
            matrix to apply boxcox transformation on
        lmdas: array-like
            lambda parameters used in boxcox transformation

    Returns:
        der_bxcx_X: array-like
            estimated derivate of boxcox transformed matrix
    """
    X_matrix[X_matrix == 0] = 1e-100  # avoids errors causes by log(0)
    der_bxcx_X = np.zeros_like(X_matrix)
    X_matrix = X_matrix.astype("float64")
    der_bxcx_X = der_bxcx_X.astype("float64")
    for i in range(len(lmdas)):
        if lmdas[i] == 0:
            der_bxcx_X[:, :, :, i] = ((np.log(X_matrix[:, :, :, i])) ** 2)/2
        else:
            der_bxcx_X[:, :, :, i] = np.nan_to_num(
                (lmdas[i]*(np.power(X_matrix[:, :, :, i], lmdas[i])) *
                 np.log(X_matrix[:, :, :, i]) -
                 (np.power(X_matrix[:, :, :, i], lmdas[i]))+1) /
                (lmdas[i]**2))
    return der_bxcx_X

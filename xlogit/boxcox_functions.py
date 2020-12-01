import numpy as np


def boxcox_transformation(X, lambdas):
    """boxcox transformation of variables
    X:
    lambdas:

    returns:
    """
    boxcox_X = np.zeros_like(X)
    for i in range(len(lambdas)):
        i -= 1
        if lambdas[i] == 0:
            boxcox_X[:, :, i] = np.log(X[:, :, i])
        else:
            boxcox_X[:, :, i] = np.nan_to_num((np.power(X[:, :, i], lambdas[i]) - 1) /
                                     lambdas[i])

    return boxcox_X


def boxcox_param_deriv(X, lambdas):
    """estimate derivative of boxcox transformation parameter (lambda)
    X:
    lambdas:

    returns:
    """
    der_boxcox_X = np.zeros_like(X)
    for i in range(len(lambdas)):
        i -= 1
        if lambdas[i] == 0:
            der_boxcox_X[:, :, i] = ((np.log(X[:, :, i])) ** 2)/2  # where??
        else:
            der_boxcox_X[:, :, i] = ((lambdas[i] * np.power(X[:, :, i],
                                                            lambdas[i])) *
                                     (np.log(X[:, :, i]) -
                                     (np.power(X[:, :, i], lambdas[i])) + 1) /
                                     (lambdas[i] ** 2))

    return der_boxcox_X


def boxcox_transformation_mixed(X, lambdas):
    """boxcox transformation of variables
    X:
    lambdas:

    returns:
    """
    boxcox_X = np.zeros_like(X)
    for i in range(len(lambdas)):
        i -= 1
        if lambdas[i] == 0:
            boxcox_X[:, :, :, i] = np.log(X[:, :, :, i])
        else:
            boxcox_X[:, :, :, i] = np.nan_to_num((np.power(X[:, :, :, i],
                                                  lambdas[i]) - 1) / lambdas[i])
    return boxcox_X


def boxcox_param_deriv_mixed(X, lambdas):
    """estimate derivative of boxcox transformation parameter (lambda)
    X:
    lambdas:

    returns:
    """
    der_boxcox_X = np.zeros_like(X)
    for i in range(len(lambdas)):
        i -= 1
        if lambdas[i] == 0:
            der_boxcox_X[:, :, :, i] = ((np.log(X[:, :, :, i])) ** 2)/2  # where??
        else:
            der_boxcox_X[:, :, :, i] = ((lambdas[i] * np.power(X[:, :, :, i],
                                                              lambdas[i])) *
                                     (np.log(X[:, :, :, i]) -
                                     (np.power(X[:, :, :, i], lambdas[i])) + 1) /
                                     (lambdas[i] ** 2))
    return der_boxcox_X

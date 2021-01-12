import numpy as np


def boxcox_transformation(X, lambdas):
    """boxcox transformation of variables
    X:
    lambdas:

    returns:
    """
    #TODO: CHECK
    boxcox_X = np.zeros_like(X)
    for i in range(len(lambdas)):
        # i -= 1
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
    #TODO: CHECK
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


def boxcox_transformation_mixed(X_matrix, lmdas):
    """boxcox transformation of variables
    X_matrix:
    lmdas:

    returns:
    """
    bxcx_X = np.zeros_like(X_matrix)
    for i in range(len(lmdas)):
        if lmdas[i] == 0:
            bxcx_X[:,:,:,i] = np.log(X_matrix[:,:,:,i])
        else:
            bxcx_X[:,:,:,i] = np.nan_to_num((np.power(X_matrix[:,:,:,i],lmdas[i])-1)/lmdas[i])
    return bxcx_X


def boxcox_param_deriv_mixed(X_matrix, lmdas):
    """estimate derivative of boxcox transformation parameter (lambda)
    X:
    lambdas:

    returns:
    """
    der_bxcx_X = np.zeros_like(X_matrix)
    for i in range(len(lmdas)):
        i -= 1
        if lmdas[i] == 0:
            der_bxcx_X[:, :, :, i] = ((np.log(X_matrix[:, :, :, i])) ** 2)/2
        else:
            der_bxcx_X[:,:,:,i] = ((lmdas[i]*(np.power(X_matrix[:,:,:,i],lmdas[i]))
            *np.log(X_matrix[:,:,:,i])-(np.power(X_matrix[:,:,:,i],lmdas[i]))+1)
            /(lmdas[i]**2))
    return der_bxcx_X

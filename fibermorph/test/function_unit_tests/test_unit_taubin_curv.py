import numpy as np
import warnings

def taubin_curv(coords, resolution):
    """
    Algebraic circle fit by Taubin
      G. Taubin, "Estimation Of Planar Curves, Surfaces And Nonplanar
                  Space Curves Defined By Implicit Equations, With
                  Applications To Edge And Range Image Segmentation",
      IEEE Trans. PAMI, Vol. 13, pages 1115-1138, (1991)

    :param XYcoords:    list [[x_1, y_1], [x_2, y_2], ....]
    :return:            a, b, r.  a and b are the center of the fitting circle, and r is the curv

    Parameters
    ----------
    resolution

    """
    warnings.filterwarnings("ignore")  # suppress RuntimeWarnings from dividing by zero
    XY = np.array(coords)
    X = XY[:, 0] - np.mean(XY[:, 0])  # norming points by x avg
    Y = XY[:, 1] - np.mean(XY[:, 1])  # norming points by y avg
    centroid = [np.mean(XY[:, 0]), np.mean(XY[:, 1])]
    Z = X * X + Y * Y
    Zmean = np.mean(Z)
    Z0 = ((Z - Zmean) / (2. * np.sqrt(Zmean)))  # changed from using old_div to Python 3 native division
    ZXY = np.array([Z0, X, Y]).T
    U, S, V = np.linalg.svd(ZXY, full_matrices=False)  #
    V = V.transpose()
    A = V[:, 2]
    A[0] = (A[0]) / (2. * np.sqrt(Zmean))
    A = np.concatenate([A, [(-1. * Zmean * A[0])]], axis=0)
    # a, b = (-1 * A[1:3]) / A[0] / 2 + centroid
    r = np.sqrt(A[1] * A[1] + A[2] * A[2] - 4 * A[0] * A[3]) / abs(A[0]) / 2

    if np.isfinite(r):
        curv = 1 / (r / resolution)
        return curv
    elif np.isfinite(r) or np.isnan(r):
        return 0

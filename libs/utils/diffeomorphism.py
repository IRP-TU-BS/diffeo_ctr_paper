import numpy as np

# from frechetdist import frdist
from scipy.optimize import leastsq
from similaritymeasures import similaritymeasures


def kernel(x, y, p, mu=0.1):
    return np.exp(-((mu * p) ** 2.0) * np.linalg.norm(x - y, axis=1) ** 2.0)


def Gauss(p, x, c, v):
    return x + np.asarray([kernel(x, c, p)]).T * v


def iGauss(p, y, c, v):
    return y - np.asarray([kernel(y, c, p)]).T * v


def pmax(v):
    return 1 / (np.sqrt(2) * np.linalg.norm(v)) * np.exp(1 / 2)


def sklearn_loss(x, c, v, target):
    def loss(p):
        # return np.sqrt(np.sum(np.square(np.linalg.norm(Gauss(p, x, c, v) - target, axis=1)), axis=0)) # + 0.6*(np.linalg.norm(Gauss(p, x, c, v)[0,:]))
        return 0.6 * similaritymeasures.frechet_dist(
            Gauss(p, x, c, v), target
        ) + 0.4 * similaritymeasures.pcm(Gauss(p, x, c, v), target)

    return loss


def sklearn_optimize(p, x, c, v, target, mu=0.9, print_loss=True):
    ps = p  # np.max([0,mu*p])
    loss = sklearn_loss(x, c, v, target)
    ret = leastsq(loss, ps, ftol=0.0001)
    return ret[0]


def inv_diffeo_map(X, param, ma, v):
    Z = np.asarray(X)
    for i in range(len(ma) - 1, -1, -1):
        c = Z[ma[i], :] * np.ones(Z.shape)
        v_tmp = v[i] * np.ones(Z.shape)
        Z = iGauss(param[i], Z, c, v_tmp)
    return Z


def diffeo_map(X, Y, K=10, beta=0.5):
    Z = X
    pj = []
    ms = []
    v = []
    parameters = []

    Y = np.asarray(Y)
    Z = np.asarray(Z)
    Zs = []

    for j in range(K):
        m = np.argmax(
            np.linalg.norm(Z - Y, axis=1), axis=0
        )  # [0]  # TODO using [0] in the hope that both values are always the same
        ms.append(m)
        pj.append(Z[m, :])
        q = Y[m, :]
        v.append(beta * (q - pj[j]))
        c = pj[j] * np.ones(Z.shape)
        v_tmp = v[j] * np.ones(Z.shape)
        c = np.asarray(c)
        v_tmp = np.asarray(v_tmp)
        Z_tmp = Z
        param = np.asarray([1.0])
        param = sklearn_optimize(param, Z, c, v_tmp, Y)
        parameters.append(param)

        Z = Gauss(param, Z_tmp, c, v_tmp)
        Zs.append(Z)

    return Z, parameters, pj, ms, v, Zs

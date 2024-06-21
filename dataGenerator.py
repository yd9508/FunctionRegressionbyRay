from fda import *
import numpy as np
import pickle

def dataGenerator():
    print('Simulating Data...')
    # number of basis functions in basis system
    t = 20
    rangeval = [0, 100]
    basisobj = create_bspline_basis(rangeval, t)
    # number of observations pre local server n
    n = 100
    # number of features p 20
    p = 20
    betaPar = fdPar(basisobj, 0, 0)
    # number of observations pre local server n
    n = 100
    # number of features p 20
    p = 20
    predictors = np.zeros([101, p, n])
    weight = np.zeros([t, p])
    cmat = np.zeros([t, p])
    numworkersseq = [2, 4, 6, 8, 10]

    for numworker in numworkersseq:
        for l in range(1, numworker + 1):
            Bmat = np.ndarray([t, t], buffer=np.random.normal(10, 1, t * t))
            for i in range(p):
                cmat[:, i] = np.ndarray([t], buffer=np.random.normal(1, 1e-5, t)) + np.ndarray([t],
                                                                                               buffer=np.random.lognormal(
                                                                                                   i * 0.2, 5e-1, t))

            weight = cmat

            for i in range(n):
                bbspl2 = bifd(np.linspace(1, pow(t, 2), pow(t, 2)).reshape((t, t)), create_bspline_basis(rangeval, t),
                              create_bspline_basis(rangeval, t))
                bifdbasis = bifdPar(bbspl2, 0, 0, 0, 0)
                betaList = [betaPar, bifdbasis]

                for j in range(p):
                    temp = fd(weight[:, j], basisobj)
                    y = eval_fd([i for i in range(rangeval[0], rangeval[1] + 1)], temp)
                    y = np.transpose(y)
                    X = smooth_basis([i for i in range(rangeval[0], rangeval[1] + 1)], y, basisobj).fd
                    predictors[:, j, i] = eval_fd([i for i in range(rangeval[0], rangeval[1] + 1)], X).reshape(
                        rangeval[1] + 1)

            yfdobj = fd(np.zeros([t, n]), basisobj)
            beta0estfd = fd(np.ndarray([t], buffer=np.random.normal(0, 0.1, t)), betaList[0].fd.basisobj)
            beta1estbifd = bifd(Bmat, betaList[1].bifd.sbasis, betaList[1].bifd.tbasis)
            lin = linmodList(beta0estfd=beta0estfd, beta1estbifd=beta1estbifd, yhatfdobj=0)

            predictorLst = []
            for i in range(p):
                temp = smooth_basis([i for i in range(rangeval[0], rangeval[1] + 1)], predictors[:, i, :], basisobj).fd
                predictorLst.append(temp)
                if i in range(5):
                    aa = predit_linmod(lin, predictorLst[i])
                    yfdobj = yfdobj + predit_linmod(lin, predictorLst[i])

            yfdobj = yfdobj + fd(np.ndarray([t, n], buffer=np.random.normal(0, 0.1, t * n)), basisobj)

            with open('tmp/yfdobj_' + str(l) + '_' + str(numworker), 'wb') as file:
                pickle.dump(yfdobj, file)
            with open('tmp/predictorLst_' + str(l) + '_' + str(numworker), 'wb') as file:
                pickle.dump(predictorLst, file)
    print('Data generated.')
    return 0

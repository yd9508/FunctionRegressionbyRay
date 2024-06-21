import numpy as np
from fda import *
import pickle


def federatedFunctionalGradBoostLSA(x, y, betaList, boost_control, step_length):
    numworkers = len(y)
    init = y[0].mean()
    for i in range(1, numworkers):
        init.coef = np.hstack((init.coef, y[i].mean().coef))
    init = init.mean()
    init = init.mean()
    residual = []
    numSamplesEachWorker = np.zeros([numworkers])
    for i in range(numworkers):
        res = []
        m = y[i].coef.shape[1]
        numSamplesEachWorker[i] = m
        initi = init
        initi.coef = np.ndarray([y[i].coef.shape[0], m], buffer=np.repeat(init.coef, m))
        residual.append(y[i] - initi)
        res.append(initi)
        with open('tmp/tempres' + str(i) + '_' + str(m), 'wb') as file:
            pickle.dump(res, file)

    numPredictors = len(x[0])
    j = 2
    while j <= boost_control:
        tempLst = []
        # compute gradient, fit base learner, estimate local B
        for i in range(numworkers):
            lst = []
            for k in range(numPredictors):
                lin = linmod(x[i][k], residual[i], betaList)
                lst.append(lin)
            tempLst.append(lst)

        coefestimate = []

        # global parameters B estimate
        for k in range(numPredictors):
            coefTemp = tempLst[0][k].coefvec
            for i in range(1, numworkers):
                coefTemp = coefTemp + tempLst[i][k].coefvec
            coefestimate.append(coefTemp / numworkers)

        # local y hat for RSS
        for i in range(numworkers):
            for k in range(numPredictors):
                tempLst[i][k].coefvec = coefestimate[k]
                alphacoefsdim = tempLst[i][k].beta0estfd.coef.shape
                betacoefdim = tempLst[i][k].beta1estbifd.coef.shape
                tempLst[i][k].beta0estfd.coef = tempLst[i][k].coefvec[:alphacoefsdim[0]]
                tempLst[i][k].beta1estbifd.coef = np.transpose(
                    tempLst[i][k].coefvec[alphacoefsdim[0]:].reshape(betacoefdim))
                yhatfdobj = predit_linmod(tempLst[i][k], x[i][k])
                tempLst[i][k].yhatfdobj = yhatfdobj

        sse = np.zeros([numPredictors])
        for i in range(numworkers):
            for k in range(numPredictors):
                sse[k] = sse[k] + inprod(
                    ((tempLst[i][k].yhatfdobj - residual[i]) * (tempLst[i][k].yhatfdobj - residual[i]))).sum()

        best = np.argmin(sse)
        print(f'LSA Round {j} : {best} selected.')

        residual = []
        for i in range(numworkers):
            with open('tmp/tempres' + str(i) + '_' + str(m), 'rb') as file:
                res = pickle.load(file)
            res.append(tempLst[i][best])
            with open('tmp/tempres' + str(i) + '_' + str(m), 'wb') as file:
                pickle.dump(res, file)
            residual.append(y[i] - pred_gradboost1(res, step_length))

        j = j + 1

    return (res)


def federatedFunctionalGradBoostAvg(x, y, betaList, boost_control, step_length):
    numworkers = len(y)
    numworkers = len(y)
    init = y[0].mean()
    for i in range(1, numworkers):
        init.coef = np.hstack((init.coef, y[i].mean().coef))
    init = init.mean()
    init = init.mean()
    residual = []
    numSamplesEachWorker = np.zeros([numworkers])
    for i in range(numworkers):
        res = []
        m = y[i].coef.shape[1]
        numSamplesEachWorker[i] = m
        initi = init
        initi.coef = np.ndarray([y[i].coef.shape[0], m], buffer=np.repeat(init.coef, m))
        residual.append(y[i] - initi)
        res.append(initi)
        with open('tmp/tempres' + str(i) + '_' + str(m), 'wb') as file:
            pickle.dump(res, file)

    numPredictors = len(x[0])
    j = 2
    while j <= boost_control:
        tempLst = []

        coefMat = np.ones([420, numPredictors])
        Step1 = 1
        while Step1 <= 20:
            for k in range(numPredictors):
                currCoefVec = np.ndarray([420, numworkers], buffer=np.repeat(coefMat[:, k], numworkers))
                for i in range(numworkers):
                    # N2
                    lin = linmod(x[i][k], residual[i], betaList)
                    Step2 = 1
                    while Step2 < 20:
                        r = lin.Dmat - np.ndarray([420, 1], buffer=np.matmul(lin.Cmat, currCoefVec[:, i]))
                        gamma = np.matmul(np.transpose(r), r) / np.matmul(np.matmul(np.transpose(r), lin.Cmat), r)
                        # local step length eta
                        length = gamma * r
                        # B_n2+1 for number p feature
                        currCoefVec[:, i] = currCoefVec[:, i] + length[:, 0]
                        Step2 = Step2 + 1
                # aggregate local B_n2 to B
                coefMat[:, k] = np.mean(currCoefVec, 1)
                Step1 = Step1 + 1

        tempLst = []
        for i in range(numworkers):
            lst = []
            for k in range(numPredictors):
                lin = linmod(x[i][k], residual[i], betaList)
                lst.append(lin)
            tempLst.append(lst)

        coefestimate = []

        for k in range(numPredictors):
            coefestimate.append(coefMat[:, k])

        for i in range(numworkers):
            for k in range(numPredictors):
                tempLst[i][k].coefvec = coefestimate[k]
                alphacoefsdim = tempLst[i][k].beta0estfd.coef.shape
                betacoefdim = tempLst[i][k].beta1estbifd.coef.shape
                tempLst[i][k].beta0estfd.coef = tempLst[i][k].coefvec[:alphacoefsdim[0]]
                tempLst[i][k].beta1estbifd.coef = np.transpose(
                    tempLst[i][k].coefvec[alphacoefsdim[0]:].reshape(betacoefdim))
                yhatfdobj = predit_linmod(tempLst[i][k], x[i][k])
                tempLst[i][k].yhatfdobj = yhatfdobj

        sse = np.zeros([numPredictors])
        for i in range(numworkers):
            for k in range(numPredictors):
                sse[k] = sse[k] + inprod(
                    ((tempLst[i][k].yhatfdobj - residual[i]) * (tempLst[i][k].yhatfdobj - residual[i]))).sum()

        best = np.argmin(sse)
        print(f'AVG Round {j} : {best} selected.')

        residual = []
        for i in range(numworkers):
            with open('tmp/tempres' + str(i) + '_' + str(m), 'rb') as file:
                res = pickle.load(file)
            res.append(tempLst[i][best])
            with open('tmp/tempres' + str(i) + '_' + str(m), 'wb') as file:
                pickle.dump(res, file)
            residual.append(y[i] - pred_gradboost1(res, step_length))

        j = j + 1

    return (res)

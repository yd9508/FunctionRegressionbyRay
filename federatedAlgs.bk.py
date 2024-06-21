import numpy as np
from fda import *
import pickle
import time
import ray
import sys
ray.init()


@ray.remote
def linLstGenerate(k, i_xk, resid, betaL, in_time):
    trans_time = time.time() - in_time
    # time1=time.time() 1705521784.236855
    lin = linmod(i_xk, resid, betaL)
    # time2=time.time() 1705521812.2464714
    trans_time -= time.time()
    return [k, lin, trans_time, sys.getsizeof(k) + sys.getsizeof(i_xk) + sys.getsizeof(resid) + sys.getsizeof(betaL) + sys.getsizeof(in_time)]


@ray.remote
def tempLstGenerate(i, i_x, resid, betaL, numPredictors, in_time):
    trans_time = time.time()-in_time
    # 1705521783.6395488
    trans_data = sys.getsizeof(i) + sys.getsizeof(i_x) + sys.getsizeof(resid) + sys.getsizeof(betaL) + sys.getsizeof(numPredictors)
    cur_time = time.time()
    linworks = [linLstGenerate.remote(k, i_x[k], resid, betaL, cur_time) for k in range(numPredictors)]
    linIdx = ray.get(linworks)
    cur_time = time.time()
    sub_trans_time = np.mean([item[2] + cur_time for item in linIdx])
    trans_data += sum([item[3] for item in linIdx]) + sys.getsizeof(linIdx)
    linIdx.sort()
    lst = [item[1] for item in linIdx]
    # print([item[2:] for item in linIdx])
    # print("5", time.time(), i)
    # 1705521855.63953
    trans_time -= time.time()
    return [i, lst, trans_time, trans_data, sub_trans_time]


def federatedFunctionalGradBoostLSA(x, y, betaList, boost_control, step_length):
    trans_time_list = []
    sub_trans_time_list = []
    trans_data_total = 0
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
        # print("1", time.time())
        # 1705387323.2915132 1705388451.6095245
        # 1705521783.2146869
        cur_time = time.time()
        remote_works = [tempLstGenerate.remote(i, x[i], residual[i], betaList, numPredictors, cur_time) for i in range(numworkers)]
        tempLstIdx = ray.get(remote_works)
        cur_time = time.time()
        trans_time_list.append(np.mean([item[2] + cur_time for item in tempLstIdx]))
        sub_trans_time_list.append(np.mean([item[4] for item in tempLstIdx]))
        trans_data_total += sum([item[3] for item in tempLstIdx]) + sys.getsizeof(tempLstIdx)

        # print("6", time.time())
        # 1705521855.6742206
        # exit(0)
        tempLstIdx.sort()
        tempLst = [item[1] for item in tempLstIdx]
        coefestimate = []
        # print("2", time.time())
        # exit(0)
        # 1705387370.324656 1705388456.0331485
        for k in range(numPredictors):
            coefTemp = tempLst[0][k].coefvec
            for i in range(1, numworkers):
                coefTemp = coefTemp + tempLst[i][k].coefvec
            coefestimate.append(coefTemp / numworkers)

        # print("3", time.time()) 1705387370.324656
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

        # print("4", time.time()) 1705387373.236033
        sse = np.zeros([numPredictors])
        for i in range(numworkers):
            for k in range(numPredictors):
                sse[k] = sse[k] + inprod(
                    ((tempLst[i][k].yhatfdobj - residual[i]) * (tempLst[i][k].yhatfdobj - residual[i]))).sum()

        best = np.argmin(sse)
        print(f'LSA Round {j} : {best} selected.')

        # print("5", time.time()) 1705387376.2423809
        residual = []
        for i in range(numworkers):
            with open('tmp/tempres' + str(i) + '_' + str(m), 'rb') as file:
                res = pickle.load(file)
            res.append(tempLst[i][best])
            with open('tmp/tempres' + str(i) + '_' + str(m), 'wb') as file:
                pickle.dump(res, file)
            residual.append(y[i] - pred_gradboost1(res, step_length))

        # print("6", time.time()) 1705387376.2455325
        j = j + 1
        # exit(0)
    return (res, np.mean(trans_time_list), trans_data_total, np.mean(sub_trans_time_list))


def federatedFunctionalGradBoostAvg(x, y, betaList, boost_control, step_length):
    trans_time_list = []
    sub_trans_time_list = []
    trans_data_total = 0
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
                    lin = linmod(x[i][k], residual[i], betaList)
                    Step2 = 1
                    while Step2 < 20:
                        r = lin.Dmat - np.ndarray([420, 1], buffer=np.matmul(lin.Cmat, currCoefVec[:, i]))
                        gamma = np.matmul(np.transpose(r), r) / np.matmul(np.matmul(np.transpose(r), lin.Cmat), r)
                        length = gamma * r
                        currCoefVec[:, i] = currCoefVec[:, i] + length[:, 0]
                        Step2 = Step2 + 1
                coefMat[:, k] = np.mean(currCoefVec, 1)
                Step1 = Step1 + 1

        # tempLst = []
        # for i in range(numworkers):
        #     lst = []
        #     for k in range(numPredictors):
        #         lin = linmod(x[i][k], residual[i], betaList)
        #         lst.append(lin)
        #     tempLst.append(lst)
        cur_time = time.time()
        remote_works = [tempLstGenerate.remote(i, x[i], residual[i], betaList, numPredictors, cur_time) for i in range(numworkers)]
        tempLstIdx = ray.get(remote_works)
        cur_time = time.time()
        trans_time_list.append(np.mean([item[2] + cur_time for item in tempLstIdx]))
        sub_trans_time_list.append(np.mean([item[4] for item in tempLstIdx]))
        trans_data_total += sum([item[3] for item in tempLstIdx]) + sys.getsizeof(tempLstIdx)

        tempLstIdx.sort()
        tempLst = [item[1] for item in tempLstIdx]
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

    return (res, np.mean(trans_time_list), trans_data_total, np.mean(sub_trans_time_list))

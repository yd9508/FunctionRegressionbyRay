import numpy as np
from fda import *
import pickle
import time
import ray
import sys
ray.init()


@ray.remote
def coefVecEstimate(i, x_ik, residual_i, betaL, currCoefVec_i, in_time):
    trans_time = time.time() - in_time
    trans_data = (sys.getsizeof(i) + sys.getsizeof(x_ik) +
                  sys.getsizeof(residual_i) + sys.getsizeof(betaL) +
                  sys.getsizeof(currCoefVec_i) + sys.getsizeof(in_time))
    lin = linmod(x_ik, residual_i, betaL)
    Step2 = 1
    while Step2 < 20:
        r = lin.Dmat - np.ndarray([420, 1], buffer=np.matmul(lin.Cmat, currCoefVec_i))
        gamma = np.matmul(np.transpose(r), r) / np.matmul(np.matmul(np.transpose(r), lin.Cmat), r)
        length = gamma * r
        currCoefVec_i = currCoefVec_i + length[:, 0]
        Step2 = Step2 + 1
    trans_time -= time.time()
    return [i, currCoefVec_i, trans_time, trans_data]
    
@ray.remote
def tempLstGenerate(i, x_i, residual_i, betaL, numPredictors, in_time):
    trans_time = time.time()-in_time
    # 1705521783.6395488
    trans_data = (sys.getsizeof(i) + sys.getsizeof(x_i) +
                  sys.getsizeof(residual_i) + sys.getsizeof(betaL) +
                  sys.getsizeof(numPredictors) + sys.getsizeof(in_time))
    lst = []
    # print("2.1", time.time())
    for k in range(numPredictors):
        lin = linmod(x_i[k], residual_i, betaL)
        lst.append(lin)
    # print("2.2", time.time())
    # print([item[2:] for item in linIdx])
    # print("5", time.time(), i)
    # 1705521855.63953
    trans_time -= time.time()
    return [i, lst, trans_time, trans_data]

@ray.remote
def yhatGenerate(i, tempLst_i, numPredictors, coefestimate, x_i, in_time):
    trans_time = time.time()-in_time
    trans_data = (sys.getsizeof(i) + sys.getsizeof(tempLst_i) +
                  sys.getsizeof(numPredictors) + sys.getsizeof(coefestimate) +
                  sys.getsizeof(x_i) + sys.getsizeof(in_time))

    yhat_list = []
    for k in range(numPredictors):
        tempLst_i[k].coefvec = coefestimate[k]
        alphacoefsdim = tempLst_i[k].beta0estfd.coef.shape
        betacoefdim = tempLst_i[k].beta1estbifd.coef.shape
        tempLst_i[k].beta0estfd.coef = tempLst_i[k].coefvec[:alphacoefsdim[0]]
        tempLst_i[k].beta1estbifd.coef = np.transpose(
            tempLst_i[k].coefvec[alphacoefsdim[0]:].reshape(betacoefdim))
        yhatfdobj = predit_linmod(tempLst_i[k], x_i[k])
        yhat_list.append(yhatfdobj)
    trans_time -= time.time()
    return [i, yhat_list, trans_time, trans_data]

@ray.remote
def sseGenerate(i, tempLst_i, numPredictors, residual_i, in_time):
    trans_time = time.time()-in_time
    trans_data = (sys.getsizeof(i) + sys.getsizeof(tempLst_i) +
                  sys.getsizeof(numPredictors) + sys.getsizeof(residual_i) +
                  sys.getsizeof(in_time))
    sse = np.zeros([numPredictors])
    for k in range(numPredictors):
        sse[k] = inprod(
            ((tempLst_i[k].yhatfdobj - residual_i) * (tempLst_i[k].yhatfdobj - residual_i))).sum()
    trans_time -= time.time()
    return [i, sse, trans_time, trans_data]


def federatedFunctionalGradBoostLSA(x, y, betaList, boost_control, step_length):
    trans_time_list = []
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
        # 1 1705589866.4481773
        cur_time = time.time()
        remote_works = [tempLstGenerate.remote(i, x[i], residual[i], betaList, numPredictors, cur_time) for i in range(numworkers)]
        tempLstIdx = ray.get(remote_works)
        cur_time = time.time()
        trans_time_list.append(np.mean([item[2] + cur_time for item in tempLstIdx]))
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
        # 2 1705589991.0933747
        for k in range(numPredictors):
            coefTemp = tempLst[0][k].coefvec
            for i in range(1, numworkers):
                coefTemp = coefTemp + tempLst[i][k].coefvec
            coefestimate.append(coefTemp / numworkers)

        # print("3", time.time()) 1705387370.324656
        cur_time = time.time()
        remote_works = [
            yhatGenerate.remote(i, tempLst[i], numPredictors, coefestimate, x[i], cur_time) for i in
                        range(numworkers)]
        yhatIdx = ray.get(remote_works)
        cur_time = time.time()
        trans_time_list[-1] += np.mean([item[2] + cur_time for item in yhatIdx])
        trans_data_total += sum([item[3] for item in yhatIdx]) + sys.getsizeof(yhatIdx)
        yhatIdx.sort()
        for i in range(numworkers):
            for k in range(numPredictors):
                tempLst[i][k].yhatfdobj = yhatIdx[i][1][k]

        # print("4", time.time()) 1705387373.236033
        sse = np.zeros([numPredictors])
        cur_time = time.time()
        remote_works = [
            sseGenerate.remote(i, tempLst[i], numPredictors, residual[i], cur_time) for i in
                        range(numworkers)]
        sseIdx = ray.get(remote_works)
        cur_time = time.time()
        trans_time_list[-1] += np.mean([item[2] + cur_time for item in sseIdx])
        trans_data_total += sum([item[3] for item in sseIdx]) + sys.getsizeof(sseIdx)
        sseIdx.sort()
        for i in range(numworkers):
            for k in range(numPredictors):
                sse[k] = sse[k] + sseIdx[i][1][k]

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
    return (res, np.mean(trans_time_list), trans_data_total)


def federatedFunctionalGradBoostAvg(x, y, betaList, boost_control, step_length):
    trans_time_list = []
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
        # print("1", time.time())
        # 1705588959.9011655
        trans_time_list_sub = []
        while Step1 <= 20:
            for k in range(numPredictors):
                currCoefVec = np.ndarray([420, numworkers], buffer=np.repeat(coefMat[:, k], numworkers))
                # for i in range(numworkers)
                    # lin = linmod(x[i][k], residual[i], betaList)
                    # Step2 = 1
                    # while Step2 < 20:
                    #     r = lin.Dmat - np.ndarray([420, 1], buffer=np.matmul(lin.Cmat, currCoefVec[:, i]))
                    #     gamma = np.matmul(np.transpose(r), r) / np.matmul(np.matmul(np.transpose(r), lin.Cmat), r)
                    #     length = gamma * r
                    #     currCoefVec[:, i] = currCoefVec[:, i] + length[:, 0]
                    #     Step2 = Step2 + 1
                cur_time = time.time()
                remote_works = [coefVecEstimate.remote(i, x[i][k], residual[i], betaList, currCoefVec[:, i], cur_time) for
                                i in range(numworkers)]
                coefVecEstimateIdx = ray.get(remote_works)
                cur_time = time.time()
                trans_time_list_sub.append(np.mean([item[2] + cur_time for item in coefVecEstimateIdx]))
                trans_data_total += sum([item[3] for item in coefVecEstimateIdx]) + sys.getsizeof(coefVecEstimateIdx)
                coefVecEstimateIdx.sort()
                for i in range(numworkers):
                    currCoefVec[:, i] = coefVecEstimateIdx[i][1]
                coefMat[:, k] = np.mean(currCoefVec, 1)
                Step1 = Step1 + 1
        trans_time_list.append(np.mean(trans_time_list_sub))
        # print("2", time.time())
        # 1705588974.5712345
        # 1705589279.6038835
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
        trans_time_list[-1] += np.mean([item[2] + cur_time for item in tempLstIdx])
        trans_data_total += sum([item[3] for item in tempLstIdx]) + sys.getsizeof(tempLstIdx)
        tempLstIdx.sort()
        tempLst = [item[1] for item in tempLstIdx]
        # print("3", time.time())
        # 1705589091.5911021
        # 1705589291.6858494
        coefestimate = []
        # print("4", time.time()) 1705589091.5911021
        for k in range(numPredictors):
            coefestimate.append(coefMat[:, k])

        cur_time = time.time()
        remote_works = [
            yhatGenerate.remote(i, tempLst[i], numPredictors, coefestimate, x[i], cur_time) for i in
                        range(numworkers)]
        yhatIdx = ray.get(remote_works)
        cur_time = time.time()
        trans_time_list[-1] += np.mean([item[2] + cur_time for item in yhatIdx])
        trans_data_total += sum([item[3] for item in yhatIdx]) + sys.getsizeof(yhatIdx)
        yhatIdx.sort()
        for i in range(numworkers):
            for k in range(numPredictors):
                tempLst[i][k].yhatfdobj = yhatIdx[i][1][k]
        # print("5", time.time()) 1705589093.4396863
        sse = np.zeros([numPredictors])
        cur_time = time.time()
        remote_works = [
            sseGenerate.remote(i, tempLst[i], numPredictors, residual[i], cur_time) for i in
                        range(numworkers)]
        sseIdx = ray.get(remote_works)
        cur_time = time.time()
        trans_time_list[-1] += np.mean([item[2] + cur_time for item in sseIdx])
        trans_data_total += sum([item[3] for item in sseIdx]) + sys.getsizeof(sseIdx)
        sseIdx.sort()
        for i in range(numworkers):
            for k in range(numPredictors):
                sse[k] = sse[k] + sseIdx[i][1][k]
        # print("6", time.time()) 1705589094.8537838
        best = np.argmin(sse)
        print(f'AVG Round {j} : {best} selected.')
        # exit(0)
        residual = []
        for i in range(numworkers):
            with open('tmp/tempres' + str(i) + '_' + str(m), 'rb') as file:
                res = pickle.load(file)
            res.append(tempLst[i][best])
            with open('tmp/tempres' + str(i) + '_' + str(m), 'wb') as file:
                pickle.dump(res, file)
            residual.append(y[i] - pred_gradboost1(res, step_length))

        j = j + 1

    return (res, np.mean(trans_time_list), trans_data_total)

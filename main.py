import numpy as np
from fda import *
import pickle
from federatedAlgs import *
from dataGenerator import *
import time
def main():
    dataGenerator()
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
    samplesPerWorker = 100
    # number of features p 20
    numPredictors = 20
    yfine = np.linspace(1, 100, 100)
    allsamples = set([i for i in range(100)])
    numworkersseq = [2, 4, 6, 8, 10]
    bbspl2 = bifd(np.linspace(1, pow(t, 2), pow(t, 2)).reshape((t, t)), create_bspline_basis(rangeval, t),
                  create_bspline_basis(rangeval, t))
    bifdbasis = bifdPar(bbspl2, 0, 0, 0, 0)
    betaList = [betaPar, bifdbasis]
    time1 = np.zeros([len(numworkersseq), 4])
    time2 = np.zeros([len(numworkersseq), 4])

    for numworkers in numworkersseq:
        # 4-fold CV
        for hh in range(4):
            print(f'Number of local servers : {numworkers}; {hh} fold.')
            test = set([i for i in range(hh * 25, (hh + 1) * 25)])
            train = list(allsamples - test)
            test = setGenerator(test, samplesPerWorker, numworkers)
            predictors = np.zeros([len(yfine), samplesPerWorker * numworkers, numPredictors])
            response = np.zeros([len(yfine), samplesPerWorker * numworkers])
            x = []
            y = []
            for l in range(1, numworkers + 1):
                with open('tmp/yfdobj_' + str(l) + '_' + str(numworkers), 'rb') as file:
                    yfdobj = pickle.load(file)
                with open('tmp/predictorLst_' + str(l) + '_' + str(numworkers), 'rb') as file:
                    predictorLst = pickle.load(file)
                xfdobjTrainLst = []
                for i in range(numPredictors):
                    predictors[:, (samplesPerWorker * (l - 1)): (samplesPerWorker * l), i] = eval_fd(yfine,
                                                                                                     predictorLst[i])
                    temp1 = smooth_basis(yfine, predictors[:, [i + samplesPerWorker * (l - 1) - 1 for i in train], i],
                                         basisobj).fd
                    xfdobjTrainLst.append(temp1)
                x.append(xfdobjTrainLst)
                response[:, (samplesPerWorker * (l - 1)): (samplesPerWorker * l)] = eval_fd(yfine, yfdobj)
                responsefdobjTrain = smooth_basis(yfine,
                                                  response[:, [i + samplesPerWorker * (l - 1) - 1 for i in train]],
                                                  basisobj).fd
                y.append(responsefdobjTrain)

            yfdobjTest = smooth_basis(yfine, response[:, test], basisobj).fd
            responseTest = eval_fd(yfine, yfdobjTest)
            boost_control = 10
            step_length = 0.5
            lin = linmod(x[0][1], y[0], betaList)
            step_length = 0.5
            start = time.time()
            federatedFunctionalGradBoostLSA(x, y, betaList, boost_control, step_length)
            end = time.time()
            time1[numworkersseq.index(numworkers), hh] = end - start
            start = time.time()
            federatedFunctionalGradBoostAvg(x, y, betaList, boost_control, step_length)
            end = time.time()
            time2[numworkersseq.index(numworkers), hh] = end - start
    print("Comparison of operation times of fed-GB-LSA (LSA) and fed-GB-Average (Avg)")
    print('operation times of fed-GB-LSA (LSA)')
    print(np.mean(time1, 1))
    print('operation times of fed-GB-Average (Avg)')
    print(np.mean(time2, 1))
    return 0

if __name__ == "__main__":
    main()



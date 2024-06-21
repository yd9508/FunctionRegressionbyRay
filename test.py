import numpy as np
from fda import *
import pickle
from federatedAlgs import *
from dataGenerator import *
import time
import sys
from tqdm import tqdm

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
trans_time1 = np.zeros([len(numworkersseq), 4])
sub_trans_time1 = np.zeros([len(numworkersseq), 4])
trans_data1 = np.zeros([len(numworkersseq), 4])
time2 = np.zeros([len(numworkersseq), 4])
trans_time2 = np.zeros([len(numworkersseq), 4])
sub_trans_time2 = np.zeros([len(numworkersseq), 4])
trans_data2 = np.zeros([len(numworkersseq), 4])

print(time.time())

log_path = "log/" + str(time.time()) + ".txt"

for numworkers in tqdm(numworkersseq):
    # 4-fold CV
    for hh in range(4):
        with open(log_path, "a+") as fw:
            fw.write(f'Number of local servers : {numworkers}; {hh} fold.\n')

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
        test1_status = federatedFunctionalGradBoostLSA(x, y, betaList, boost_control, step_length)
        end = time.time()
        time1[numworkersseq.index(numworkers), hh] = end - start
        trans_time1[numworkersseq.index(numworkers), hh] = test1_status[1]
        # sub_trans_time1[numworkersseq.index(numworkers), hh] = test1_status[3]
        trans_data1[numworkersseq.index(numworkers), hh] = test1_status[2]
        start = time.time()
        test2_status = federatedFunctionalGradBoostAvg(x, y, betaList, boost_control, step_length)
        end = time.time()
        time2[numworkersseq.index(numworkers), hh] = end - start
        trans_time2[numworkersseq.index(numworkers), hh] = test2_status[1]
        # sub_trans_time2[numworkersseq.index(numworkers), hh] = test2_status[3]
        trans_data2[numworkersseq.index(numworkers), hh] = test2_status[2]

with open(log_path, "a+") as fw:
    fw.write("Comparison of operation times of fed-GB-LSA (LSA) and fed-GB-Average (Avg)\n")
    fw.write('operation times of fed-GB-LSA (LSA)\n')
    fw.write(str(np.mean(time1, 1)))
    fw.write('\noperation times of fed-GB-Average (Avg)\n')
    fw.write(str(np.mean(time2, 1)))
    fw.write('\nAverage transmission times of fed-GB-LSA (LSA)\n')
    fw.write(str(np.mean(trans_time1, 1)))
    # fw.write(str(np.mean(sub_trans_time1, 1)))
    fw.write('\nAverage transmission times of fed-GB-Average (Avg)\n')
    fw.write(str(np.mean(trans_time2, 1)))
    # fw.write(str(np.mean(sub_trans_time2, 1)))
    fw.write('\nTotal transmission data (bytes) of fed-GB-LSA (LSA)\n')
    fw.write(str(np.mean(trans_data1, 1)))
    fw.write('\nTotal transmission data (bytes) of fed-GB-Average (Avg)\n')
    fw.write(str(np.mean(trans_data2, 1)))
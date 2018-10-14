# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:23:15 2018

@author: tf
"""

import numpy as np
import operator
import csv
import json

def readDataSet(filename: str) -> (np.ndarray, list):
    '''
    read data from csv file
    '''
    trainSet = []
    trainLabels = []
    with open(filename, encoding='utf-8-sig') as f: 
        f_csv = csv.DictReader(f)
        for row in f_csv:
            trainSet.append([float(row['SL']), float(row['EEG']), float(row['BP']), float(row['HR']), float(row['CIRCLUATION'])])
            trainLabels.append(int(row['ACTIVITY'])) 
    return np.array(trainSet), trainLabels

def testReadDataSet(filename: str) -> (np.ndarray, list, np.ndarray, list):
    '''
    read data from csv file
    '''
    trainSet = []
    trainLabels = []
    testSet = []
    testLabels = []
    with open(filename, encoding='utf-8-sig') as f: 
        f_csv = csv.DictReader(f)
        m = 16292
        for i, row in enumerate(f_csv):
            if i < 9 * m / 10:
                trainSet.append([float(row['SL']), float(row['EEG']), float(row['BP']), float(row['HR']), float(row['CIRCLUATION'])])
                trainLabels.append(int(row['ACTIVITY']))
            else:
                testSet.append([float(row['SL']), float(row['EEG']), float(row['BP']), float(row['HR']), float(row['CIRCLUATION'])])
                testLabels.append(int(row['ACTIVITY']))
            
    return np.array(trainSet), trainLabels, np.array(testSet), testLabels

def classify(inX: list, dataSet: np.ndarray, labels: list, k: int) -> int:
    '''
    classification with kNN
    '''
    dataSetSize = dataSet.shape[0];
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
    
def testClassify(dataSet: np.ndarray, labels: list, inMat: np.ndarray, inLabels: list, k: int) -> float:
    '''
    calculate error rate
    '''
    m, n = dataSet.shape
    errCnt = 0
    for i, inX in enumerate(inMat):
        dataSetSize = dataSet.shape[0];
        diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances**0.5
        sortedDistIndices = distances.argsort()
        classCount = {}
        for i in range(k):
            voteIlabel = labels[sortedDistIndices[i]]
            classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        if sortedClassCount[0][0] != inLabels[i]:
            errCnt += 1
    errRate = errCnt / m
    return errRate

def kNNpredict(inX: list) -> str:
    '''
    prediction with kNN
    return str(in json)
    inX: list len=5
    '''
    trainMat, trainLabels= readDataSet('falldeteciton.csv')
    label = classify(inX, trainMat, trainLabels, 3)
    data = {'SL': inX[0], 'EEG': inX[1], 'BP': inX[2], 'HR': inX[3], 'CIRCLUATION': inX[4], 'ACTIVITY': label}
    return json.dumps(data, sort_keys=True, indent=4, separators=(',', ': '))

'''
if __name__ == '__main__':
    trainMat, trainLabels, testMat, testLabels = testReadDataSet('falldeteciton.csv')
    errRate = testClassify(trainMat, trainLabels, testMat, testLabels, 4)
    print('Accuracy:', 1 - errRate)
'''
print(kNNpredict([39149.1, -2970, 21, 196, 1885]))
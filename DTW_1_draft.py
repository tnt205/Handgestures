import cv2
import numpy as np
import os
import csv

import tensorflow as tf
from scipy.spatial.distance import cityblock, cosine 
import matplotlib.pyplot as plt
from dtwParallel import dtw_functions as dtw
from tslearn.neighbors import KNeighborsTimeSeriesClassifier



# =============================================================================
# Define some utility functions
# =============================================================================


root = os.path.dirname(os.path.abspath(__file__))
traindir = root + "/" + "traindata"
testdir = root  + "/" + "test"
trainframes = root + "/traindata/frames"
testframes = testdir +"/" + "frames"

result = root + "/" + "Results.csv"
traindata1X = root + "/" + "train1X.csv"
traindata2X = root + "/" + "train2X.csv"

traindata1Y = root + "/" + "train1Y.csv"
traindata2Y = root + "/" + "train2Y.csv"

testdata1X = root + "/" + "test1X.csv"
testdata2X = root + "/" + "test2X.csv"
testdata1Y = root + "/" + "test1Y.csv"
testdata2Y = root + "/" + "test2Y.csv"

gestureLabel = {'Num0': 0,
                'Num1': 1,
                'Num2': 2,
                'Num3': 3,
                'Num4': 4,
                'Num5': 5,
                'Num6': 6,
                'Num7': 7,
                'Num8': 8,
                'Num9': 9,
                'FanDown': 10,
                'FanOn': 11,
                'FanOff': 12,
                'FanUp': 13,
                'LightOff': 14,
                'LightOn': 15,
                'SetThermo': 16}

train1X = np.loadtxt(traindata1X)
train1X = train1X.reshape(train1X.shape[0], train1X.shape[1]//7, 7)
train1Y = np.loadtxt(traindata1Y)

test1X = np.loadtxt(testdata1X)
test1X = test1X.reshape(test1X.shape[0], test1X.shape[1]//7, 7)
with open(testdata1Y, "r") as file:
    test1Y = file.read().split("\n")
    
s1 = test1X[0]
distance_list = []
"""
for i in range(train1X.shape[0]):
    s2 = train1X[i]
    dtw_distance = cosine(s1, s2)
    distance_list.append(dtw_distance)
    
"""
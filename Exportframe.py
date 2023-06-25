import cv2
import numpy as np
import os
import csv

import tensorflow as tf
import frameextractor as FE
import handshape_feature_extractor as hfe
from scipy.spatial.distance import cosine

# =============================================================================
# Define some utility functions
# =============================================================================

def filename_array(path):
    filelist = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    filelist.sort()
    return filelist

root = os.path.dirname(os.path.abspath(__file__))
traindir = root + "/" + "traindata"
testdir = root  + "/" + "test"
trainframes = root + "/traindata/frames"
testframes = testdir +"/" + "frames"

result = root + "/" + "Results.csv"
traindataX = root + "/" + "traindataX.csv"
traindataY = root + "/" + "traindataY.csv"
testdataX = root + "/" + "testdataX.csv"
testdataY = root + "/" + "testdataY.csv"

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



# Test on file "FanDown_Practice_1_TruongTran

def filename_array(path):
    filelist = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    filelist.sort()
    return filelist


def videoToFrames(video_path, frame_path, n, interval):
    
    #Extract the file name from video_path
    f = video_path.split("/")[-1]
    gestureName = f.split(".")[0]
    count = 0
    
    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame = video_length//2
    
    # Extract 
    for i in range(mid_frame - interval*n, mid_frame + interval*n, interval):
        cap.set(1, i)
        ret, frame = cap.read()
        filename = frame_path + "/" + gestureName + "_" + "%d.png" %(count+1)
        cv2.imwrite(filename, frame)
        count +=1
        
        
video_path = traindir + "/" + "FanDown_PRACTICE_2_MIRYALA.mp4"

videoToFrames(video_path, trainframes, 20, 2)

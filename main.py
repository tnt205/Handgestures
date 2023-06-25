# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 17:24:39 2023

@author: Ngoc Dao
"""
import cv2
import numpy as np
import os
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
## import the handfeature extractor class

def videoTodata(video_path, frames_path):
    videolist = filename_array(video_path)
    count = 0
    features = []
    label = {}
    for f in videolist:
        FE.frameExtractor(video_path + "/" + f, frames_path, count)
        label[count+1]=f
        count += 1
    
    imglist = filename_array(frames_path)
    for img in imglist:
        image = cv2.imread(frames_path + "/" + img, cv2.IMREAD_UNCHANGED)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f = hfe.HandShapeFeatureExtractor.get_instance()
        feature_vector = f.extract_feature( gray )
        features.append( [img,feature_vector] )
        
    return features, label
        

# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video

trainX, trainlabel = videoTodata(traindir, trainframes)
testX, testlabel = videoTodata(testdir, testframes)

output = []
""" For each image in test data, calculate cosine similarity with all
        images in train data. The predicted value will be the image that
        has highest loss value
        We track back the information of image and label and export it
        to Results.csv file
        """
for img in testX:
    # Extract video name from image data
    frame_number = int(img[0].split(".")[0])
    video_name = testlabel[frame_number].split(".")[0]
    
    # Initiate loss value
    loss = 0
    trainframe =0
    a = img[1][0]
    for i in trainX:
        b = i[1][0]
        similarity = 1-cosine(a,b)
        print(loss)
        print(similarity)
        
        if similarity > loss:
            loss = similarity
            trainframe = int(i[0].split(".")[0])
    
    ges_name = trainlabel[trainframe].split("_")[0]
    
    output.append(gestureLabel[ges_name])


np.savetxt(result, output, fmt='%i', delimiter=',')
    
    
    
    
         
    
            
    

            

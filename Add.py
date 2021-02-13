#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 08:46:41 2021

@author: kevin
"""

import cv2
from align_custom import AlignCustom
from face_feature import FaceFeature
from mtcnn_detect import MTCNNDetect
from tf_graph import FaceRecGraph
import openpyxl as xl
import datetime
import time
import sys
import json
import numpy as np
import pandas as pd

def create_manual_data():
    vs = cv2.VideoCapture(0); #get input from webcam
    new_name = input("Please input new user name: ")
    # roll=input("Unique ID: ")
  

    f = open('./facerec_128D.txt','r+');
    data_set = json.loads(f.read());
    person_imgs = {"Left" : [], "Right": [], "Center": []};
    person_features = {"Left" : [], "Right": [], "Center": []};
    print("Please start turning slowly. Press 'q' to save and add this new user to the dataset");
    while True:
        _, frame = vs.read();
        rects, landmarks = face_detect.detect_face(frame, 80);  # min face size is set to 80x80
        for (i, rect) in enumerate(rects):
            aligned_frame, pos = aligner.align(160,frame,landmarks[i]);
            if len(aligned_frame) == 160 and len(aligned_frame[0]) == 160:
                person_imgs[pos].append(aligned_frame)
                print("Face captured!")
                cv2.imshow("Captured face", aligned_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 30 or key == ord("q"):
            break


    # g = open('./names.txt','r');
    # jso
    # names = {2:'Rishabh'}
    names = json.loads(open('names.txt').read())
    names.update({(len(names)+2):new_name})
    json.dump(names, open("names.txt",'w'))


    for pos in person_imgs: #there r some exceptions here, but I'll just leave it as this to keep it simple
        person_features[pos] = [np.mean(extract_feature.get_features(person_imgs[pos]),axis=0).tolist()]
    data_set[new_name] = person_features;
    f = open('./facerec_128D.txt', 'w+');
    f.write(json.dumps(data_set))

    print("Done")


face_detect = None
aligner = None
extract_feature = None
FRGraph = None

if __name__ == '__main__':
    FRGraph = FaceRecGraph();
    aligner = AlignCustom();
    extract_feature = FaceFeature(FRGraph)
    face_detect = MTCNNDetect(FRGraph, scale_factor=2); #scale_factor, rescales image for faster detection
    create_manual_data()


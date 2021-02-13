import cv2
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

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        FRGraph = FaceRecGraph();
        self.aligner = AlignCustom();
        self.extract_feature = FaceFeature(FRGraph)
        self.face_detect = MTCNNDetect(FRGraph, scale_factor=2); #scale_factor, rescales image for faster detection

        self.person_imgs = {"Left" : [], "Right": [], "Center": []};
        self.person_features = {"Left" : [], "Right": [], "Center": []};
        self.video = cv2.VideoCapture(2)

        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        rects, landmarks = self.face_detect.detect_face(image, 80);  # min face size is set to 80x80
        for (i, rect) in enumerate(rects):
            aligned_frame, pos = self.aligner.align(160,image,landmarks[i]);
            # print(pos)
            if len(aligned_frame) == 160 and len(aligned_frame[0]) == 160:
                self.person_imgs[pos].append(aligned_frame)
                cv2.rectangle(image,(rect[0],rect[1]),(rect[0] + rect[2],rect[1]+rect[3]),(0,255,0),2) #draw bounding box for the face
                # print("Face captured!")
                # cv2.imshow("Captured face", aligned_frame)
        key = cv2.waitKey(1) & 0xFF

        
        
        # cv2.putText(frame,recog_data[i][0]+" - "+str(recog_data[i][1])+"%",(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    f = open('./facerec_128D.txt','r+');
    data_set = json.loads(f.read());

    for pos in camera.person_imgs: #there r some exceptions here, but I'll just leave it as this to keep it simple
        camera.person_features[pos] = [np.mean(camera.extract_feature.get_features(person_imgs[pos]),axis=0).tolist()]
    data_set[new_name] = camera.person_features;
    f = open('./facerec_128D.txt', 'w+');
    f.write(json.dumps(data_set))

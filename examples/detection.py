

import cv2
import sys

sys.path.append('..')

from face_ai_kit.FaceRecognition import FaceRecognition

face_lib = FaceRecognition(recognition='magface')

frame = cv2.imread('data/imga.jpg')
results1 = face_lib.face_detection(frame, align='square')
face1_roi = results1[0]["roi"]
face_img1 = frame[face1_roi[0][1]:face1_roi[1][1],face1_roi[0][0]:face1_roi[1][0]]

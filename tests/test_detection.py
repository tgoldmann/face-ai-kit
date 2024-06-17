import unittest
import sys

sys.path.append('..')
from face_ai_kit.FaceRecognition import FaceRecognition


import cv2
import os

class FaceAIKitTests(unittest.TestCase):
    def test_detection(self):

        script_dir = os.path.dirname(os.path.abspath(__file__))

        fce = FaceRecognition(recognition='arcface')
        frame_1 = cv2.imread(script_dir + '../examples/data/imga.jpg')
        #frame_2 = cv2.resize(frame_2, (640, 480))
 
        results1 = fce.face_detection(frame_1)
        face1_roi = results1[0]["roi"]
        face_img1 = frame_1[face1_roi[0][1]:face1_roi[1][1],face1_roi[0][0]:face1_roi[1][0]]

        self.assertEqual(len(results1), 1)


    def test_lendmarks(self):

        script_dir = os.path.dirname(os.path.abspath(__file__))

        fce = FaceRecognition(recognition='arcface')
        frame_1 = cv2.imread(script_dir + '../examples/data/imga.jpg')
        #frame_2 = cv2.resize(frame_2, (640, 480))
 
        results1 = fce.face_detection(frame_1)
        face1_roi = results1[0]["roi"]
        coordinates = fce.landmarks(frame_1, face1_roi)


        #self.assertEqual(len(results1), 1)

    def test_recognition(self):

        script_dir = os.path.dirname(os.path.abspath(__file__))

        fce = FaceRecognition(recognition='arcface')

        frame_1 = cv2.imread('../examples/data/imga.jpg')
        frame_2 = cv2.imread('../examples/data/imgb.jpg')

        results1 = fce.face_detection(frame_1, align='square')

        #fce.rotation(frame_1,face1_roi )
        results2 = fce.face_detection(frame_2)

        distance = fce.verify_rois(frame_1, results1[0]["roi"],frame_2, results2[0]["roi"])

        print(distance)
        #self.assertEqual(len(results1), 1)
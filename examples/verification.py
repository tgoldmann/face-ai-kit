

from face_ai_kit.FaceRecognition import FaceRecognition
import cv2

fce = FaceRecognition(recognition='magface')

frame_1 = cv2.imread('data/imga.jpg')
frame_2 = cv2.imread('data/imga.jpg')

results1 = fce.face_detection(frame_1, align='square')
face1_roi = results1[0]["roi"]
face_img1 = frame_1[face1_roi[0][1]:face1_roi[1][1],face1_roi[0][0]:face1_roi[1][0]]

results2 = fce.face_detection(frame_2)
face2_roi = results2[0]["roi"]
face_img2 = frame_2[face2_roi[0][1]:face2_roi[1][1],face2_roi[0][0]:face2_roi[1][0]]

distance = fce.verify_rois(frame_1, results1[0]["roi"],frame_2, results2[0]["roi"])

fce.represent(face_img1)
print(distance)


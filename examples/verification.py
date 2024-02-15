
"""
Example: Face verification

Author: Tomas Goldmann
Date Created: Dec 26, 2023
Date Modified: Dec 26, 2023
License: MIT License
"""


import cv2
import sys


sys.path.append('..')


from face_ai_kit.FaceRecognition import FaceRecognition

#1) Load library
face_lib = FaceRecognition(recognition='magface_cwh')

#2) Load images

frame_1 = cv2.imread('data/imga.jpg')
frame_2 = cv2.imread('data/imgb.jpg')


#3) Get face ROI and aligned face image
results1 = face_lib.face_detection(frame_1, align='keypoints')
face1_roi = results1[0]["roi"]
face_img1 = results1[0]["img"]


results2 = face_lib.face_detection(frame_2, align='keypoints')
face2_roi = results2[0]["roi"]
face_img2 = results2[0]["img"]

#4A - use face ROI for face recgonition

distance = face_lib.verify_rois(frame_1, results1[0]["roi"],frame_2, results2[0]["roi"])
print(f"L2 distance between faces {distance}")

#4B - use face ROI for face recgonition

distance = face_lib.verify(face_img1, face_img2)
print(f"L2 distance between faces {distance}")


cv2.imshow("Image1", face_img1)
cv2.imshow("Image2", face_img2)
cv2.waitKey(0)

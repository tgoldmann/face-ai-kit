
"""
Example: Face detection

Author: Tomas Goldmann
Date Created: Dec 26, 2023
Date Modified: Dec 26, 2023
License: MIT License
"""

import cv2
import sys

from face_ai_kit.FaceRecognition import FaceRecognition


def draw_circles_on_image(image, coordinates, radius=2, color=(0, 0, 255), thickness=2):
    # Load the image
    # Draw circles based on the coordinates
    for (x, y) in coordinates:
        #print(x,y)
        cv2.circle(image, (int(x), int(y)), radius, color, thickness)

#1) Load library
face_lib = FaceRecognition(recognition='magface')


#2) Load image
image = cv2.imread('data/imga.jpg')

#3) Detect faces in a image
results = face_lib.face_detection(image, align='none')


#4) If the faces were detected, then a output is consist of list of dict. Each dict includes? img, roi, keypoints, score. 
# Img is a aligned face image in numpy format.

face_roi = results[0]["roi"]
face_img = results[0]["img"]

#5) Draw a rectangle based on the face ROI. The face_img is different from the face in the ROI area due to transformations when using kepoints align.

cv2.rectangle(image, face_roi[0], face_roi[1], (255,0,0), thickness=2)

#6) Draw landmarks from RetinaFace
draw_circles_on_image(image, results[0]["keypoints"])


#7) Show image with detection
cv2.imshow("Test", image)
cv2.waitKey(0) 
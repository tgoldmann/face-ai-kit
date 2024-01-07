
"""
Example: Face landmarks detection

Author: Tomas Goldmann
Date Created: Dec 26, 2023
Date Modified: Dec 26, 2023
License: MIT License
"""


import cv2
import sys


sys.path.append('..')
from face_ai_kit.FaceRecognition import FaceRecognition


def draw_circles_on_image(image, coordinates, _x,_y, radius=2, color=(0, 0, 255), thickness=2):
    # Load the image
    # Draw circles based on the coordinates
    for (x, y,z) in coordinates:
        #print(x,y)
        cv2.circle(image, (int(x+_x), int(y+_y)), radius, color, thickness)


face_lib = FaceRecognition(recognition='magface')

image = cv2.imread('data/imga.jpg')

results1 = face_lib.face_detection(image, align='square')
face1_roi = results1[0]["roi"]
face_img1 = image[face1_roi[0][1]:face1_roi[1][1],face1_roi[0][0]:face1_roi[1][0]]

coordinates = face_lib.landmarks(image, face1_roi)

draw_circles_on_image(image, coordinates,face1_roi[0][0],face1_roi[0][1])
cv2.imshow("Image2", image)
cv2.waitKey(0)
    

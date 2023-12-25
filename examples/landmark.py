

from face_ai_kit.FaceRecognition import FaceRecognition
import cv2


def draw_circles_on_image(image, coordinates, _x,_y, radius=2, color=(0, 0, 255), thickness=2):
    # Load the image
    # Draw circles based on the coordinates
    for (x, y) in coordinates:
        #print(x,y)
        cv2.circle(image, (int(x+_x), int(y+_y)), radius, color, thickness)


face_lib = FaceRecognition(recognition='magface')

frame_1 = cv2.imread('data/imga.jpg')
frame_1 = cv2.resize(frame_1, (640, 480))

results1 = face_lib.face_detection(frame_1, align='square')
face1_roi = results1[0]["roi"]
face_img1 = frame_1[face1_roi[0][1]:face1_roi[1][1],face1_roi[0][0]:face1_roi[1][0]]

coordinates = face_lib.landmarks(frame_1, face1_roi)

print(coordinates)
draw_circles_on_image(frame_1, coordinates,face1_roi[0][0],face1_roi[0][1])


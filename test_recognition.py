

from face_ai_kit.FaceRecognition import FaceRecognition
import cv2
import numpy as np


def draw_circles_on_image(image, coordinates, _x,_y, radius=2, color=(0, 0, 255), thickness=2):
    # Load the image

    # Ensure the coordinates array has an even number of elements

    # Draw circles based on the coordinates
    for (x, y) in coordinates:
        #print(x,y)
        cv2.circle(image, (int(x+_x), int(y+_y)), radius, color, thickness)

    # Display the image with circles
    cv2.imshow("Image with Circles", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_circles_on_image3D(image, coordinates, _x,_y, radius=2, color=(0, 0, 255), thickness=2):
    # Load the image

    # Ensure the coordinates array has an even number of elements

    # Draw circles based on the coordinates
    for (x, y, z) in coordinates:
        #print(x,y)
        cv2.circle(image, (int(x+_x), int(y+_y)), radius, color, thickness)

    # Display the image with circles
    cv2.imshow("Image with Circles", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw(image, items):
    for item in items:

        roi = item['roi']
        keypoints = item['keypoints']
        score = item['score']

        # Extract coordinates
        x1, y1= roi
        keypoints = [(int(x), int(y)) for x, y in keypoints]

        # Draw the rectangle
        image_with_rect = cv2.rectangle(image, x1, y1, (0, 255, 0), 2)

        # Draw keypoints
        for keypoint in keypoints:
            cv2.circle(image_with_rect, keypoint, 2, (0, 0, 255), -1)

        # Add text with score
        cv2.putText(image_with_rect, f'Score: {score:.2f}', (x1[0], x1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

fce = FaceRecognition(recognition='magface')
#print(fce.verify_one('image0.jpeg','image1.jpeg'))

frame_1 = cv2.imread('tests/image1.jpeg')
frame_1 = cv2.resize(frame_1, (640, 480))

frame_2 = cv2.imread('tests/image0.jpeg')
#frame_2 = cv2.resize(frame_2, (640, 480))


frame = cv2.imread('tests/imgb.png')
results1 = fce.face_detection(frame, align='square')
face1_roi = results1[0]["roi"]
face_img1 = frame[face1_roi[0][1]:face1_roi[1][1],face1_roi[0][0]:face1_roi[1][0]]
print(face_img1.shape)
print(results1)
#cv2.imshow('Test', face_img1)
#cv2.waitKey(0)


results1 = fce.face_detection(frame_1, align='square')
face1_roi = results1[0]["roi"]
face_img1 = frame_1[face1_roi[0][1]:face1_roi[1][1],face1_roi[0][0]:face1_roi[1][0]]


#fce.rotation(frame_1,face1_roi )
results2 = fce.face_detection(frame_2)
face2_roi = results2[0]["roi"]
face_img2 = frame_2[face2_roi[0][1]:face2_roi[1][1],face2_roi[0][0]:face2_roi[1][0]]


coordinates = fce.landmarks(frame_1, face1_roi)
#print(coordinates)
#draw_circles_on_image(frame_1, coordinates,face1_roi[0][0],face1_roi[0][1])
draw_circles_on_image3D(frame_1, coordinates,face1_roi[0][0],face1_roi[0][1])



distance = fce.verify_rois(frame_1, results1[0]["roi"],frame_2, results2[0]["roi"])

fce.represent(face_img1)
print(distance)
exit(0)

#distance = fce.verify_batch([(frame_1, results1[0]["roi"]),(frame_1, results1[0]["roi"])],[(frame_1, results1[0]["roi"]),(frame_1, results1[0]["roi"])])


print(distance)
draw(frame_1, results1)
draw(frame_2, results2)

cv2.imshow('Test', frame_1)
cv2.imshow('Test2', frame_2)

cv2.waitKey(0)

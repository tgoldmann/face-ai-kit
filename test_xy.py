import cv2
from RetinaFaceEfficient.RetinaFaceOnnx import RetinaFaceOnnx

from align_trans import get_reference_facial_points, warp_and_crop_face, warp_and_crop_face_by_box


det = RetinaFaceOnnx('config/hailo_test_det.yaml','RetinaFaceEfficient/resnet_dynamic.onnx')
img = cv2.imread('../data/tomas.jpg')
results = det.interface('../data/tomas.jpg')
print(results)

face = warp_and_crop_face_by_box(img, [results[0]['roi'][0][0], results[0]['roi'][0][1], results[0]['roi'][1][0], results[0]['roi'][1][1] ])
cv2.imwrite("out.png", face)

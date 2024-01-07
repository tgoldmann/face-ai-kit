
from .RetinaFaceDetectorFactory import FaceDetectorFactory

FaceDetectorFactory.register_provider('onnx-cpu', 'RetinaFaceOnnx')
FaceDetectorFactory.register_provider('rknn-3568', 'RetinaFaceRknn3568')
FaceDetectorFactory.register_provider('rknn-3588', 'RetinaFaceRknn3588')
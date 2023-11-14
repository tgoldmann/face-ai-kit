
from .RetinaFaceDetectorFactory import FaceDetectorFactory
from .inference.RetinaFaceOnnx import RetinaFaceOnnx
from .inference.RetinaFaceRknn3568 import RetinaFaceRknn3568
from .inference.RetinaFaceRknn3588 import RetinaFaceRknn3588

FaceDetectorFactory.register_provider('onnx-cpu', RetinaFaceOnnx)
FaceDetectorFactory.register_provider('rknn-3568', RetinaFaceRknn3568)
FaceDetectorFactory.register_provider('rknn-3588', RetinaFaceRknn3588)
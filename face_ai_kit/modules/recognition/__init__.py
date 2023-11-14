
from .RecognitionFactory import RecognitionFactory
from .arcface.inference.ArcFaceOnnx import ArcFaceOnnx
from .arcface.inference.ArcFaceRknn3568 import ArcFaceRknn3568
from .arcface.inference.ArcFaceRknn3588 import ArcFaceRknn3588

from .magface.inference.MagFaceOnnx import MagFaceOnnx
RecognitionFactory.register_provider('arcface', 'onnx-cpu', ArcFaceOnnx)
RecognitionFactory.register_provider('arcface','rknn-3568', ArcFaceRknn3568)
RecognitionFactory.register_provider('arcface','rknn-3588', ArcFaceRknn3588)


RecognitionFactory.register_provider('magface', 'onnx-cpu', MagFaceOnnx)
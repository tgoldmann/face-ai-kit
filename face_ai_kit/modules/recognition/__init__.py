
from .RecognitionFactory import RecognitionFactory

RecognitionFactory.register_provider('arcface', 'onnx-cpu', 'ArcFaceOnnx')
RecognitionFactory.register_provider('magface', 'onnx-cpu', 'MagFaceOnnx')
RecognitionFactory.register_provider('magface_cwh', 'onnx-cpu', 'MagFaceCWHOnnx')

RecognitionFactory.register_provider('arcface', 'onnx-gpu', 'ArcFaceOnnx')
RecognitionFactory.register_provider('magface', 'onnx-gpu', 'MagFaceOnnx')
RecognitionFactory.register_provider('magface_cwh', 'onnx-gpu', 'MagFaceCWHOnnx')

RecognitionFactory.register_provider('arcface','rknn-3568', 'ArcFaceRknn3568')
RecognitionFactory.register_provider('arcface','rknn-3588', 'ArcFaceRknn3588')



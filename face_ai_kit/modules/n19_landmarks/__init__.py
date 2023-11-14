
from .N19LandmarksFactory import N19LandmarksFactory
from .inference.KeypointsOnnx import KeypointsOnnx


N19LandmarksFactory.register_provider('n19', 'onnx-cpu', KeypointsOnnx)

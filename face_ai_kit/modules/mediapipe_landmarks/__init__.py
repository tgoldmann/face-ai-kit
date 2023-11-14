
from .MediapipeLandmarksFactory import MediapipeLandmarksFactory
from .inference.MediapipeLandmarksOnnx import MediapipeLandmarksOnnx

MediapipeLandmarksFactory.register_provider('mediapipe', 'onnx-cpu', MediapipeLandmarksOnnx)

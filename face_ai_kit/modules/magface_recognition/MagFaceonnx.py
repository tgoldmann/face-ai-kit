import onnxruntime as ort
import numpy as np
import cv2

#from .MagFace.magface import MagFace


class MagFaceOnnx:

    def __init__(self, model, provider) -> None:
        self.size = 112
        print(provider)

        self.algorithm = MagFace()
        self.algorithm_str = 'magface'
        pass

    def inference(self, image):

        outputs1 = self.algorithm.interface(warped_face)


    def inference_batch(self, image_batch):


        return outputs



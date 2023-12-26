"""
Description: MagFace 

Author: Tomas Goldmann
Date Created: Dec 26, 2023
Date Modified: Dec 26, 2023
License: MIT License
"""

import onnxruntime as ort
import numpy as np
import cv2

from abc import ABCMeta, abstractmethod

from ....core.timer import Timer


class MagFace:

    def __init__(self, provider, model) -> None:
        self.size = 112
        self.provider = provider

        self._t = {'forward_pass': Timer(), 'misc': Timer()}


    @abstractmethod
    def infer(img):
        
        pass

    def inference(self, image):
        if image.shape[0]!=image.shape[1]:
            raise Exception

        image = cv2.resize(image, (self.size, self.size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_img = image/255.
        face_img = face_img.transpose(2, 0, 1)

        face_img = np.expand_dims(face_img,0).astype(np.float32)
        outputs1 = self.infer(face_img)
        return outputs1[0]

    def inference_batch(self, image_batch):
        image_batch = np.array(image_batch).astype(np.float32)/255.
        print(image_batch.shape)
        outputs = self.infer(image_batch)
        return outputs

        



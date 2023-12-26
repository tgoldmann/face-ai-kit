"""
Description: Interface for Mediapipe landmarks detector 

Author: Tomas Goldmann
Date Created: Dec 26, 2023
Date Modified: Dec 26, 2023
License: MIT License
"""


import onnxruntime as ort
import numpy as np
import cv2

from abc import ABCMeta, abstractmethod

from ...core.timer import Timer


class MediapipeLandmarks:

    def __init__(self, provider, model) -> None:
        self.size = 192
        self.provider = provider

        self._t = {'forward_pass': Timer(), 'misc': Timer()}


    @abstractmethod
    def infer(img):
        
        pass

    def inference(self, image):
        if image.shape[0]!=image.shape[1]:
            raise Exception
        scale = self.size/image.shape[0]
        image = cv2.resize(image, (self.size, self.size))[:,:,::-1] 
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_img = image/255.
        face_img = np.expand_dims(face_img,0).astype(np.float32)

        face_img = np.asarray(face_img, dtype=np.float32).transpose(0,3,1,2)

        score, outputs1 = self.infer(face_img)


        return outputs1[0]/scale

    def inference_batch(self, image_batch):
        image_batch = np.array(image_batch).astype(np.float32)/255.

        outputs = self.infer(image_batch)
        return outputs



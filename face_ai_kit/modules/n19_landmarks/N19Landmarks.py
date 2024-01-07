"""
Description: N19 landmarks detector 

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


class Keypoints:

    def __init__(self, provider, model) -> None:
        self.size = 256
        self.provider = provider

        self._t = {'forward_pass': Timer(), 'misc': Timer()}


    @abstractmethod
    def infer(img):
        
        pass



    def inference(self, image, pad_x, pad_y):
        if image.shape[0]!=image.shape[1]:
            raise Exception
        scale = self.size/image.shape[0]
        print(image.shape, scale)

        image = cv2.resize(image, (self.size, self.size))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_img = image/255.
        face_img = np.expand_dims(face_img,0).astype(np.float32)
        outputs1 = self.infer(face_img)

        predicted_labels_array = (lambda point_cor: (((point_cor)*int((self.size/scale)/2)+int((self.size/scale)/2))))(outputs1[0])
        predicted_labels_array = predicted_labels_array.reshape(-1,2)

        zeros_array = np.zeros((98,1))
        predicted_labels_array = np.concatenate((predicted_labels_array,zeros_array),axis = 1)
        predicted_labels_array = predicted_labels_array + np.array([pad_x, pad_y,0])

        return predicted_labels_array

    def inference_batch(self, image_batch):
        image_batch = np.array(image_batch).astype(np.float32)/255.
        outputs = self.infer(image_batch)
        return outputs



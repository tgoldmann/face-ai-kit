"""
Description: RetinaFace detector.

Author: Tomas Goldmann
Date Created: Dec 26, 2023
Date Modified: Dec 26, 2023
License: MIT License
"""


import logging as log
import sys
import cv2
import numpy as np

from pathlib import Path
from abc import ABCMeta, abstractmethod


from ...core.timer import Timer
from .postprocessing import RetinaPostProcessing
from ...core.transforms import Transforms
from ...core.align_trans import  warp_and_crop_face



class RetinaFace():

    def __init__(self, inference, model_path, cfg_det) -> None:
        log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

        self.cfg = cfg_det

        self.postprocessing = RetinaPostProcessing(cfg_det["nms_threshold"],  cfg_det["conf_threshold"], cfg_det["variance"], cfg_det, image_size=(cfg_det['resolution'][1], cfg_det['resolution'][0]))

        self._t = {'forward_pass': Timer(), 'misc': Timer()}

    def clip_coordinates(self,coordinates, image_shape):
        x_min, y_min = coordinates[0]
        x_max, y_max = coordinates[1]

        x_min = max(0, min(x_min, image_shape[1]))
        y_min = max(0, min(y_min, image_shape[0]))
        x_max = max(0, min(x_max, image_shape[1]))
        y_max = max(0, min(y_max, image_shape[0]))

        return  ((x_min, y_min), (x_max, y_max))

    @abstractmethod
    def infer(img):
        
        pass

    def detect(self, image, align='keypoints'):

        if isinstance(image, np.ndarray):
                img_raw = image
        elif isinstance(image, str):
            try:
                img_raw = cv2.imread(image, cv2.IMREAD_COLOR)

                if img_raw is  None:
                    print("Image not loaded. Check the file path.")
                    return None, None
            except Exception as e:
                print("An error occurred:", str(e))
        else:
            return None, None

        img_to_detection = np.float32(img_raw)
        im_height, im_width, _ = img_to_detection.shape

        resize = 1.0
        scale = [img_to_detection.shape[1], img_to_detection.shape[0], img_to_detection.shape[1], img_to_detection.shape[0]]
        img = img_to_detection - (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = np.float32(np.expand_dims(img, 0))

        outputs = self.infer(img )

        loc =  outputs[0]
        conf =  outputs[1]
        landms =  outputs[2]



        dets, landms = self.postprocessing.postprocessing(loc, conf, landms, img_raw.shape, scale, resize,0,0)
        dets = np.concatenate((dets, landms), axis=1)

        # --------------------------------------------------------------------
        output = list()
        for b in dets:
            score = b[4]
            if b[4] < self.cfg["vis_thres"]:
                continue
            b = list(map(int, b))

            roi = self.clip_coordinates(((b[0], b[1]), (b[2], b[3])), (im_height, im_width))
            kpts= [(b[5], b[6]), (b[7], b[8]),(b[9], b[10]),(b[11], b[12]),(b[13], b[14])]
            if align=='none':
                roi = ((b[0], b[1]), (b[2], b[3]))
                warped_face = img_raw[b[1]:b[3], b[0]:b[2]]
            elif align == 'square':
                warped_face, xyxy = Transforms.return_one(np.array([[b[0], b[1], b[2], b[3]]]), img_raw,  gain=1.2, pad=0, square=True, BGR=True)
                xyxy = xyxy[0]
                roi=((xyxy[0], xyxy[1]),(xyxy[2], xyxy[3]))
            elif align=='keypoints':
                self.crop_size = 112
                warped_face, roi = warp_and_crop_face(np.array(img_raw), kpts, None, crop_size=(self.crop_size, self.crop_size))
                roi=((int(roi[0][0]), int(roi[0][1])),(int(roi[2][0]), int(roi[2][1])))
            else:
                print('Bad align argument')
            top_left = (max(0, roi[0][0]), max(0, roi[0][1]))
            bottom_right = (min(img_raw.shape[1], roi[1][0]), min(img_raw.shape[0], roi[1][1]))

            roi = (top_left, bottom_right)

            output.append({'img': warped_face, 'roi':roi, 'keypoints': kpts,'score': score})

        return output, img_raw


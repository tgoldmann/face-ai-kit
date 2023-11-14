#!/usr/bin/env python3

# retina_postprocessing.py

"""
Description: Script to extrarct detection from neural network output
Author: Tomas Goldmann, some parts are based on https://github.com/biubug6/Pytorch_Retinaface/
Date Created: May 2, 2023
Date Modified: September 4, 2023
Version: 1.0
Python Version: 3.9
License: MIT License
"""


from .retina_utils.utils.box_utils import decode, decode_landm
from ...core.timer import Timer
from .retina_utils.layers.functions.prior_box import PriorBox
from .retina_utils.utils.nms.py_cpu_nms import py_cpu_nms


import numpy as np
import cv2

class RetinaPostProcessing():
    def __init__(self,nms_threshold, confidence_threshold, variance, cfg, image_size=(480, 640), ceilx=True):
        self.nms_threshold = nms_threshold
        self.confidence_threshold = confidence_threshold   
        self.variance = variance
        self.cfg = cfg
        self.size = image_size
        self.ceilx= ceilx

        if self.size[0]!=None or self.size[1]!=None:
            priorbox = PriorBox(self.cfg, image_size=image_size, ceilx = ceilx)
            self.priors = priorbox.forward()


    def postprocessing(self, loc, conf, landms, shape, scale, resize, x_margin, y_margin):

        scale_w = self.size[1]
        scale_h = self.size[0]

        if self.size[0]==None or self.size[1]==None:

            priorbox = PriorBox(self.cfg, image_size=shape, ceilx = self.ceilx)
            self.priors = priorbox.forward()
            scale_w = shape[1]
            scale_h = shape[0]

        scale1 = [scale_w, scale_h, scale_w, scale_h, scale_w, scale_h, scale_w, scale_h,scale_w, scale_h]


        prior_data = self.priors
        boxes = decode(loc[0], prior_data, self.cfg['variance'])

        boxes =  boxes * scale
        boxes[:,0] = boxes[:,0]- x_margin
        boxes[:,1] = boxes[:,1]- y_margin
        boxes[:,2] = boxes[:,2]- x_margin
        boxes[:,3] = boxes[:,3]- y_margin
        boxes = boxes / resize

        scores = conf[0][:, 1]
        inds = np.where(scores > self.confidence_threshold)[0]
        landms = decode_landm(landms[0], prior_data, self.cfg['variance'])


        landms =  landms * scale1
        for i in range(0,5):
            landms[:,i*2] = landms[:,i*2]- x_margin
            landms[:,i*2+1] = landms[:,i*2+1]- y_margin

        landms = landms / resize

        # ignore low scores
        boxes = boxes[inds]
        #print(boxes)
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]
        #print(landms)
        # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]

        #dets = np.concatenate((dets, landms), axis=1)
        #landms = landms.reshape(-1,2)
        return dets, landms


    # https://github.com/biubug6/Pytorch_Retinaface/
    def plot(self, dets, lands, image):
        for b, l in zip(dets, lands):
            l = list(map(int, l))
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))

            cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)

            cx = b[0]
            cy = b[1] + 12
            cv2.putText(image, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(image, (l[0], l[1]), 1, (0, 0, 255), 4)
            cv2.circle(image, (l[2], l[3]), 1, (0, 255, 255), 4)
            cv2.circle(image, (l[4], l[5]), 1, (255, 0, 255), 4)
            cv2.circle(image, (l[6], l[7]), 1, (0, 255, 0), 4)
            cv2.circle(image, (l[8], l[9]), 1, (255, 0, 0), 4)
        return image
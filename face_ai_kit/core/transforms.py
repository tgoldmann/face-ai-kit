"""
Description: Functions to create squared ROI around face

Author:  
Date Created: Dec 26, 2023
Date Modified: Dec 26, 2023
License: MIT License
"""

import numpy as np
import cv2

class Transforms:

    @staticmethod
    def xyxy2xywh(x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    @staticmethod
    def xywh2xyxy(x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
    
    @staticmethod
    def clip_coords(boxes, img_shape, step=2):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0::step].clip(0, img_shape[1])  # x1
        boxes[:, 1::step].clip(0, img_shape[0])  # y1

    @staticmethod
    def return_one(xyxy, im,  gain=1.02, pad=10, square=False, BGR=False):
        #convert xyxy to xywh
        b = Transforms.xyxy2xywh(xyxy)
        if square:
            b[:, 2:] =  np.expand_dims(b[:, 2:].max(1), axis=1)    # attempt rectangle to square

        b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad

        xyxy = Transforms.xywh2xyxy(b)
        Transforms.clip_coords(xyxy, im.shape)
        crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2])]
        return crop if BGR else crop[..., ::-1], xyxy

    @staticmethod

    def square_pads(current_img,target_size=(112,112)):
        # resize and padding
        if current_img.shape[0] > 0 and current_img.shape[1] > 0:
            factor_0 = target_size[0] / current_img.shape[0]
            factor_1 = target_size[1] / current_img.shape[1]
            factor = min(factor_0, factor_1)

            dsize = (
                int(current_img.shape[1] * factor),
                int(current_img.shape[0] * factor),
            )
            current_img = cv2.resize(current_img, dsize)

            diff_0 = target_size[0] - current_img.shape[0]
            diff_1 = target_size[1] - current_img.shape[1]

            current_img = np.pad(
                current_img,
                (
                    (diff_0 // 2, diff_0 - diff_0 // 2),
                    (diff_1 // 2, diff_1 - diff_1 // 2),
                    (0, 0),
                ),
                "constant",
            )

        # double check: if target image is not still the same size with target.
        if current_img.shape[0:2] != target_size:
            current_img = cv2.resize(current_img, target_size)

        return current_img

    @staticmethod

    def add_square_padding(current_img):
        # padding to make it square
        if current_img.shape[0] > 0 and current_img.shape[1] > 0:
            max_dim = max(current_img.shape[0], current_img.shape[1])

            diff_0 = max_dim - current_img.shape[0]
            diff_1 = max_dim - current_img.shape[1]

            current_img = np.pad(
                current_img,
                (
                    (diff_0 // 2, diff_0 - diff_0 // 2),
                    (diff_1 // 2, diff_1 - diff_1 // 2),
                    (0, 0),
                ),
                "constant",
            )

        return current_img, diff_0 // 2, diff_1 // 2
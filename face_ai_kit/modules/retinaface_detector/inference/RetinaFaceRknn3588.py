"""
Description: RetinaFace: Interface for rknn3588 - not implemented yet.

Author: Tomas Goldmann
Date Created: Dec 26, 2023
Date Modified: Dec 26, 2023
License: MIT License
"""
import numpy as np

from ..RetinaFace import RetinaFace
from rknnlite.api import RKNNLite


class RetinaFaceRknn3588(RetinaFace):

    def __init__(self, inference, model_path, cfg_det) -> None:
        super().__init__(inference, model_path, cfg_det)

        self.session  = RKNNLite()
        ret = self.session.load_rknn(model_path)
        if ret != 0:
            print('Export rknn model failed!')
            exit(ret)
        # Init runtime environment
        print('--> Init runtime environment')
        ret = self.session.init_runtime()
        if ret != 0:
            print('Init runtime environment failed!')
            exit(ret)

        
    def infer(self, img):
        img = np.float32(np.expand_dims(img, 0))

        self._t['forward_pass'].tic()

        boxses = self.session.inference(inputs=[img])
        loc = boxses[0].reshape(1,-1,4)
        conf = boxses[1].reshape(1,-1,2)
        landms = boxses[2].reshape(1,-1,10)

        self._t['forward_pass'].toc()

        return [loc, conf, landms]

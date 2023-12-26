"""
Description: RetinaFace: Interface for rknn3568 - not implemented yet.

Author: Tomas Goldmann
Date Created: Dec 26, 2023
Date Modified: Dec 26, 2023
License: MIT License
"""


from ..RetinaFace import RetinaFace

import onnxruntime as ort


class RetinaFaceRknn3568(RetinaFace):

    def __init__(self, inference, model_path, cfg_det) -> None:
        super().__init__(inference, model_path, cfg_det)

        #self.session = ort.InferenceSession(model_path, providers=[ 'CUDAExecutionProvider'])
        
    def infer(self, img):

        self._t['forward_pass'].tic()
        #outputs = self.session.run(None, {'input0': img})
        self._t['forward_pass'].toc()

        return None
        # Implement face detection using ONNX
        print(f"Using ONNX provider to detect faces with model at {self.model_path}")

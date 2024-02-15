"""
Description: ArcFace - interface for CPU and GPU

Author: Tomas Goldmann
Date Created: Dec 26, 2023
Date Modified: Dec 26, 2023
License: MIT License
"""


from ..MagFace import MagFace

import onnxruntime as ort

class MagFaceOnnx(MagFace):

    def __init__(self, inf, model_path) -> None:
        super().__init__(inf, model_path)
        providers = list()
        if inf=='onnx-cpu':
            providers = ['CPUExecutionProvider']
        elif inf=='onnx-gpu':
            providers = ['CUDAExecutionProvider']
        else:
            raise  RuntimeError("Face recognition: Unkown provider")

        self.session = ort.InferenceSession(model_path, providers=providers)
        
    def infer(self, face_img):

        self._t['forward_pass'].tic()
        outputs = self.session.run(None, {'input.1': face_img})
        self._t['forward_pass'].toc()

        return outputs


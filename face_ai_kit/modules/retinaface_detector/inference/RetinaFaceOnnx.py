
from ..RetinaFace import RetinaFace

import onnxruntime as ort
import numpy as np

class RetinaFaceOnnx(RetinaFace):

    def __init__(self, inference, model_path, cfg_det) -> None:
        super().__init__(inference, model_path, cfg_det)

        providers = list()
        if inference=='onnx-cpu':
            providers = ['CPUExecutionProvider']
        elif inference=='onnx-gpu':
            providers = ['CUDAExecutionProvider']
        else:
            raise  RuntimeError("Face recognition: Unkown provider")

        self.session = ort.InferenceSession(model_path, providers=providers)
        
    def infer(self, img):
        img = img.transpose(2, 0, 1)
        img = np.float32(np.expand_dims(img, 0))

        self._t['forward_pass'].tic()
        outputs = self.session.run(None, {'input0': img})
        self._t['forward_pass'].toc()

        return outputs

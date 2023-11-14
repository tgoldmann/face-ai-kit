
from ..ArcFace import ArcFace

import onnxruntime as ort

class ArcFaceOnnx(ArcFace):

    def __init__(self, inf, model_path) -> None:
        super().__init__(inf, model_path)
        self.session = ort.InferenceSession(model_path, providers=[ 'CUDAExecutionProvider'])
        
    def infer(self, face_img):

        self._t['forward_pass'].tic()
        outputs = self.session.run(None, {'face_input': face_img})
        self._t['forward_pass'].toc()

        return outputs



from ..N19Landmarks import Keypoints

import onnxruntime as ort

class KeypointsOnnx(Keypoints):

    def __init__(self, inf, model_path) -> None:
        super().__init__(inf, model_path)
        self.session = ort.InferenceSession(model_path, providers=[ 'CUDAExecutionProvider'])
        
    def infer(self, face_img):

        self._t['forward_pass'].tic()
        outputs = self.session.run(None, {'input': face_img})
        self._t['forward_pass'].toc()

        return outputs


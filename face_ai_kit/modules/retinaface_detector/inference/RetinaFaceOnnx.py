
from ..RetinaFace import RetinaFace

import onnxruntime as ort


class RetinaFaceOnnx(RetinaFace):

    def __init__(self, inference, model_path, cfg_det) -> None:
        super().__init__(inference, model_path, cfg_det)

        self.session = ort.InferenceSession(model_path, providers=[ 'CUDAExecutionProvider'])
        
    def infer(self, img):

        self._t['forward_pass'].tic()
        outputs = self.session.run(None, {'input0': img})
        self._t['forward_pass'].toc()

        return outputs
        # Implement face detection using ONNX
        print(f"Using ONNX provider to detect faces with model at {self.model_path}")


from ..MediapipeLandmarks import MediapipeLandmarks

import onnxruntime as ort
import numpy as np

class MediapipeLandmarksOnnx(MediapipeLandmarks):

    def __init__(self, inf, model_path) -> None:
        super().__init__(inf, model_path)
        self.session = ort.InferenceSession(model_path, providers=[ 'CUDAExecutionProvider'])

        self.face_mesh_input_name = [
        input.name for input in self.session.get_inputs()
        ]
        self.face_mesh_output_names = [
            output.name for output in self.session.get_outputs()
        ]
        
    def infer(self, face_img):

        self._t['forward_pass'].tic()



        scores, final_landmarks = self.session.run(
        output_names =self.face_mesh_output_names,
        input_feed = {
            self.face_mesh_input_name[0]: face_img,
            self.face_mesh_input_name[1]: np.array([[0]], dtype=np.int32),
            self.face_mesh_input_name[2]: np.array([[0]], dtype=np.int32),
            self.face_mesh_input_name[3]: np.array([[192]], dtype=np.int32),
            self.face_mesh_input_name[4]: np.array([[192]], dtype=np.int32),
        }
        )

        self._t['forward_pass'].toc()

        return scores, final_landmarks 



"""
Description: The core file of this library.

Author: Tomas Goldmann
Date Created: Dec 26, 2023
Date Modified: Dec 26, 2023
License: MIT License
"""

import cv2
import numpy as np
import os
import onnxruntime as ort
import confuse
import gdown
from urllib.request import urlretrieve

from pathlib import Path
from abc import ABCMeta, abstractmethod

from .modules.retinaface_detector.RetinaFaceDetectorFactory import FaceDetectorFactory
from .core.embedding_metrics import EmbeddingMetrics
from .core.align_trans import  warp_and_crop_face
from .core.transforms import Transforms
from .common.utils import load_yaml
from .modules.recognition.RecognitionFactory import RecognitionFactory
from .modules.n19_landmarks import N19LandmarksFactory
from .modules.n19_rotation.N19Rotation import N19Rotation
from .modules.mediapipe_landmarks import MediapipeLandmarksFactory

config = confuse.Configuration('FaceAIKit', __name__)
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'base.yaml' )
if not 'lib' in config:
    config.set_file(DEFAULT_CONFIG_PATH)

class FaceRecognition:

    def __init__(self, recognition='arcface', config_file=None) -> None:

        if config_file != None:
            if os.path.exists(config_file):
                config.set_file(config_file, base_for_paths=True)
            else:
                raise RuntimeError("Config files does not exist!")
 
        script_dir = os.path.dirname(os.path.abspath(__file__))

        self.home = str(os.getenv("FACEAIKITDIR", default=str(Path.home())))
        self.model_folder = os.path.join(self.home, 'faceaikit')

        self.check_and_download_models(self.model_folder, config, recognition)

        try:
            model_path = os.path.join(self.model_folder,'recognition',config['recognition_'+recognition]['model'].get())
            self.recg = RecognitionFactory.create(recognition, config['recognition_'+recognition]['provider'].get(), model_path)
        except:
            raise Exception("Face recognition library failed!")

        try:
            model_path = os.path.join(self.model_folder,'detector',config['retinaface_detector']['model'].get())
            self.det = FaceDetectorFactory.create_detector(config['retinaface_detector']['provider'].get(), model_path, config['retinaface_detector'].get())
        except:
            raise Exception("RetinaFace factory failed!")
        try:
            model_path = os.path.join(self.model_folder,'landmarks',config['landmarks']['model'].get())
            if config['landmarks']['module'].get()=='n19':
                self.keypoints = N19LandmarksFactory.create('n19', config['landmarks']['provider'].get(), model_path )
            elif config['landmarks']['module'].get()=='mediapipe':
                self.keypoints = MediapipeLandmarksFactory.create('mediapipe', config['landmarks']['provider'].get(), model_path )

        except:
            raise Exception("Factory of facial landmark detector failed!")

        self._rotation = N19Rotation()
        
        #keypoints = max(results, key=lambda x: x['score'])['keypoints']

    def check_and_download_models(self, model_folder, config, recognition):
        #face detection
        url = config['lib']['model_url'].get()

        face_detection_model = os.path.join(model_folder,'detector','resnet_dynamic.onnx' )
        face_recognition_model = os.path.join(model_folder,'recognition',config['recognition_'+recognition]['model'].get() )
        face_landmark_model = os.path.join(model_folder,'landmarks',config['landmarks']['model'].get() )

        if os.path.isfile(face_detection_model) != True:
            os.makedirs(os.path.join(model_folder,'detector'), exist_ok=True)
            #urlretrieve(url+ '/detector/' + config['retinaface_detector']['model'].get(), face_detection_model)
            gdown.download(url+ '/' + config['retinaface_detector']['model'].get(), face_detection_model, quiet=False)

        if os.path.isfile(face_recognition_model) != True:
            os.makedirs(os.path.join(model_folder,'recognition'), exist_ok=True)
            #urlretrieve(url+ '/recognition/' + config['recognition_'+recognition]['model'].get(), face_recognition_model)

            gdown.download(url+ '/' + config['recognition_'+recognition]['model'].get(), face_recognition_model, quiet=False)

        if os.path.isfile(face_landmark_model) != True:
            os.makedirs(os.path.join(model_folder,'landmarks'), exist_ok=True)
            #urlretrieve(url+ '/landmarks/' + config['landmarks']['model'].get(), face_landmark_model)

            gdown.download(url+ '/' + config['landmarks']['model'].get(), face_landmark_model, quiet=False)


    def landmarks(self, face_image1, face1_roi,):
        """
        Extracts facial landmarks from a specified region of interest (ROI) in an input face image.

        Parameters:
        - face_image1 (numpy.ndarray): The input face image.
        - face1_roi (tuple): A tuple representing the coordinates of the ROI in the format ((top-left-x, top-left-y), (bottom-right-x, bottom-right-y)).

        Returns:
        - landmarks (numpy.ndarray): An array containing the coordinates of facial landmarks after processing.
        """
        face_image1=face_image1[face1_roi[0][1]:face1_roi[1][1],face1_roi[0][0]:face1_roi[1][0]]
        face_image1, pad_x, pad_y = Transforms.add_square_padding(current_img = face_image1)

        landmarks = self.keypoints.inference(face_image1)
        landmarks = landmarks + np.array([pad_x, pad_y,0])

        return landmarks


    def face_detection(self,face_image1, align='keypoints' ):
        results, _ = self.det.detect(face_image1, align)
        return results

    def calculate_distance(self,emb1, emb2, distance_metric):
        """
        Calculates the distance between two facial embeddings using the specified distance metric.

        Parameters:
        - emb1 (numpy.ndarray): The first facial embedding.
        - emb2 (numpy.ndarray): The second facial embedding.
        - distance_metric (str): The distance metric to be used, currently supporting 'euclidean_l2' and 'euclidean'.

        Returns:
        - distance (float): The calculated distance between the two embeddings.
        
        Raises:
        - Exception: If an unknown distance metric is provided.
        """


        if distance_metric=='euclidean_l2':
            distance = EmbeddingMetrics.EuclideanDistanceL2(emb1, emb2)
        else:
            raise Exception("Unkown distance matric. Use: euclidean_l2, euclidean")

        return distance

    def verify_rois(self, face_image1, face1_roi, face_image2, face2_roi,  distance_metric="euclidean_l2"):
        """
        Verifies the similarity between two facial regions of interest (ROIs) using facial embeddings and a specified distance metric.

        Parameters:
        - face_image1 (numpy.ndarray): The input face image containing the first ROI.
        - face1_roi (tuple): A tuple representing the coordinates of the first ROI in the format ((top-left-x, top-left-y), (bottom-right-x, bottom-right-y)).
        - face_image2 (numpy.ndarray): The input face image containing the second ROI.
        - face2_roi (tuple): A tuple representing the coordinates of the second ROI in the format ((top-left-x, top-left-y), (bottom-right-x, bottom-right-y)).
        - distance_metric (str): The distance metric to be used, currently supporting 'euclidean_l2' and 'euclidean'.

        Returns:
        - distance (float): The calculated distance between the facial embeddings of the two ROIs.
        """
        face1=face_image1[face1_roi[0][1]:face1_roi[1][1],face1_roi[0][0]:face1_roi[1][0]]
        face2=face_image2[face2_roi[0][1]:face2_roi[1][1],face2_roi[0][0]:face2_roi[1][0]]

        face1,_,_ = Transforms.add_square_padding(current_img = face1)
        face2,_,_ = Transforms.add_square_padding(current_img = face2)

        emb1 = self.recg.inference(face1)
        emb2 = self.recg.inference(face2)

        #print(emb1, emb2)
        distance = self.calculate_distance(emb1, emb2, distance_metric)
        return distance

    def convert_rois_to_faces(self,face_batch_record):
        """
        Converts a batch of facial region of interest (ROI) records to a list of standardized face images.

        Parameters:
        - face_batch_record (list): A list containing tuples, each representing a facial ROI record with the format (image, roi).
                                The image is the input face image, and roi is a tuple representing the ROI coordinates.

        Returns:
        - faces (list): A list of standardized face images, each resized to the required recognition size and in RGB format.
        """
        faces = list()
        for item in face_batch_record:
            image= item[0]
            roi = item[1]
            face = image[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]]
            if face.shape[0] != face.shape[1]:

                face,_,_ = Transforms.add_square_padding(current_img = face)

            image = cv2.resize(face, (self.recg.size, self.recg.size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            faces.append(image)
        return faces

    def verify_batch(self, face_batch1, face_batch2,  distance_metric="euclidean_l2"):
        faces1 = self.convert_rois_to_faces(face_batch1)
        faces2 = self.convert_rois_to_faces(face_batch2)

        embs1 = self.recognition.inference_batch(faces1)
        embs2 = self.recognition.inference_batch(faces2)

        distances = EmbeddingMetrics.EuclideanDistanceL2(embs1, embs2)

        return distances

    def verify(self, face_image1, face_image2,  distance_metric="euclidean_l2"):
        """
        Verifies the similarity between two face image using facial embeddings and a specified distance metric.

        Parameters:
        - face_image1 (numpy.ndarray): The input face image .
        - face_image2 (numpy.ndarray): The input face image .
        - distance_metric (str): The distance metric to be used, currently supporting 'euclidean_l2' and 'euclidean'.

        Returns:
        - distance (float): The calculated distance between the facial embeddings of the two ROIs.
        """
        if face_image1.shape[0] != face_image1.shape[1]:
            face_image1,_,_ = Transforms.add_square_padding(current_img = face_image1)
        if face_image2.shape[0] != face_image2.shape[1]:
            face_image2,_,_ = Transforms.add_square_padding(current_img = face_image2)

        emb1 = self.recg.inference(face_image1)
        emb2 = self.recg.inference(face_image2)

        #print(emb1, emb2)
        distance = self.calculate_distance(emb1, emb2, distance_metric)
        return distance


    def represent(self, face_image):
        if face_image.shape[0] != face_image.shape[1]:
            face_image,_,_ = Transforms.add_square_padding(current_img = face_image)
        
        return self.recg.inference(face_image)



    def rotation(self, face_img, roi=None, kpts=None):
        """
        Applies rotation adjustments to a face image based on facial landmarks within a specified region of interest (ROI).

        Parameters:
        - face_img (numpy.ndarray): The input face image.
        - roi (tuple): A tuple representing the coordinates of the ROI in the format ((top-left-x, top-left-y), (bottom-right-x, bottom-right-y)).
        - kpts (list)

        Returns:
        - None: If the specified facial landmarks module is not 'n19'.
        - Tuple (roll_angle, yaw_angle): Tuple containing the roll and yaw angles if the landmarks module is 'n19'.
        """
        if roi != None:
            if config['landmarks']['module'].get()=='n19':
                kpts = self.keypoints.inference(face_img[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]])
                print(kpts.shape)

                if len(kpts)==98:
                    right_eye_corner = kpts[72] 
                    left_eye_corner = kpts[60]

                    roll_angle, roll_M, eyesCenter = self._rotation.get_roll(face_img,left_eye_corner,right_eye_corner)
                    kpts_roll_norm = cv2.transform(np.array([kpts]), roll_M)[0]
                    
                    nose_tip_pt = kpts_roll_norm[51]
                    lf_breadth_pt = kpts_roll_norm[0]
                    rg_breadth_pt = kpts_roll_norm[32]

                    yaw_angle, M = self._rotation.get_yaw(nose_tip_pt, lf_breadth_pt, rg_breadth_pt)

                    return yaw_angle, roll_angle
                else:
                    raise Exception("The nb of kpts must be 98 included in list, that are produced by n19 face landmark detector")
            
            elif config['landmarks']['module'].get()=='mediapipe':
                kpts = self.keypoints.inference(face_img[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]])

                if len(kpts)==98:
                    right_eye_corner = kpts[72] 
                    left_eye_corner = kpts[60]

                    roll_angle, roll_M, eyesCenter = self._rotation.get_roll(face_img,left_eye_corner,right_eye_corner)
                    kpts_roll_norm = cv2.transform(np.array([kpts]), roll_M)[0]
                    
                    nose_tip_pt = kpts_roll_norm[51]
                    lf_breadth_pt = kpts_roll_norm[0]
                    rg_breadth_pt = kpts_roll_norm[32]

                    yaw_angle, M = self._rotation.get_yaw(nose_tip_pt, lf_breadth_pt, rg_breadth_pt)

                    return yaw_angle, roll_angle
                else:
                    raise Exception("The nb of kpts must be 98 included in list, that are produced by n19 face landmark detector")
            
            else:
                raise RuntimeError("Appropriate landmarks detector does not found")
        if kpts != None:
            if len(kpts)!=5:
                raise Exception("The nb of kpts must be 5 included in list, that are produced by RetinaFace detector")

            roll_angle, roll_M, eyesCenter = self._rotation.get_roll(face_img,kpts[0],kpts[1])
            kpts_roll_norm = cv2.transform(np.array([kpts]), roll_M)[0]
            
            nose_tip_pt = kpts[2]
            lf_breadth_pt = kpts[3]
            rg_breadth_pt = kpts[4]

            yaw_angle, M = self._rotation.get_yaw_5pt(nose_tip_pt, lf_breadth_pt, rg_breadth_pt)

            return yaw_angle, roll_angle
            
        raise RuntimeError("Rotation need face ROI or face keypoints")
        



        


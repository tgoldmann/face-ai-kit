"""
Description: Module for calculate head rotation based on landmarks 

Author: Tomas Goldmann
Date Created: Dec 26, 2023
Date Modified: Dec 26, 2023
License: MIT License
"""

import onnxruntime as ort
import numpy as np
import cv2

from abc import ABCMeta, abstractmethod

from ...core.timer import Timer


import numpy as np
import cv2
import math


class N19Rotation:
    HEAD_BREADTH_MALE = 151.5
    BIOCULAR_BREADTH_MALE = 65.6

    def __init__(self, desiredFaceWidth=256,desiredLeftEye=(0.25,0.25)):
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceWidth
        self.desiredLeftEye = desiredLeftEye

    #https://pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
    def get_roll(self, image, left_eye_pt,right_eye_pt):
        dY = right_eye_pt[1] - left_eye_pt[1]
        dX = right_eye_pt[0] - left_eye_pt[0]
        angle = np.arctan2(dY, dX)
        #print(angle, dX, dY)
        eyesCenter = ((left_eye_pt[0] + right_eye_pt[0]) // 2, (left_eye_pt[1] + right_eye_pt[1]) // 2)
        #print(eyesCenter, angle,scale)
        M = cv2.getRotationMatrix2D((float(eyesCenter[0]),float(eyesCenter[1])), float(angle), 1.0)
        return angle, M, eyesCenter

    def norm_roll(self,image, left_eye_pt,right_eye_pt, lf_breadth_pt, rg_breadth_pt, img=None, kpts = None):

        if  img is None:
            dY = right_eye_pt[1] - left_eye_pt[1]
            dX = right_eye_pt[0] - left_eye_pt[0]
            angle = np.degrees(np.arctan2(dY, dX))
            eyesCenter = ((left_eye_pt[0] + right_eye_pt[0]) // 2, (left_eye_pt[1] + right_eye_pt[1]) // 2)
            

            # grab the rotation matrix for rotating and scaling the face
            desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

            dist = np.sqrt((dX ** 2) + (dY ** 2))
            desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
            desiredDist *= self.desiredFaceWidth
            scale = desiredDist / dist

            M = cv2.getRotationMatrix2D((float(eyesCenter[0]),float(eyesCenter[1])), float(angle), float(scale))
            tX = self.desiredFaceWidth * 0.5
            tY = self.desiredFaceHeight * self.desiredLeftEye[1]
            M[0, 2] += (tX - eyesCenter[0])
            M[1, 2] += (tY - eyesCenter[1])

            (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
            output = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC)

            output_all = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC)
        else:

            #print(left_eye_pt[0], left_eye_pt[1],right_eye_pt[0], right_eye_pt[1] )
            dY = right_eye_pt[1] - left_eye_pt[1]
            dX = right_eye_pt[0] - left_eye_pt[0]
            angle = np.degrees(np.arctan2(dY, dX))
            eyesCenter = ((left_eye_pt[0] + right_eye_pt[0]) // 2, (left_eye_pt[1] + right_eye_pt[1]) // 2)

            center_x = (rg_breadth_pt[0] + lf_breadth_pt[0])/2


            desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

            dist = np.sqrt((dX ** 2) + (dY ** 2))
            desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
            desiredDist *= self.desiredFaceWidth
            scale = desiredDist / dist

            M = cv2.getRotationMatrix2D((float(eyesCenter[0]),float(eyesCenter[1])), float(angle), float(scale))
            tX = self.desiredFaceWidth * 0.5
            tY = self.desiredFaceHeight * self.desiredLeftEye[1]
            #M[0, 2] += (tX - eyesCenter[0])
            M[0, 2] += (tX - center_x)

            M[1, 2] += (tY - eyesCenter[1])

            (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
            output = cv2.warpAffine(img, M, (w, h),flags=cv2.INTER_CUBIC)


        #print(kpts)
        if kpts is not None:
            if len(kpts.shape)== 1:
                kpts =np.array(kpts).reshape(kpts.shape[0]/2,2)
            elif  len(kpts.shape)== 3:
                raise Exception

            kpts = cv2.transform(np.array([kpts]), M)[0]


        return output, kpts


    #https://www.sciencedirect.com/science/article/pii/S0360835218300305
    #for WFLW 98
    def get_yaw(self,nose_tip_pt, lf_breadth_pt, rg_breadth_pt):

        breadth_px = rg_breadth_pt[0]-lf_breadth_pt[0]
        rotation_center = lf_breadth_pt[0] + breadth_px/2.0
        CONS_r_breadth_r_eye = (93.3)/self.HEAD_BREADTH_MALE

        print((rotation_center-nose_tip_pt[0]),  (breadth_px/2))
        center_deviation_norm = (rotation_center-nose_tip_pt[0]) / (breadth_px/2) 
        angle =  np.degrees(np.arcsin(center_deviation_norm))

        #print("yaw",eyesCenter, angle,scale)
        #identity matrix
        M = np.zeros((3,3))
        M[0,0] = 1.0
        M[1,1] = 1.0
        M[2,2] = 1.0

        M[0,0] = np.cos(angle)
        M[2,2] = np.cos(angle)
        M[2,0] = - np.sin(angle)
        M[0,2] =  np.sin(angle)

        return angle, M
    
    #for 5 points YOLO
    def get_yaw_5pt(self,nose_tip_pt, lf_eye_pt, rg_eye_pt):

        biocular_breadth = rg_eye_pt[0]-lf_eye_pt[0]
        rotation_center = lf_eye_pt[0] + biocular_breadth/2.0
        coef_biocular_to_head_breadtg = self.HEAD_BREADTH_MALE/self.BIOCULAR_BREADTH_MALE

        center_deviation_norm = (rotation_center-nose_tip_pt[0]) / (biocular_breadth*coef_biocular_to_head_breadtg/2) 

        angle =  np.arcsin(center_deviation_norm)

        #print(eyesCenter, angle,scale)
        #identity matrix
        M = np.zeros((3,3))
        M[0,0] = 1.0
        M[1,1] = 1.0
        M[2,2] = 1.0

        M[0,0] = np.cos(angle)
        M[2,2] = np.cos(angle)
        M[2,0] = - np.sin(angle)
        M[0,2] =  np.sin(angle)

        return angle, M

    def get_pitch_5pt(self,nose_tip_pt, lf_eye_pt, rg_eye_pt):
        return 0.0


    def get_pitch(self,img, align_points):
        pt=align_points
        font = cv2.FONT_HERSHEY_SIMPLEX

        CONS_breadth_to_t_gn = (151.5/153)
        breadth_px = pt[32][0] - pt[0][0] 

        radius_px = breadth_px*CONS_breadth_to_t_gn

        midpupil_right= [0,0]
        midpupil_left= [0,0]

        midpupil_right[0] = int(pt[60][0] + (pt[64][0] - pt[60][0])/2.0)
        midpupil_right[1] = int(pt[60][1] + (pt[64][1] - pt[60][1])/2.0)
        midpupil_left[0] = int(pt[68][0] + (pt[72][0] - pt[68][0])/2.0)
        midpupil_left[1] = int(pt[68][1] + (pt[72][1] - pt[68][1])/2.0)


        right_eye_width =  np.linalg.norm((pt[64])- (pt[60]))
        left_eye_width = np.linalg.norm((pt[68])- (pt[72]))

        eye_width = 24.2 #https://www.hindawi.com/journals/joph/2014/503645/
        orbital_height = 36.97 #https://www.researchgate.net/figure/Comparison-of-Parameters-According-to-Gender_tbl2_309165710

        orbital_height_in_image_scale_px = orbital_height * (((left_eye_width +right_eye_width)/2.0)/eye_width)


        left_orbit_bottom = midpupil_left
        left_orbit_bottom[1] = int(left_orbit_bottom[1] + (orbital_height_in_image_scale_px/2))


        right_orbit_bottom = midpupil_right
        right_orbit_bottom[1] = int(right_orbit_bottom[1] + (orbital_height_in_image_scale_px/2))

        y_diff_right = right_orbit_bottom[1] - pt[2][1]
        y_diff_left = right_orbit_bottom[1] - pt[30][1]

        SELLION_TO_TRAGION = 93.3

        radius = SELLION_TO_TRAGION

        angle = np.arctan(y_diff_left/radius)


        for p in range(len(pt)):
            cv2.circle(img, (int(pt[p][0]), int(pt[p][1])), 1, (0,0,255), 2)
            #cv2.putText(img, str(p), (int(pt[p][0]), int(pt[p][1])), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        print(left_orbit_bottom)
        cv2.circle(img, tuple(left_orbit_bottom), 1, (255,0,255), 2)
        cv2.circle(img, tuple(right_orbit_bottom), 1, (255,0,255), 2)

        cv2.imshow("Rot. estimation",img)
        cv2.waitKey(0)

        return angle


"""
Created on Mon Apr 24 15:43:29 2017
@author: zhaoy, modified Tomas Goldmann

"""

import numpy as np
import cv2
#from .matlab_cp2tform import get_similarity_transform_for_cv2
from skimage.transform import SimilarityTransform


# reference facial points, a list of coordinates (x,y)
REFERENCE_FACIAL_POINTS = [        # default reference facial points for crop_size = (112, 112); should adjust REFERENCE_FACIAL_POINTS accordingly for other crop_size
    [38.411846, 52.59001],
    [73.68209, 52.300644],
    [56.092415, 72.949585],
    [40.763634, 90.94648],
    [71.64599, 90.62956]
]

DEFAULT_CROP_SIZE = (112, 112)

class FaceWarpException(Exception):
    def __str__(self):
        return 'In File {}:{}'.format(
            __file__, super.__str__(self))

def warp_and_crop_face(src_img,
                       facial_pts,
                       reference_pts = None,
                       crop_size=(96, 112),
                       align_type = 'similarity'):
    """
    Function:
    ----------
        apply affine transform 'trans' to uv
    Parameters:
    ----------
        @src_img: 3x3 np.array
            input image
        @facial_pts: could be
            1)a list of K coordinates (x,y)
        or
            2) Kx2 or 2xK np.array
            each row or col is a pair of coordinates (x, y)
        @reference_pts: could be
            1) a list of K coordinates (x,y)
        or
            2) Kx2 or 2xK np.array
            each row or col is a pair of coordinates (x, y)
        or
            3) None
            if None, use default reference facial points
        @crop_size: (w, h)
            output face image size
        @align_type: transform type, could be one of
            1) 'similarity': use similarity transform
            2) 'cv2_affine': use the first 3 points to do affine transform,
                    by calling cv2.getAffineTransform()
            3) 'affine': use all points to do affine transform
    Returns:
    ----------
        @face_img: output face image with size (w, h) = @crop_size
    """

    reference_pts = REFERENCE_FACIAL_POINTS

    ref_pts = np.float32(reference_pts)
    ref_pts_shp = ref_pts.shape
    if max(ref_pts_shp) < 3 or min(ref_pts_shp) != 2:
        raise FaceWarpException(
            'reference_pts.shape must be (K,2) or (2,K) and K>2')

    if ref_pts_shp[0] == 2:
        ref_pts = ref_pts.T

    src_pts = np.float32(facial_pts)
    src_pts_shp = src_pts.shape
    if max(src_pts_shp) < 3 or min(src_pts_shp) != 2:
        raise FaceWarpException(
            'facial_pts.shape must be (K,2) or (2,K) and K>2')

    if src_pts_shp[0] == 2:
        src_pts = src_pts.T

#    #print('--->src_pts:\n', src_pts
#    #print('--->ref_pts\n', ref_pts

    if src_pts.shape != ref_pts.shape:
        raise FaceWarpException(
            'facial_pts and reference_pts must have the same shape')

    if align_type is 'cv2_affine':
        tfm = cv2.getAffineTransform(src_pts[0:3], ref_pts[0:3])
#        #print(('cv2.getAffineTransform() returns tfm=\n' + str(tfm))
    elif align_type is 'affine':
        tfm = get_affine_transform_matrix(src_pts, ref_pts)
#        #print(('get_affine_transform_matrix() returns tfm=\n' + str(tfm))
    else:
        #tfm = get_similarity_transform_for_cv2(src_pts, ref_pts)
#        #print(('get_similarity_transform_for_cv2() returns tfm=\n' + str(tfm))
        tform = SimilarityTransform()

        tform.estimate(src_pts, ref_pts)
        tfm = tform.params
        tfm_inv = np.linalg.inv(tfm)

        dest_corners = np.array([[0, 0,1], [112, 0,1], [112, 112,1], [0, 112,1]], dtype=np.float32)

        source_roi = cv2.transform(dest_corners.reshape(1, -1, 3), tfm_inv)
        min_x = int(np.min(source_roi[:, :, 0]))
        max_x = int(np.max(source_roi[:, :, 0]))
        min_y = int(np.min(source_roi[:, :, 1]))
        max_y = int(np.max(source_roi[:, :, 1]))



        # Calculate the center of the square
        center_x = int((min_x + max_x) / 2)
        center_y = int((min_y + max_y) / 2)

        # Determine the size of the square (assuming square, so just one side length)
        side_length = int(max(max_x - min_x, max_y - min_y))

        square_x1 = max(0, center_x - side_length // 2)
        square_x2 = min(src_img.shape[1], center_x + side_length // 2)
        square_y1 = max(0, center_y - side_length // 2)
        square_y2 = min(src_img.shape[0], center_y + side_length // 2)


        # Define the coordinates of the square as integer values
        source_roi = np.array([[square_x1, square_y1],
                            [square_x2, square_y1],
                            [square_x2, square_y2],
                            [square_x1, square_y2]])
        #print(source_roi)
                

#    #print('--->Transform matrix: '
#    #print(('type(tfm):' + str(type(tfm)))
#    #print(('tfm.dtype:' + str(tfm.dtype))
#    #print( tfm
    tfm = tform.params[0:2, :]
    face_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]))

    return face_img, source_roi
"""
Description: Functions for calculating distance between vectors

Author: Tomas Goldmann
Date Created: Dec 26, 2023
Date Modified: Dec 26, 2023
License: MIT License
"""


import numpy as np


class EmbeddingMetrics:

    @staticmethod
    def CosineDistance(source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    @staticmethod
    def EuclideanDistance(source_representation, test_representation):
        if isinstance(source_representation, list):
            source_representation = np.array(source_representation)

        if isinstance(test_representation, list):
            test_representation = np.array(test_representation)

        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance

    @staticmethod
    def l2_normalize(x):
        return x / np.sqrt(np.sum(np.multiply(x, x)))

    @staticmethod
    def EuclideanDistanceL2(source_representation, test_representation):
        if isinstance(source_representation, list):
            source_representation = np.array(source_representation)

        if isinstance(test_representation, list):
            test_representation = np.array(test_representation)

        euclidean_distance = EmbeddingMetrics.l2_normalize(source_representation) - EmbeddingMetrics.l2_normalize(test_representation) 
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)

        return euclidean_distance
from math import atan2, sqrt, pi

import cv2
import numpy as np


class SolvePnpMethod:
    def __init__(self):
        pass

    rad_to_deg = 57.2958

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    @staticmethod
    def get_matrix_angle(m):
        rad_to_deg = 57.2958
        x = atan2(m[2][1], m[2][2])
        y = atan2(-m[2][0], sqrt(pow(m[2][1], 2) + pow(m[2][2], 2)))
        z = atan2(m[1][0], m[0][0])
        return (pi - x) * rad_to_deg, y * rad_to_deg, z * rad_to_deg

    @staticmethod
    def get_angles(image, points):
        size = image.shape

        # Camera internals

        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        # 2D image points. If you change the image, you need to change vector
        image_points = np.array([
            (points[30].x, points[30].y),  # Nose tip
            (points[8].x, points[8].y),  # Chin
            (points[36].x, points[36].y),  # Left eye left corner
            (points[45].x, points[45].y),  # Right eye right corne
            (points[60].x, points[60].y),  # Left Mouth corner
            (points[64].x, points[64].y)  # Right mouth corner
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(SolvePnpMethod.model_points,
                                                                      image_points, camera_matrix,
                                                                      dist_coeffs,
                                                                      flags=cv2.SOLVEPNP_ITERATIVE)

        matrix = cv2.Rodrigues(rotation_vector)
        rotation_vector *= SolvePnpMethod.rad_to_deg
        rotation_vector[0] += np.pi
        return SolvePnpMethod.get_matrix_angle(matrix[0])

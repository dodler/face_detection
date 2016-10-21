from math import atan2, sqrt, atan

import cv2
import dlib
import numpy as np


class FaceAngleDetector:
    predictor_path = '/home/lyan/PycharmProjects/face_detection/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    METHOD_SOLVE_PNP = 1
    METHOD_SOLVE_CUSTOM_POINTS = 2

    def __init__(self, method):
        self.method = method

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    rad_to_deg = 57.2958

    eyes_angle_ratio = 167.0
    nose_angle_ratio = 0.738

    # Camera internals


    def get_matrix_angle(m):
        x = atan2(m[2][1], m[2][2])
        y = atan2(-m[2][0], sqrt(pow(m[2][1], 2) + pow(m[2][2], 2)))
        z = atan2(m[1][0], m[0][0])
        return x, y, z

    def get_face_angle_solve_pnp(self, image, points):
        size = image.shape

        # Camera internals

        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(FaceAngleDetector.model_points,
                                                                      points, camera_matrix,
                                                                      dist_coeffs,
                                                                      flags=cv2.SOLVEPNP_ITERATIVE)

        matrix = cv2.Rodrigues(rotation_vector)
        return self.get_matrix_angle(matrix[0])

    def get_angle_by_2_points(point1, point2):
        width = abs(point1.x - point2.x)
        height = abs(point1.x - point2.y)
        return FaceAngleDetector.rad_to_deg * (width / (1.0 * height))

    def get_line_coef_by_points_coords(x1, y1, x2, y2):
        a = y2 - y1
        b = x2 - x1
        c = x1 * y2 - x2 * y1

        return (a, b, c)

    def get_intersection_coords(a1, b1, c1, a2, b2, c2):
        if a1 * b2 - a2 * b1 == 0:
            return 0, 0  # means that lines are parallel

        x = 1.0 * (c2 * b1 - c1 * b2) / (a1 * b2 - a2 * b1)
        y = 1.0 * (a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)
        return x, y

    def get_line_len_by_2_points(x1, y1, x2, y2):
        return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))

    def rotation_angle(left_eye_1,
                       left_eye_2,
                       right_eye_1,
                       right_eye_2,
                       lips_left,
                       lips_right):
        left_eye_angle = FaceAngleDetector.get_angle_by_2_points(left_eye_1, left_eye_2)
        right_eye_angle = FaceAngleDetector.get_angle_by_2_points(right_eye_1, right_eye_2)
        lips_angle = FaceAngleDetector.get_angle_by_2_points(lips_left, lips_right)

        return sum([left_eye_angle, right_eye_angle, lips_angle]) / 3.0

    def nose_ratio(nose_left_x, nose_left_y,
                   nose_right_x, nose_right_y,
                   nose_top_x, nose_top_y,
                   nose_bot_x, nose_bot_y):
        (a1, b1, c1) = FaceAngleDetector.get_line_coef_by_points_coords(nose_left_x, nose_left_y,
                                                                        nose_right_x, nose_right_y)

        (a2, b2, c2) = FaceAngleDetector.get_line_coef_by_points_coords(nose_top_x, nose_top_y,
                                                                        nose_bot_x, nose_bot_y)

        (x, y) = (FaceAngleDetector.get_intersection_coords(a1, b1, c1, a2, b2, c2))
        x = abs(x)
        y = abs(y)

        nose_top = FaceAngleDetector.get_line_len_by_2_points(nose_top_x, nose_top_y,
                                                              x, y)
        nose_bot = FaceAngleDetector.get_line_len_by_2_points(nose_bot_x, nose_bot_y,
                                                              x, y)

        return nose_top / nose_bot

    def get_matrix_angle(m):
        x = atan2(m[2][1], m[2][2])
        y = atan2(-m[2][0], sqrt(pow(m[2][1], 2) + pow(m[2][2], 2)))
        z = atan2(m[1][0], m[0][0])
        return x, y, z

    # this method returns face angles according to
    # special custom facial ratios

    def face_angle(self, shape_parts):

        eye_left_left_p_x = shape_parts[36].x
        eye_left_left_p_y = shape_parts[36].y

        eye_left_right_p_x = shape_parts[39].x
        eye_left_right_p_y = shape_parts[39].y

        eye_right_left_p_x = shape_parts[42].x
        eye_right_left_p_y = shape_parts[42].y

        eye_right_right_p_x = shape_parts[45].x
        eye_right_right_p_y = shape_parts[45].y

        left_eye_width = abs(eye_left_left_p_x - eye_left_right_p_x) + 0.01
        left_eye_height = abs(eye_left_left_p_y - eye_left_right_p_y) + 0.01
        right_eye_width = abs(eye_right_left_p_x - eye_right_right_p_x) + 0.01

        nose_top_y = shape_parts[30].y

        nose_left_y = shape_parts[31].y

        nose_right_y = shape_parts[35].y

        nose_bot_y = shape_parts[34].y

        nose_horizontal_mean_y = (nose_left_y + nose_right_y) / 2.0
        nose_top_size = abs(nose_top_y - nose_horizontal_mean_y) + 0.01  # should be positive but this is safer
        nose_bot_size = abs(nose_bot_y - nose_horizontal_mean_y) + 0.01

        nose_angle = FaceAngleDetector.nose_angle_ratio * (4.41 - nose_top_size / nose_bot_size)
        eye_angle = (1.0 - (left_eye_width / (1.0 * right_eye_width))) * FaceAngleDetector.eyes_angle_ratio
        rotation = FaceAngleDetector.rad_to_deg * atan(left_eye_height / (1.0 * left_eye_width))

        return nose_angle, eye_angle, rotation

    BASE_WIDTH = 320.0 # todo implement in more fast manner

    def get_face_angle(self, image):

        dets = FaceAngleDetector.detector(image, 1)
        result = []

        width = 300

        for d in dets:
            shape = FaceAngleDetector.predictor(image, d)

            if self.method == FaceAngleDetector.METHOD_SOLVE_PNP:
                result.append(self.get_face_angle_solve_pnp(image, shape.parts()))
            elif self.method == FaceAngleDetector.METHOD_SOLVE_CUSTOM_POINTS:
                result.append(self.face_angle(shape.parts()))

        return result

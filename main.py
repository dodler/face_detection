import sys
import os

import cv2
import dlib
import glob
from skimage import io
from math import pi, sqrt
from math import atan

predictor_path = '/home/lyan/PycharmProjects/face_detection/shape_predictor_68_face_landmarks.dat'
faces_folder_path = '/home/lyan/Documents/work/data/faces/test2/'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()

rad_to_deg = 57.2958
eyes_angle_ratio = 167.0
nose_angle_ratio = 0.738


def get_angle_by_2_points(point1, point2):
    width = abs(point1.x - point2.x)
    height = abs(point1.x - point2.y)
    return rad_to_deg * (width / (1.0 * height))


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
    return x,y


def get_line_len_by_2_points(x1, y1, x2, y2):
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))


def rotation_angle(left_eye_1,
                   left_eye_2,
                   right_eye_1,
                   right_eye_2,
                   lips_left,
                   lips_right):
    left_eye_angle = get_angle_by_2_points(left_eye_1, left_eye_2)
    right_eye_angle = get_angle_by_2_points(right_eye_1, right_eye_2)
    lips_angle = get_angle_by_2_points(lips_left, lips_right)

    return sum([left_eye_angle, right_eye_angle, lips_angle]) / 3.0


def nose_ratio(nose_left_x, nose_left_y,
               nose_right_x, nose_right_y,
               nose_top_x, nose_top_y,
               nose_bot_x, nose_bot_y):
    (a1, b1, c1) = get_line_coef_by_points_coords(nose_left_x, nose_left_y,
                                                  nose_right_x, nose_right_y)

    (a2, b2, c2) = get_line_coef_by_points_coords(nose_top_x, nose_top_y,
                                                  nose_bot_x, nose_bot_y)

    (x, y) = (get_intersection_coords(a1, b1, c1, a2, b2, c2))
    x = abs(x)
    y = abs(y)

    nose_top = get_line_len_by_2_points(nose_top_x, nose_top_y,
                                        x, y)
    nose_bot = get_line_len_by_2_points(nose_bot_x, nose_bot_y,
                                        x, y)

    return nose_top/nose_bot


def face_angle(shape_parts, image):
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

    nose_angle = nose_angle_ratio * (4.41 - nose_top_size / nose_bot_size)

    print 'nose angle=', nose_angle

    print 'eye width ratio=', (1.0 - (left_eye_width / (1.0 * right_eye_width))) * eyes_angle_ratio
    # aka face rotation

    print 'eye angle=', rad_to_deg * atan(left_eye_height / (1.0 * left_eye_width))


for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = io.imread(f)

    win.clear_overlay()
    win.set_image(img)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make ever
    # ything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print dets
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)

        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))

        # print face_angle(shape_parts=shape.parts(), image=img)
        print 'rotation_angle', rotation_angle(shape.parts()[36], shape.parts()[39],
                                               shape.parts()[42], shape.parts()[45],
                                               shape.parts()[60], shape.parts()[64])

        parts = shape.parts()
        ratio = nose_ratio(parts[31].x, parts[31].y,
                          parts[35].x, parts[35].y,
                          parts[30].x, parts[30].y,
                          parts[34].x, parts[34].y)
        print 'nose ratio:', 0.89 * (3.56 - ratio)


        i = 0
        for part in shape.parts():
            i += 1
            if part.x < 260 and part.x > 250 and part.y > 420 and part.y < 430:
                print i
                print part

        # Draw the face landmarks on the screen.
        win.add_overlay(shape)

    win.add_overlay(dets)
    dlib.hit_enter_to_continue()

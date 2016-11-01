#!/usr/bin/env python
# coding=utf-8

import dlib
import numpy as np

path = '/home/lyan/Documents/work/data/faces/test2/'
predictor_path = '/home/lyan/PycharmProjects/face_detection/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

rad_to_deg = 57.2958

import cv2
import dlib
from math import atan

predictor_path = '/home/lyan/PycharmProjects/face_detection/shape_predictor_68_face_landmarks.dat'
faces_folder_path = '/home/lyan/Documents/work/data/faces/test3'

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


def rotation_angle(left_eye_1,
                   left_eye_2,
                   right_eye_1,
                   right_eye_2,
                   lips_left,
                   lips_right):
    left_eye_angle = get_angle_by_2_points(left_eye_1, left_eye_2)
    right_eye_angle = get_angle_by_2_points(right_eye_1, right_eye_2)
    lips_angle = get_angle_by_2_points(lips_left, lips_right)

    return sum([left_eye_angle, right_eye_angle, lips_angle])/3.0

def face_angle(shape_parts):

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

    nose_horizontal_mean_y = (nose_left_y + nose_right_y)/2.0
    nose_top_size = abs(nose_top_y - nose_horizontal_mean_y) + 0.01 # should be positive but this is safer
    nose_bot_size = abs(nose_bot_y-nose_horizontal_mean_y) + 0.01

    nose_angle = nose_angle_ratio*(4.41-nose_top_size/nose_bot_size)

    print 'pitch=', nose_angle

    print 'yaw=', (1.0-(left_eye_width / (1.0 * right_eye_width))) * eyes_angle_ratio
    #aka face rotation

    print 'roll=', rad_to_deg*atan(left_eye_height / (1.0 * left_eye_width))


# Read Image

# Camera 0 is the integrated web cam on my netbook
camera_port = 0

# Number of frames to throw away while the camera adjusts to light levels
ramp_frames = 30

# Now we can initialize the camera capture object with the cv2.VideoCapture class.
# All it needs is the index to a camera port.
camera = cv2.VideoCapture(camera_port)


# Captures a single image from the camera and returns it in PIL format
def get_image():
    # read is the easiest way to get a full image out of a VideoCapture object.
    retval, im = camera.read()
    return im


# Ramp the camera - these frames will be discarded and are only used to allow v4l2
# to adjust light levels, if necessary
for i in xrange(ramp_frames):
    temp = get_image()
print("Taking image...")
# Take the actual image we want to keep

# 3D model points.
model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corne
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner

])

while True:
    im = get_image()

    size = im.shape

    dets = detector(im, 1)
    for k, d in enumerate(dets):
        shape = predictor(im, d)

    points = shape.parts()
    # 2D image points. If you change the image, you need to change vector

    face_angle(points)

    image_points = np.array([
        (points[30].x, points[30].y),  # Nose tip
        (points[8].x, points[8].y),  # Chin
        (points[36].x, points[36].y),  # Left eye left corner
        (points[45].x, points[45].y),  # Right eye right corne
        (points[60].x, points[60].y),  # Left Mouth corner
        (points[64].x, points[64].y)  # Right mouth corner
    ], dtype="double")


    for p in image_points:
        cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

    cv2.imshow('Video', im)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

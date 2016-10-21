#!/usr/bin/env python

import cv2
import dlib
import numpy as np
from math import atan, pi, acos, sqrt, atan2, cos, sin

path = '/home/lyan/Documents/work/data/faces/test2/'
predictor_path = '/home/lyan/PycharmProjects/face_detection/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

rad_to_deg = 57.2958


def get_matrix_angle(m):
    x = atan2(m[2][1], m[2][2])
    y = atan2(-m[2][0], sqrt(pow(m[2][1], 2) + pow(m[2][2], 2)))
    z = atan2(m[1][0], m[0][0])
    return x, y, z


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
    image_points = np.array([
        (points[30].x, points[30].y),  # Nose tip
        (points[8].x, points[8].y),  # Chin
        (points[36].x, points[36].y),  # Left eye left corner
        (points[45].x, points[45].y),  # Right eye right corne
        (points[60].x, points[60].y),  # Left Mouth corner
        (points[64].x, points[64].y)  # Right mouth corner
    ], dtype="double")

    # Camera internals

    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    # print "Camera Matrix :\n {0}".format(camera_matrix)

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs,
                                                                  flags=cv2.SOLVEPNP_ITERATIVE)
    # matrix = np.zeros((3, 3), np.float32)
    # cv2.Rodrigues(rotation_vector, matrix)
    matrix = cv2.Rodrigues(rotation_vector)
    # print matrix
    print get_matrix_angle(matrix[0])

    # print "Rotation Vector:\n {0}".format(rotation_vector)
    # print "Translation Vector:\n {0}".format(translation_vector)

    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose


    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                     translation_vector,
                                                     camera_matrix, dist_coeffs)

    for p in image_points:
        cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
        #
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.line(im, p1, p2, (255, 0, 0), 2)
        # print nose_end_point2D
        #
        # Display image
        # Display the resulting frame
    cv2.imshow('Video', im)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

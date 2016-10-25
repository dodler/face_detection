import glob
import os
from skimage import io

import cv2
import dlib
import numpy as np

path = '/home/lyan/Documents/work/data/faces/test2/'
predictor_path = '/home/lyan/PycharmProjects/face_detection/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()

points_path = 'test_image_face_points.txt'

# use only points 27, 5, 11
FACE_POINT_1 = 27
FACE_POINT_2 = 5
FACE_POINT_3 = 11


def init_points(path):
    temp = []
    with open(path) as points_file:
        content = points_file.readlines()
        for line in content:
            if line[0] == '#':
                continue

            tmp = line[1:len(line) - 2]
            x, y = tmp.split(',')
            temp.append((x, y))

    result = np.float32([
        [temp[FACE_POINT_1][0], temp[FACE_POINT_1][1]],
        [temp[FACE_POINT_2][0], temp[FACE_POINT_2][1]],
        [temp[FACE_POINT_3][0], temp[FACE_POINT_3][1]]
    ])
    print result
    return result


points_storage = init_points(points_path)


def fit_face(image, shape):
    img_points = np.float32([
        [shape.parts()[FACE_POINT_1].x, shape.parts()[FACE_POINT_1].y],
        [shape.parts()[FACE_POINT_2].x, shape.parts()[FACE_POINT_2].y],
        [shape.parts()[FACE_POINT_3].x, shape.parts()[FACE_POINT_3].y],
    ])

    affine_transform = cv2.getAffineTransform(img_points, points_storage)
    result = cv2.warpAffine(image, affine_transform, dsize=(96, 96))
    return result


camera_port = 0

# Number of frames to throw away while the camera adjusts to light levels
ramp_frames = 30

# Now we can initialize the camera capture object with the cv2.VideoCapture class.
# All it needs is the index to a camera port.
camera = cv2.VideoCapture(camera_port)

camera.set(3, 320)
camera.set(4, 240)


# Captures a single image from the camera and returns it in PIL format
def get_image():
    # read is the easiest way to get a full image out of a VideoCapture object.
    retval, im = camera.read()
    return im


for i in xrange(ramp_frames):
    temp = get_image()
print("Taking image...")

while True:
    img = get_image()

    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = predictor(img, d)

        face = fit_face(image=img, shape=shape)

        cv2.imshow('Video', face)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

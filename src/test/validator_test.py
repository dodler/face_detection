import os

import cv2

from src.validator import Validator

dir = os.path.dirname(__file__)
filename = os.path.join(dir, '../../data/settings.json')
Validator.init_angle_detector(filename)


camera_port = 0
ramp_frames = 30
camera = cv2.VideoCapture(camera_port)

def get_image():
    # read is the easiest way to get a full image out of a VideoCapture object.
    retval, im = camera.read()
    return im


# Ramp the camera - these frames will be discarded and are only used to allow v4l2
# to adjust light levels, if necessary
for i in xrange(ramp_frames):
    temp = get_image()
print("Taking image...")

while True:
    im = get_image()
    print Validator.validate(im)
    cv2.imshow('result', im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

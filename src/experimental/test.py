# coding=utf-8
import cv2

from src.FaceAngleDetector import FaceAngleDetector

angle_detector = FaceAngleDetector(FaceAngleDetector.METHOD_FACIAL_LANDMARKS)

camera_port = 0
ramp_frames = 30
camera = cv2.VideoCapture(camera_port)


# camera.set(3, 320)
# camera.set(4, 240)


# test_image = cv2.imread('test_face.jpg')
# face_points = angle_detector.get_face_points(test_image)
#
# for face_point in face_points:
#     for face_point_part in face_point.parts():
#         print face_point_part


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


# stands for format into 4 digits
def f4d(num):
    return "%.2f" % num


def prepare_text(inp):

    if len(inp) == 0:
        return ['', '', '']

    pnp_angles = f4d(inp[0][0]) + ' , ' + f4d(inp[0][1]) + ' , ' + f4d(inp[0][2])
    print pnp_angles

    return pnp_angles


# output as follows:
# -------------------------------------------------------------
# поперечная ось -----|вертикальная ось --|-продольлная ось---|
# --------------------|-------------------|-------------------|
# pitch --------------|-yaw --------------|---roll------------|
# --------------------|-------------------|-------------------|
# тангаж -------------|-рысканье----------|----крен-----------|
# --------------------|-------------------|-------------------|

while True:
    im = get_image()

    angles = angle_detector.get_face_angle(im)
    #
    text = prepare_text(angles)
    cv2.putText(im, text=text[0], org=(0, 40), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.7, color=0)
    # cv2.putText(im, text=text[1], org=(0, 160), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.7, color=0)
    # cv2.putText(im, text=text[2], org=(0, 260), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.7, color=0)
    #

    cv2.imshow('video', im)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

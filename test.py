import cv2

from face_angle_detector import FaceAngleDetector

angle_detector = FaceAngleDetector(FaceAngleDetector.METHOD_SOLVE_CUSTOM_POINTS)

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

while True:
    im = get_image()

    angle_detector.get_face_angle(im)

    cv2.imshow('Video', im)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

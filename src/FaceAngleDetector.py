import dlib


# main class that performs prediction
from src.SolvePnpMethod import SolvePnpMethod


class FaceAngleDetector:

    METHOD_SOLVE_PNP = 'solve_pnp'
    METHOD_FACIAL_LANDMARKS = 'facial_landmarks' # this method is experimental
    # is not recommended for use

    def __init__(self, method, detector, predictor):
        self.method = method
        FaceAngleDetector.detector = detector
        FaceAngleDetector.predictor = predictor

    def get_face_angle(self, image):

        dets = FaceAngleDetector.detector(image, 1)
        result = []

        for d in dets:
            shape = FaceAngleDetector.predictor(image, d)
            result.append(SolvePnpMethod.get_angles(image, shape.parts()))

        return result

import dlib


# main class that performs prediction
from src.FacialPointsMethod import FacialPointsMethod
from src.SolvePnpMethod import SolvePnpMethod


class FaceAngleDetector:

    METHOD_SOLVE_PNP = 'solve_pnp'
    METHOD_FACIAL_LANDMARKS = 'facial_landmarks' # this method is experimental
    # is not recommended for use

    def __init__(self, method, detector, predictor):
        self.method = method
        # FaceAngleDetector.predictor_path = '../data/shape_predictor_68_face_landmarks.dat'
        FaceAngleDetector.detector = detector
        FaceAngleDetector.predictor = predictor

    def get_face_angle(self, image):

        dets = FaceAngleDetector.detector(image, 1)
        result = []

        for d in dets:
            shape = FaceAngleDetector.predictor(image, d)

            if self.method == FaceAngleDetector.METHOD_SOLVE_PNP:
                result.append(SolvePnpMethod.get_angles(image, shape.parts()))
            elif self.method == FaceAngleDetector.METHOD_FACIAL_LANDMARKS:
                result.append(FacialPointsMethod.get_angles(shape.parts()))

        return result

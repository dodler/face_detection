# @author Lyan Artyom
# more info in README file

import json
import os

import dlib

from FaceAngleDetector import FaceAngleDetector


# this class performs validation of face angle
# if face is tilted with angles higher than thresholds
# validation will give false

class Validator:

    @staticmethod
    def init_static_instance(pitch_threshold, roll_threshold, yaw_threshold, angle_detector):
        Validator.angle_detector = angle_detector
        Validator.pitch_threshold = pitch_threshold
        Validator.roll_threshold = roll_threshold
        Validator.yaw_threshold = yaw_threshold

    def __init__(self, pitch, roll, yaw, angle_detector):
        self.pitch_threshold = pitch
        self.yaw_threshold = yaw
        self.roll_threshold = roll
        self.angle_detector = angle_detector

    @staticmethod
    def init_angle_detector(path):
        json_str = ''
        with open(path) as f:
            for line in f.readlines():
                json_str += line

        # print json_str

        settings = json.loads(json_str)
        points_path = str(settings['landmarks_path'])

        print points_path

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(points_path)

        if settings['method'] == FaceAngleDetector.METHOD_SOLVE_PNP:
            angle_detector = FaceAngleDetector(FaceAngleDetector.METHOD_SOLVE_PNP, detector, predictor)
        elif settings['method'] == FaceAngleDetector.METHOD_FACIAL_LANDMARKS:
            angle_detector = FaceAngleDetector(FaceAngleDetector.METHOD_FACIAL_LANDMARKS, detector, predictor)

        yaw_threshold = settings['yaw_threshold']
        pitch_threshold = settings['pitch_threshold']
        roll_threshold = settings['roll_threshold']

        instance = Validator(pitch_threshold, roll_threshold, yaw_threshold, angle_detector)
        Validator.init_static_instance(pitch_threshold, roll_threshold, yaw_threshold, angle_detector)
        return instance


    @staticmethod
    def validate(face):

        if ~hasattr(Validator, 'angle_detector'):
            dir = os.path.dirname(__file__)
            filename = os.path.join(dir, '../../data/settings.json')
            Validator.init_angle_detector(filename)


        angles = Validator.angle_detector.get_face_angle(face)

        if len(angles) == 0:
            return False

        if abs(angles[0][0]) > Validator.pitch_threshold:
            return False
        if abs(angles[0][1]) > Validator.yaw_threshold:
            return False
        if abs(angles[0][2]) > Validator.roll_threshold:
            return False

        return True

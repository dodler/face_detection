from math import atan


class FacialPointsMethod:
    # TODO
    # improve accuracy - try to use machine learning method to calculate
    # facial angles using facial landmarks

    rad_to_deg = 57.2958
    eyes_angle_ratio = 167.0
    nose_angle_ratio = 0.738

    @staticmethod
    def get_angles(shape_parts):
        eye_left_left_p_x = shape_parts[36].x
        eye_left_left_p_y = shape_parts[36].y

        eye_left_right_p_x = shape_parts[39].x
        eye_left_right_p_y = shape_parts[39].y

        eye_right_left_p_x = shape_parts[42].x

        eye_right_right_p_x = shape_parts[45].x

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

        nose_angle = FacialPointsMethod.nose_angle_ratio * (4.41 - nose_top_size / nose_bot_size)
        eye_angle = (1.0 - (left_eye_width / (1.0 * right_eye_width))) * FacialPointsMethod.eyes_angle_ratio
        rotation = FacialPointsMethod.rad_to_deg * atan(left_eye_height / (1.0 * left_eye_width))

        return nose_angle, eye_angle, rotation
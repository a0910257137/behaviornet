from pprint import pprint
import numpy as np
import copy


class Base:
    def __init__(self):
        self.anno_keys = {
            "countour_face": 17,
            "left_eyebrow": 5,
            "right_eyebrow": 5,
            "left_eye": 6,
            "right_eye": 6,
            "nose": 9,
            "outer_lip": 12,
            "inner_lip": 8
        }

        self.lnmk_25 = [
            'countour_lnmk_0',
            'countour_lnmk_8',
            'countour_lnmk_16',
            'nose_lnmk_27',
            'nose_lnmk_33',
            'left_eye_lnmk_36',
            'left_eye_lnmk_37',
            'left_eye_lnmk_38',
            'left_eye_lnmk_39',
            'left_eye_lnmk_40',
            'left_eye_lnmk_41',
            'right_eye_lnmk_42',
            'right_eye_lnmk_43',
            'right_eye_lnmk_44',
            'right_eye_lnmk_45',
            'right_eye_lnmk_46',
            'right_eye_lnmk_47',
            'lip_lnmk_48',
            'lip_lnmk_50',
            'lip_lnmk_51',
            'lip_lnmk_52',
            'lip_lnmk_54',
            'lip_lnmk_56',
            'lip_lnmk_57',
            'lip_lnmk_58',
        ]
        self.lnmk_14 = [
            'countour_lnmk_8',
            'countour_lnmk_16',
            'nose_lnmk_27',
            'nose_lnmk_33',
            'left_eye_lnmk_36',
            'left_eye_lnmk_37',
            'left_eye_lnmk_38',
            'left_eye_lnmk_39',
            'left_eye_lnmk_40',
            'left_eye_lnmk_41',
            'right_eye_lnmk_42',
            'right_eye_lnmk_43',
            'right_eye_lnmk_44',
            'right_eye_lnmk_45',
            'right_eye_lnmk_46',
            'right_eye_lnmk_47',
            'lip_lnmk_48',
            'lip_lnmk_50',
            'lip_lnmk_51',
            'lip_lnmk_52',
            'lip_lnmk_54',
            'lip_lnmk_56',
            'lip_lnmk_57',
            'lip_lnmk_58',
        ]

    def parse_lnmk(self, item):
        output = dict()
        lnmk_25 = copy.deepcopy(self.lnmk_25)
        for key in self.anno_keys:
            lnmks = item[key]
            lnmk_serial_keys = lnmk_25[:len(lnmks)]
            del lnmk_25[:len(lnmks)]
            for lnmk, lnmk_serial_key in zip(lnmks, lnmk_serial_keys):
                output[lnmk_serial_key] = np.asarray(lnmk)
        return output

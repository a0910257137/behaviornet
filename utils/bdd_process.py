import numpy as np
import copy
import pandas as pd
from pprint import pprint


def to_tp_od_bdd(bdd_results, batches_preds, batches_frames, cates):
    for batch_preds, batch_frames in zip(batches_preds, batches_frames):
        for preds, frames in zip(batch_preds, batch_frames):
            pred_frame = {
                'dataset': frames['dataset'],
                'sequence': frames['sequence'],
                'name': frames['name'],
                'labels': []
            }
            valid_mask = np.all(~np.isinf(preds), axis=-1)
            preds = preds[valid_mask]
            for pred_obj, frame in zip(preds, frames):
                pred_lb = {
                    'category': cates[int(pred_obj[-1])].upper(),
                    'box2d': {
                        'y1': float(pred_obj[0]),
                        'x1': float(pred_obj[1]),
                        'y2': float(pred_obj[2]),
                        'x2': float(pred_obj[3])
                    }
                }
                pred_frame['labels'].append(pred_lb)
            bdd_results['frame_list'].append(pred_frame)
    return bdd_results


def to_lnmk_bdd(bdd_results, batches_preds, batches_frames, cates):

    lnmk_keys = {
        "countour_face": 17,
        "left_eyebrow": 5,
        "right_eyebrow": 5,
        "left_eye": 6,
        "right_eye": 6,
        "nose": 9,
        "outer_lip": 12,
        "inner_lip": 8
    }
    batches_preds = batches_preds.numpy()
    for preds, frame in zip(batches_preds, batches_frames):
        pred_frame = {
            'dataset': frame['dataset'],
            'sequence': frame['sequence'],
            'name': frame['name'],
            'labels': []
        }
        preds = preds.astype(np.float16)
        preds = preds.tolist()

        for gt_lnmks in frame['labels']:
            keypoints = gt_lnmks['keypoints']
            lnmk_keys = list(keypoints.keys())
            pred_lb = {'keypoints': {}, 'category': gt_lnmks['category']}
            for k in lnmk_keys:
                lnmks = keypoints[k]
                gt_length = len(lnmks)
                pred_lb['keypoints'][k] = preds[:gt_length]
                del preds[:gt_length]
            pred_frame['labels'].append(pred_lb)
        bdd_results['frame_list'].append(pred_frame)
    return bdd_results


def transform_pd_data(report_results,
                      is_post_cal,
                      metric_type='keypoints',
                      eval_types=('accuracy', 'precision', 'recall')):
    mean_df = None
    if metric_type.lower() == 'keypoints' or metric_type.lower(
    ) == 'landmarks':
        # pd_headers = {'center_point': [], 'top_left': [], 'bottom_right': []}
        pd_headers = {
            'countour_lnmk_0': [],
            'countour_lnmk_8': [],
            'countour_lnmk_16': [],
            'nose_lnmk_27': [],
            'nose_lnmk_33': [],
            'left_eye_lnmk_36': [],
            'left_eye_lnmk_37': [],
            'left_eye_lnmk_38': [],
            'left_eye_lnmk_39': [],
            'left_eye_lnmk_40': [],
            'left_eye_lnmk_41': [],
            'right_eye_lnmk_42': [],
            'right_eye_lnmk_43': [],
            'right_eye_lnmk_44': [],
            'right_eye_lnmk_45': [],
            'right_eye_lnmk_46': [],
            'right_eye_lnmk_47': [],
            'lip_lnmk_48': [],
            'lip_lnmk_50': [],
            'lip_lnmk_51': [],
            'lip_lnmk_52': [],
            'lip_lnmk_54': [],
            'lip_lnmk_56': [],
            'lip_lnmk_57': [],
            'lip_lnmk_58': []
        }

    elif metric_type == 'IoU' or metric_type.lower() == 'box2d':
        pd_headers = {'IoU': []}
    data_frame = {'Eval_type': eval_types}
    for t in eval_types:
        type_results = report_results[t]
        for key in pd_headers:
            eval_vals = type_results[key]
            pd_headers[key].append(eval_vals)
    if is_post_cal:
        post_mean_infos = {
            'countour': [],
            'left_eye': [],
            'right_eye': [],
            'nose': [],
            'lip': []
        }
        for pd_key in pd_headers:
            for post_key in post_mean_infos:
                if post_key in pd_key:
                    post_mean_infos[post_key].append(pd_headers[pd_key])
        for post_key in post_mean_infos:
            array = np.stack(post_mean_infos[post_key])
            array = np.mean(array, axis=0)
            post_mean_infos[post_key] = array

        mean_data_frame = copy.deepcopy(data_frame)
        mean_data_frame.update(
            {k: post_mean_infos[k]
             for k in post_mean_infos})
        mean_df = pd.DataFrame(mean_data_frame)
    data_frame.update({k: pd_headers[k] for k in pd_headers})
    df = pd.DataFrame(data_frame)
    return df, mean_df
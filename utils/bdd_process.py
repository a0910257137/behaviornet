import numpy as np
from pprint import pprint


def to_tp_od_bdd(batches_preds, batches_frames, cates):
    bdd_results = {"frame_list": []}
    for batch_preds, batch_frames in zip(batches_preds, batches_frames):
        batch_frame_preds = []
        for preds, frames in zip(batch_preds, batch_frames):
            # pred_frame = {
            #     'storage': frames['storage'],
            #     'dataset': frames['dataset'],
            #     'sequence': frames['sequence'],
            #     'name': frames['name'],
            #     'labels': []
            # }
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
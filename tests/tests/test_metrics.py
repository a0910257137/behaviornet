from metrics.compute import ComputeIOU
import json
from pprint import pprint
import pandas as pd


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def iou_report(iou, cates_order):
    precs, recs = [], []
    cates = []
    threshold = [0.95, 0.75, 0.5, 0.25, 0.1, 0.01]
    for thres in threshold:
        result_prec = iou.report('AP', thres)
        result_rec = iou.report('AR', thres)

        for cate in cates_order:
            try:
                prec_val = result_prec[cate]
                rec_val = result_rec[cate]
            except:
                prec_val = '0'
                rec_val = '0'
            precs.append(prec_val)
            recs.append(rec_val)
            cates.append(cate)

    dict = {
        "Category": cates,
        "Threshold": threshold,
        "Precision": precs,
        'Recall': recs
    }

    df = pd.DataFrame(dict)
    pprint(df)
    df.to_csv('infer_test_iou.csv', float_format='%.3f')


path = '/aidata/anders/objects/WF/annos/BDD_test.json'
annos = load_json(path)
gtbdd = evalbdd = annos
iou = ComputeIOU(gtbdd, evalbdd)
iou_report(iou, ['FACE'])
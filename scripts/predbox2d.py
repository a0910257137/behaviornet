import face_recognition
import os
from glob import glob
from pprint import pprint
import numpy as np
import cv2
import json
import copy
from pathlib import Path
from tqdm import tqdm
import sys
import commentjson

sys.path.insert(0, str(Path(__file__).parent.parent))
from behavior_predictor.inference import BehaviorPredictor
# path = "/aidata/anders/data_collection/relabel/demo_test/imgs/*.jpg"
# img_paths = glob(path)


def dump_json(path, data):
    """Dump data to json file

    Arguments:
        data {[Any]} -- data
        path {str} -- json file path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)


# bdd_base = {
#     "name":
#     None,
#     "dataset":
#     None,
#     "labels": [{
#         "box2d": {
#             'x1': None,
#             'y1': None,
#             'x2': None,
#             'y2': None
#         },
#         "keypoints": []
#     }]
# }

# bdd_results = {"frame_list": []}
# #TODO: make bdd
# for img_path in tqdm(img_paths):
#     image = face_recognition.load_image_file(img_path)
#     face_landmarks_list = face_recognition.face_landmarks(image)
#     name = img_path.split("/")[-1]
#     bdd = copy.deepcopy(bdd_base)
#     bdd['dataset'] = "test"
#     bdd['name'] = name

#     for lb in face_landmarks_list:
#         keys = lb.keys()
#         tmp = [lb[k] for k in keys]
#         kps = np.concatenate(tmp, axis=0)
#         tl = np.min(kps, axis=0).astype(np.int32)
#         br = np.max(kps, axis=0).astype(np.int32)
#         lb_base = copy.deepcopy(bdd['labels'][0])
#         bdd['labels'].pop()
#         lb_base["box2d"]['x1'] = int(tl[0])
#         lb_base["box2d"]['y1'] = int(tl[1])
#         lb_base["box2d"]['x2'] = int(br[0])
#         lb_base["box2d"]['y2'] = int(br[1])
#         bdd['labels'].append(lb_base)
#         # image = cv2.rectangle(image, tuple(tl), tuple(br), (0, 255, 0), 5)
#     bdd_results["frame_list"].append(bdd)
# dump_json(
#     path=
#     "/aidata/anders/data_collection/relabel/demo_test/annos/BDD_demo_test_model2.json",
#     data=bdd_results)

#---------------------------------------------------------------------------------------------

config_path = "./config/kps.json"
img_path_root = "/aidata/anders/data_collection/relabel/demo_test/imgs"
bdd_base = {
    "name":
    None,
    "dataset":
    None,
    "labels": [{
        "box2d": {
            'x1': None,
            'y1': None,
            'x2': None,
            'y2': None
        },
        "keypoints": []
    }]
}
bdd_results = {"frame_list": []}
with open(config_path) as f:
    config = commentjson.loads(f.read())
BATCH_SIZE = 1

print('Restore model')
predictor = BehaviorPredictor(config)
print(predictor)
img_names = list(filter(lambda x: 'jpg' in x, os.listdir(img_path_root)))
img_paths = list(map(lambda x: os.path.join(img_path_root, x), img_names))
img_path_batchs = [
    img_paths[idx:idx + BATCH_SIZE]
    for idx in range(0, len(img_paths), BATCH_SIZE)
]
img_name_batchs = [
    img_names[idx:idx + BATCH_SIZE]
    for idx in range(0, len(img_names), BATCH_SIZE)
]
for i, (img_paths, img_names) in enumerate(zip(img_path_batchs,
                                               img_name_batchs)):
    imgs, origin_shapes, orig_imgs = [], [], []
    for img_path in img_paths:
        print(img_path)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        origin_shapes.append((h, w))
        orig_imgs.append(img)
        imgs.append(img)
    rets = predictor.pred(imgs, origin_shapes)
    b_bboxes, b_lnmks, b_nose_scores = rets
    b_nose_scores = b_nose_scores.numpy()
    b_lnmks = b_lnmks.numpy()
    b_bboxes = b_bboxes.numpy()
    for bboxes, img_path, img in zip(b_bboxes, img_paths, imgs):
        name = img_path.split("/")[-1]
        bdd = copy.deepcopy(bdd_base)
        bdd['dataset'] = "test"
        bdd['name'] = name
        valid_mask = np.all(np.isfinite(bboxes), axis=-1)
        bboxes = bboxes[valid_mask]
        obj_bboxes = bboxes[:, :4]
        obj_bboxes = obj_bboxes.astype(np.int32)
        for bbox in obj_bboxes:
            tl = bbox[:2]
            tl = tuple(tl[::-1])
            br = bbox[2:]
            br = tuple(br[::-1])
            lb_base = copy.deepcopy(bdd['labels'][0])
            bdd['labels'].pop()
            lb_base["box2d"]['x1'] = int(tl[0])
            lb_base["box2d"]['y1'] = int(tl[1])
            lb_base["box2d"]['x2'] = int(br[0])
            lb_base["box2d"]['y2'] = int(br[1])
            bdd['labels'].append(lb_base)
            # img = cv2.rectangle(img, tl, br, (0, 255, 0), 2)
        # cv2.imwrite("./output.jpg", img)
        bdd_results["frame_list"].append(bdd)
dump_json(
    path=
    "/aidata/anders/data_collection/relabel/demo_test/annos/BDD_demo_test_model3.json",
    data=bdd_results)
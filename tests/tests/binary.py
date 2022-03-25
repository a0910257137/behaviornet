from glob import glob
from sys import path

from utils.io import *
import cv2
from pprint import pprint

from tqdm import tqdm


def s(paths):
    paths = sorted(list(paths))
    return paths


def loop_func(bdd_results, path_list):
    progress_bar = tqdm(total=len(path_list))

    for path in path_list:
        guy = path.split('/')[-3]
        sequence = path.split('/')[-2]
        name = path.split('/')[-1]
        img = cv2.imread(path)
        tot_name = "{}_{}_{}".format(guy, sequence, name)

        cv2.imwrite(os.path.join(save_root, tot_name), img)

        attri = {"eye_status": sequence, "category": "FACE"}
        frame_infos = {
            "dataset": "Classfication",
            "sequence": None,
            "name": tot_name,
            "labels": [attri]
        }

        bdd_results["frame_list"].append(frame_infos)
        progress_bar.update(1)
    return bdd_results


save_root = "/aidata/anders/objects/landmarks/demo_video/2021_12_24/classification/imgs"
anders_paths = glob(
    "/aidata/anders/objects/landmarks/demo_video/2021_12_24/anders_eyes/open/*.jpg"
)

evian_paths = glob(
    "/aidata/anders/objects/landmarks/demo_video/2021_12_24/evian_eyes/open/*.jpg"
)

po_yuan_paths = glob(
    "/aidata/anders/objects/landmarks/demo_video/2021_12_24/po_yuan_eyes/open/*.jpg"
)

anders_paths = s(anders_paths)
evian_paths = s(evian_paths)
po_yuan_paths = s(po_yuan_paths)
bdd_results = {"frame_list": []}
bdd_results = loop_func(bdd_results, anders_paths)
bdd_results = loop_func(bdd_results, evian_paths)
bdd_results = loop_func(bdd_results, po_yuan_paths)

anders_paths = glob(
    "/aidata/anders/objects/landmarks/demo_video/2021_12_24/anders_eyes/close/*.jpg"
)

evian_paths = glob(
    "/aidata/anders/objects/landmarks/demo_video/2021_12_24/evian_eyes/close/*.jpg"
)

po_yuan_paths = glob(
    "/aidata/anders/objects/landmarks/demo_video/2021_12_24/po_yuan_eyes/close/*.jpg"
)

anders_paths = s(anders_paths)
evian_paths = s(evian_paths)
po_yuan_paths = s(po_yuan_paths)

bdd_results = loop_func(bdd_results, anders_paths)
bdd_results = loop_func(bdd_results, evian_paths)
bdd_results = loop_func(bdd_results, po_yuan_paths)
save_path = "/aidata/anders/objects/landmarks/demo_video/2021_12_24/classification/annos/BDD_cls.json"

dump_json(path=save_path, data=bdd_results)

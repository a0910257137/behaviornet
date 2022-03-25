import copy
from pickle import FALSE
from utils.io import *
import os
import cv2
import numpy as np
from pprint import pprint
from glob import glob
from tqdm import tqdm

kp_base_dict = {
    "countour_face_lnmk_0": None,
    "countour_face_lnmk_1": None,
    "countour_face_lnmk_2": None,
    "countour_face_lnmk_3": None,
    "countour_face_lnmk_4": None,
    "countour_face_lnmk_5": None,
    "countour_face_lnmk_6": None,
    "countour_face_lnmk_7": None,
    "countour_face_lnmk_8": None,
    "countour_face_lnmk_9": None,
    "countour_face_lnmk_10": None,
    "countour_face_lnmk_11": None,
    "countour_face_lnmk_12": None,
    "countour_face_lnmk_13": None,
    "countour_face_lnmk_14": None,
    "countour_face_lnmk_15": None,
    "countour_face_lnmk_16": None,
    "left_eyebrow_lnmk_17": None,
    "left_eyebrow_lnmk_18": None,
    "left_eyebrow_lnmk_19": None,
    "left_eyebrow_lnmk_20": None,
    "left_eyebrow_lnmk_21": None,
    "right_eyebrow_lnmk_22": None,
    "right_eyebrow_lnmk_23": None,
    "right_eyebrow_lnmk_24": None,
    "right_eyebrow_lnmk_25": None,
    "right_eyebrow_lnmk_26": None,
    "left_eye_lnmk_27": None,
    "left_eye_lnmk_28": None,
    "left_eye_lnmk_29": None,
    "left_eye_lnmk_30": None,
    "left_eye_lnmk_31": None,
    "left_eye_lnmk_32": None,
    "right_eye_lnmk_33": None,
    "right_eye_lnmk_34": None,
    "right_eye_lnmk_35": None,
    "right_eye_lnmk_36": None,
    "right_eye_lnmk_37": None,
    "right_eye_lnmk_38": None,
    "nose_lnmk_39": None,
    "nose_lnmk_40": None,
    "nose_lnmk_41": None,
    "nose_lnmk_42": None,
    "nose_lnmk_43": None,
    "nose_lnmk_44": None,
    "nose_lnmk_45": None,
    "nose_lnmk_46": None,
    "nose_lnmk_47": None,
    "outer_lip_lnmk_48": None,
    "outer_lip_lnmk_49": None,
    "outer_lip_lnmk_50": None,
    "outer_lip_lnmk_51": None,
    "outer_lip_lnmk_52": None,
    "outer_lip_lnmk_53": None,
    "outer_lip_lnmk_54": None,
    "outer_lip_lnmk_55": None,
    "outer_lip_lnmk_56": None,
    "outer_lip_lnmk_57": None,
    "outer_lip_lnmk_58": None,
    "outer_lip_lnmk_59": None,
    "inner_lip_lnmk_60": None,
    "inner_lip_lnmk_61": None,
    "inner_lip_lnmk_62": None,
    "inner_lip_lnmk_63": None,
    "inner_lip_lnmk_64": None,
    "inner_lip_lnmk_65": None,
    "inner_lip_lnmk_66": None,
    "inner_lip_lnmk_67": None,
}


def convert_5_WFLW(
        annoa_path="/aidata/anders/objects/landmarks/WFLW/widerface/train/label.txt",
        img_root="/aidata/anders/objects/landmarks/WFLW/WFLW_images"):
    text_lines = load_text(annoa_path)

    tmp_dict = {}
    for i, line in enumerate(text_lines):
        if '#' == line.strip().split()[0]:
            name = line.split(' ')[-1]
            img_path = os.path.join(img_root, name)
            img_path = str(img_path)
        else:
            line = line.strip().split()
            label = np.asarray(list(map(float, line)), dtype=np.float32)

            tl = [label[0], label[1]]
            br = [label[0] + label[2], label[1] + label[3]]

            lnmk1 = [float(label[4]), float(label[5])]

            lnmk2 = [float(label[7]), float(label[8])]

            lnmk3 = [float(label[10]), float(label[11])]

            lnmk4 = [float(label[13]), float(label[14])]

            lnmk5 = [float(label[16]), float(label[17])]
            lnmks = np.stack([lnmk1, lnmk2, lnmk3, lnmk4, lnmk5])
            lnmk_tl = np.min(lnmks, axis=0)
            lnmk_br = np.max(lnmks, axis=0)
            if lnmk_tl[1] > 0:
                tl_y = abs(tl[1] - lnmk_tl[1]) * (1 / 2)
                box2d = {
                    "x1": int(tl[0]),
                    "y1": int(lnmk_tl[1] - tl_y),
                    "x2": int(br[0]),
                    "y2": int(br[1])
                }

            else:
                box2d = {
                    "x1": int(tl[0]),
                    "y1": int(tl[1] + 5),
                    "x2": int(br[0]),
                    "y2": int(br[1])
                }

            kp_dict = copy.deepcopy(kp_base_dict)

            kp_dict['left_eye_lnmk_27'] = lnmk1[::-1]
            kp_dict['left_eye_lnmk_28'] = lnmk1[::-1]
            kp_dict['left_eye_lnmk_29'] = lnmk1[::-1]
            kp_dict['left_eye_lnmk_30'] = lnmk1[::-1]
            kp_dict['left_eye_lnmk_31'] = lnmk1[::-1]
            kp_dict['left_eye_lnmk_32'] = lnmk1[::-1]

            kp_dict['right_eye_lnmk_33'] = lnmk2[::-1]
            kp_dict['right_eye_lnmk_34'] = lnmk2[::-1]
            kp_dict['right_eye_lnmk_35'] = lnmk2[::-1]
            kp_dict['right_eye_lnmk_36'] = lnmk2[::-1]
            kp_dict['right_eye_lnmk_37'] = lnmk2[::-1]
            kp_dict['right_eye_lnmk_38'] = lnmk2[::-1]

            kp_dict['nose_lnmk_42'] = lnmk3[::-1]
            kp_dict['outer_lip_lnmk_48'] = lnmk4[::-1]
            kp_dict['outer_lip_lnmk_54'] = lnmk5[::-1]
            lb = {"keypoints": kp_dict, "box2d": box2d, "category": "FACE"}
            if name not in list(tmp_dict.keys()):
                tmp_dict[name] = [lb]
            else:
                tmp_dict[name].append(lb)

    bdd_results = {"frame_list": []}
    for key in tmp_dict.keys():

        frame_infos = {
            "dataset": "WFLW",
            "sequence": key.split("/")[0],
            "name": key.split("/")[-1],
            "labels": []
        }
        for item in tmp_dict[key]:
            # for key in item["keypoints"].keys():
            #     if item["keypoints"][key] is None:
            #         item["keypoints"][key] = [100, 100]
            frame_infos["labels"].append(item)
        bdd_results["frame_list"].append(frame_infos)

    return bdd_results


def convert_WFLW(
        annoa_path="/aidata/anders/objects/landmarks/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt",
        img_root='/aidata/anders/objects/landmarks/WFLW'):
    annos = load_text(annoa_path)
    idxs = list(range(0, 34, 2)) + list(range(33, 38, 1)) + list(
        range(42, 47, 1)) + [60, 61, 63, 64, 65, 67] + [
            68, 69, 71, 72, 73, 75
        ] + list(range(51, 60, 1)) + list(range(76, 96, 1))

    # annos = load_json(path)
    bdd_results = {"frame_list": []}
    tmp_is_in = []
    progress_bar = tqdm(total=len(annos))
    for line in annos:
        line = line.strip().split()
        marks = np.asarray(list(map(float, line[:196])),
                           dtype=np.float32).reshape(-1, 2)
        marks = marks[idxs]
        marks = marks[:, ::-1]
        boxes = np.asarray(list(map(float, line[196:200])),
                           dtype=np.float32).reshape(-1, 2)
        boxes = boxes[:, ::-1]
        sequence = line[-1].split("/")[0]
        name = line[-1].split("/")[1]
        kp_dict = copy.deepcopy(kp_base_dict)
        for i, key in enumerate(list(kp_dict.keys())):
            kp = list(marks[i])
            kp_dict[key] = marks[i].tolist()
        frame_infos = {
            "dataset": "WFLW",
            "sequence": sequence,
            "name": name,
            "labels": []
        }
        kps = np.asarray(marks)
        tl = np.min(kps, axis=0)
        br = np.max(kps, axis=0)

        lb = {
            "keypoints": kp_dict,
            "box2d": {
                "x1": int(tl[1]),
                "y1": int(tl[0]),
                "x2": int(br[1]),
                "y2": int(br[0])
            },
            "category": "FACE"
        }
        if line[-1] in tmp_is_in:
            #TODO: sear ch
            for frame in bdd_results["frame_list"]:
                name = frame['name']
                if line[-1].split("/")[-1] == frame['name']:
                    frame["labels"].append(lb)
        else:
            frame_infos["labels"].append(lb)
            tmp_is_in.append(line[-1])
            bdd_results["frame_list"].append(frame_infos)
        progress_bar.update(1)
    return bdd_results


def convert_FFHQ(
        annoa_path="/aidata/anders/objects/landmarks/FFHQ/annos/BDD_FFHQ.json",
        img_root='/aidata/anders/objects/box2d/ffhq/imgs'):
    annos = load_json(annoa_path)
    bdd_results = {"frame_list": []}
    for frame in annos["frame_list"]:
        for lb in frame["labels"]:
            keypoints = lb["keypoints"]
            keys = list(keypoints.keys())
            tmp = []
            for key in keys:
                kp = keypoints[key]
                tmp.append(kp)
            marks = np.asarray(tmp)
            kps = np.asarray(marks)
            tl = np.min(kps, axis=0)
            br = np.max(kps, axis=0)
            lb["box2d"] = {
                "x1": int(tl[1]),
                "y1": int(tl[0]),
                "x2": int(br[1]),
                "y2": int(br[0])
            }
        bdd_results["frame_list"].append(frame)
    return bdd_results


def convert_300W(
        annoa_path="/aidata/anders/objects/landmarks/300W/annos/BDD_300W.json",
        img_root='/aidata/anders/objects/landmarks/300W/imgs'):
    annos = load_json(annoa_path)
    bdd_results = {"frame_list": []}
    for frame in annos["frame_list"]:
        for lb in frame["labels"]:
            keypoints = lb["keypoints"]
            keys = list(keypoints.keys())
            tmp = []
            for key in keys:
                kp = keypoints[key]
                tmp.append(kp)
            marks = np.asarray(tmp)
            kps = np.asarray(marks)
            tl = np.min(kps, axis=0)
            br = np.max(kps, axis=0)
            lb["box2d"] = {
                "x1": int(tl[1]),
                "y1": int(tl[0]),
                "x2": int(br[1]),
                "y2": int(br[0])
            }
        bdd_results["frame_list"].append(frame)
    return bdd_results


def convert_widerface(
        annoa_path="/aidata/anders/objects/landmarks/300W/annos/BDD_300W.json",
        img_root='/aidata/anders/objects/landmarks/300W/imgs'):
    img_paths = list(
        glob("/aidata/anders/objects/landmarks/WFLW/widerface/train/*.jpg"))
    texe_paths = list(
        glob(
            "/aidata/anders/objects/landmarks/WFLW/widerface/train/*.txt"))[1:]
    bdd_results = {"frame_list": []}
    for img_path, tetx_path in zip(img_paths, texe_paths):
        text_lines = load_text(tetx_path)
        img = cv2.imread(img_path)
        img_size = np.asarray(img.shape[:-1])

        frame_infos = {
            "dataset": "widerface",
            "sequence": None,
            "name": tetx_path.split("/")[-1].replace(".txt", ".jpg"),
            "labels": []
        }
        for line in text_lines:

            line = line.strip().split()
            marks = np.asarray(list(map(float, line)), dtype=np.float32)
            marks = marks[1:]
            img_size = np.tile(img_size[::-1], (7))
            marks = marks * img_size
            marks = marks.astype(int)
            center_kp = marks[:2]
            wh = marks[2:4]
            lnmks = marks[4:].reshape((-1, 2))

            tl = ((center_kp - wh / 2) + 0.5).astype(np.int32)
            br = ((center_kp + wh / 2) + 0.5).astype(np.int32)
            # br = center_kp + wh / 2
            lnmks = lnmks[:, ::-1]
            l_eye_lnmk = lnmks[0]
            r_eye_lnmk = lnmks[1]
            nose_lnmk = lnmks[2]
            l_mouth_lnmk = lnmks[3]
            r_mouth_lnmk = lnmks[4]
            kp_dict = copy.deepcopy(kp_base_dict)

            kp_dict['left_eye_lnmk_27'] = l_eye_lnmk
            kp_dict['right_eye_lnmk_33'] = r_eye_lnmk
            kp_dict['nose_lnmk_42'] = nose_lnmk
            kp_dict['outer_lip_lnmk_48'] = l_mouth_lnmk
            kp_dict['outer_lip_lnmk_54'] = r_mouth_lnmk

        landmarks = np.asarray(list(map(float, text_lines)),
                               dtype=np.float32).reshape(-1, 2)

    return


def convert_300VW(
        annoa_path="/aidata/anders/objects/landmarks/300VW/annos/BDD_300VW.json",
        img_root='/aidata/anders/objects/landmarks/300VW/imgs'):

    annos = load_json(annoa_path)
    bdd_results = {"frame_list": []}
    for frame in annos["frame_list"]:
        for lb in frame["labels"]:
            keypoints = lb["keypoints"]
            keys = list(keypoints.keys())
            tmp = []
            for key in keys:
                kp = keypoints[key]
                tmp.append(kp)
            marks = np.asarray(tmp)
            kps = np.asarray(marks)
            tl = np.min(kps, axis=0)
            br = np.max(kps, axis=0)
            lb["box2d"] = {
                "x1": int(tl[1]),
                "y1": int(tl[0]),
                "x2": int(br[1]),
                "y2": int(br[0])
            }
        bdd_results["frame_list"].append(frame)
    return bdd_results


def convert_CelebA(
        annoa_path="/aidata/anders/objects/landmarks/celeba/annos/BDD_CelebA.json",
        img_root='/aidata/anders/objects/landmarks/celeba/imgs'):
    annos = load_json(annoa_path)
    bdd_results = {"frame_list": []}
    for frame in annos["frame_list"]:
        for lb in frame["labels"]:
            keypoints = lb["keypoints"]
            keys = list(keypoints.keys())
            tmp = []
            for key in keys:
                kp = keypoints[key]
                tmp.append(kp)
            marks = np.asarray(tmp)
            kps = np.asarray(marks)
            tl = np.min(kps, axis=0)
            br = np.max(kps, axis=0)
            lb["box2d"] = {
                "x1": int(tl[1]),
                "y1": int(tl[0]),
                "x2": int(br[1]),
                "y2": int(br[0])
            }
        bdd_results["frame_list"].append(frame)
    return bdd_results


def convert_300W():
    def parse(bdd_results, sequence, img_paths, anno_paths):
        save_root = "/aidata/anders/objects/landmarks/300W/imgs"
        idx = list(range(0, 27, 1)) + list(range(36, 48, 1)) + list(
            range(27, 36, 1)) + list(range(48, 68, 1))
        for (img_path, anno_path) in zip(img_paths, anno_paths):
            name_list = img_path.split("/")[-1]
            if sequence == "afw" or sequence == "helen":
                name_list = name_list.split("_")
                if len(name_list) > 1:
                    name_list.pop(-1)
                tmp_str = str()
                for i, n in enumerate(name_list):
                    if i == 0:
                        tmp_str += n
                    else:
                        tmp_str = tmp_str + "_" + n
                name = tmp_str
            elif sequence == "ibug":
                name_list = name_list.split("_")
                if len(name_list) > 2:
                    name_list.pop(-1)
                tmp_str = str()
                for i, n in enumerate(name_list):
                    if i == 0:
                        tmp_str += n
                    else:
                        tmp_str = tmp_str + "_" + n
                name = tmp_str
            else:
                name = name_list
            name = "{}_{}.jpg".format(sequence, name)
            # if 1 no find if two > check
            img = cv2.imread(img_path)
            lnmks = np.loadtxt(anno_path,
                               comments=("version:", "n_points:", "{", "}"))
            lnmks = np.asarray(lnmks)[:, ::-1]
            lnmks = lnmks[idx]

            tl = np.min(lnmks, axis=0)
            br = np.max(lnmks, axis=0)
            kp_base = copy.deepcopy(kp_base_dict)
            keys = list(kp_base.keys())
            for key, lnmk in zip(keys, lnmks):
                kp_base[key] = lnmk.tolist()
                # lnmk = lnmk.astype(np.int32)
                # img = cv2.circle(img, tuple(lnmk[::-1]), 3, (0, 255, 0), -1)

            lb = {
                "keypoints": kp_base,
                "box2d": {
                    "x1": int(tl[1]),
                    "y1": int(tl[0]),
                    "x2": int(br[1]),
                    "y2": int(br[0])
                },
                "category": "FACE"
            }

            is_exits = False
            for frame in bdd_results["frame_list"]:
                if name == frame["name"]:
                    frame["labels"].append(lb)
                    is_exits = True
            if is_exits:
                continue

            frame_infos = {
                "dataset": "300W",
                "sequence": None,
                "name": name,
                "labels": []
            }
            cv2.imwrite(os.path.join(save_root, frame_infos["name"]), img)
            frame_infos["labels"].append(lb)
            bdd_results["frame_list"].append(frame_infos)

        return bdd_results

    # there are four splitted folder in 300W dataset
    bdd_results_p1 = {"frame_list": []}
    root = '/aidata/anders/objects/landmarks/300W'
    afw_img_paths = glob(os.path.join(root, "afw/*.jpg"))
    afw_anno_paths = glob(os.path.join(root, "afw/*.pts"))
    bdd_results_p1 = parse(bdd_results_p1, "afw", afw_img_paths,
                           afw_anno_paths)
    #----------------------------------------------------
    ibug_img_paths = glob(os.path.join(root, "ibug/*.jpg"))

    ibug_anno_paths = glob(os.path.join(root, "ibug/*.pts"))
    bdd_results_p2 = {"frame_list": []}
    bdd_results_p2 = parse(bdd_results_p2, "ibug", ibug_img_paths,
                           ibug_anno_paths)
    #----------------------------------------------------
    helen_img_paths = glob(os.path.join(root, "helen/trainset/*.jpg"))
    helen_anno_paths = glob(os.path.join(root, "helen/trainset/*.pts"))
    bdd_results_p3 = {"frame_list": []}
    bdd_results_p3 = parse(bdd_results_p3, "helen", helen_img_paths,
                           helen_anno_paths)
    #----------------------------------------------------
    lfpw_img_paths = glob(os.path.join(root, "lfpw/trainset/*.png"))
    lfpw_anno_paths = glob(os.path.join(root, "lfpw/trainset/*.pts"))
    bdd_results_p4 = {"frame_list": []}
    bdd_results_p4 = parse(bdd_results_p4, "lfpw", lfpw_img_paths,
                           lfpw_anno_paths)

    bdd_results = {"frame_list": []}
    bdd_results["frame_list"] = bdd_results_p1["frame_list"] + bdd_results_p2[
        "frame_list"] + bdd_results_p3["frame_list"] + bdd_results_p4[
            "frame_list"]
    return bdd_results


def convert_LFW(
        annoa_path="/aidata/anders/objects/landmarks/LFW/annos/BDD_LFW.json",
        img_root='/aidata/anders/objects/landmarks/LFW/imgs'):
    annos = load_json(annoa_path)
    bdd_results = {"frame_list": []}
    for frame in annos["frame_list"]:
        for lb in frame["labels"]:
            keypoints = lb["keypoints"]
            keys = list(keypoints.keys())
            tmp = []
            for key in keys:
                kp = keypoints[key]
                tmp.append(kp)
            marks = np.asarray(tmp)
            kps = np.asarray(marks)
            tl = np.min(kps, axis=0)
            br = np.max(kps, axis=0)
            lb["box2d"] = {
                "x1": int(tl[1]),
                "y1": int(tl[0]),
                "x2": int(br[1]),
                "y2": int(br[0])
            }
        bdd_results["frame_list"].append(frame)
    return bdd_results


def convert_driv(
        annoa_path="/aidata/anders/objects/landmarks/driv/drivPoints.txt",
        img_root='/aidata/anders/objects/landmarks/driv/imgs'):

    annos_lines = load_text(annoa_path)
    annos_lines = annos_lines[1:-1]
    # rules fileName,subject,imgNum,label,ang,xF,yF,wF,hF,xRE,yRE,xLE,yLE,xN,yN,xRM,yRM,xLM,yLM
    bdd_results = {"frame_list": []}
    for i, line in enumerate(annos_lines):

        line = line.split(",")

        name = line[0] + ".jpg"
        print(name)
        img_path = os.path.join(
            "/aidata/anders/objects/landmarks/driv/DrivImages/", name)
        # img = cv2.imread(img_path)
        frame_infos = {
            "dataset": "Driv",
            "sequence": None,
            "name": name,
            "labels": []
        }

        box2d_marks = np.asarray(list(map(float, line[5:])),
                                 dtype=np.float32).reshape(-1, 2)
        box2d = box2d_marks[:2]
        tl = box2d[0]
        w, h = box2d[1]

        br = tl + np.array([w, h])
        # cv2.rectangle(img, tuple(tl.astype(np.int32)),
        #   tuple(br.astype(np.int32)), (0, 0, 255), 3)
        lnmks = box2d_marks[2:]
        lnmks = lnmks[:, ::-1]

        kp_base = copy.deepcopy(kp_base_dict)
        kp_base["left_eye_lnmk_27"] = lnmks[0].tolist()
        kp_base["left_eye_lnmk_28"] = lnmks[0].tolist()
        kp_base["left_eye_lnmk_29"] = lnmks[0].tolist()
        kp_base["left_eye_lnmk_30"] = lnmks[0].tolist()
        kp_base["left_eye_lnmk_31"] = lnmks[0].tolist()
        kp_base["left_eye_lnmk_32"] = lnmks[0].tolist()
        kp_base["right_eye_lnmk_33"] = lnmks[1].tolist()
        kp_base["right_eye_lnmk_34"] = lnmks[1].tolist()
        kp_base["right_eye_lnmk_35"] = lnmks[1].tolist()
        kp_base["right_eye_lnmk_36"] = lnmks[1].tolist()
        kp_base["right_eye_lnmk_37"] = lnmks[1].tolist()
        kp_base["right_eye_lnmk_38"] = lnmks[1].tolist()

        kp_base["nose_lnmk_42"] = lnmks[2].tolist()
        kp_base["outer_lip_lnmk_48"] = lnmks[3].tolist()
        kp_base["outer_lip_lnmk_54"] = lnmks[4].tolist()
        lb_infos = {
            "category": "FACE",
            "keypoints": kp_base,
            "box2d": {
                "x1": int(tl[0] + 5),
                "y1": int(tl[1] + 5),
                "x2": int(br[0] - 5),
                "y2": int(br[1] - 5)
            }
        }
        frame_infos["labels"].append(lb_infos)
        bdd_results["frame_list"].append(frame_infos)
        for kp in lnmks:
            kp = kp.astype(np.int32)
        # cv2.circle(img, tuple(kp[::-1]), 3, (0, 255, 0), -1)
        # cv2.imwrite(
        #     os.path.join("/aidata/anders/objects/landmarks/driv/imgs",
        #                  frame_infos["name"]), img)
    return bdd_results


bdd_results = convert_driv()
# bdd_results = convert_300W()
# bdd_results = convert_5_WFLW()
# bdd_results = convert_FFHQ()
# bdd_results = convert_300W()
# bdd_results = convert_300VW()
# bdd_results = convert_LFW()
# bdd_results = convert_CelebA()
#TODO: later arrange wider face 5 landmarks
# bdd_results = convert_widerface()
save_path = "/aidata/anders/objects/landmarks/driv/annos/BDD_DRIV.json"
dump_json(save_path, bdd_results)

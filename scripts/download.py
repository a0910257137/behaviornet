import urllib.request
import pandas as pd
from glob import glob
from pprint import pprint
import csv
import cv2
import os
import argparse
from tqdm import tqdm
from box import Box
import numpy as np


def download(file_root, save_root):
    cvs_files = glob(os.path.join(file_root, '*.csv'))
    print('Beginning file download with urllib2...')
    download_records = Box({"Lost": 0, "Sccessful": 0, "Total": 0})
    for file in tqdm(cvs_files):
        file = open(file)
        csvreader = csv.reader(file)
        for row in csvreader:
            index = 5
            for i in range(3):
                extract_line = row[i * index:(i + 1) * index]
                pic_url = extract_line[0]
                box2d_infos = np.asarray(extract_line[1:]).astype(float)
                img_name = pic_url.split('/')[-2] + '_' + pic_url.split(
                    '/')[-1]
                save_path = os.path.join(save_root, img_name)
                try:
                    urllib.request.urlretrieve(pic_url, save_path)
                    # guess as yolo format
                    print('-' * 100)
                    print(save_path)
                    img = cv2.imread(save_path)
                    h, w, c = img.shape
                    # y x
                    tl = (np.array(box2d_infos[0], box2d_infos[2]) *
                          np.array([w, h])).astype(int)
                    br = (np.array(box2d_infos[1], box2d_infos[3]) *
                          np.array([w, h])).astype(int)
                    # x1, y1 = box2d_infos[:2] * np.array([h, w])
                    # center_kp = np.asarray([x1, y1])
                    # box_wh = box2d_infos[2:4] * np.array([h, w])
                    # tl = (center_kp - box_wh / 2).astype(int)
                    # br = (center_kp + box_wh / 2).astype(int)
                    img = cv2.rectangle(img, tuple(tl), tuple(br), (0, 255, 0),
                                        3)
                    cv2.imwrite('outpt.jpg', img)
                    download_records.Sccessful += 1
                except:
                    download_records.Lost += 1
                    xxxx
                    break
    download_records.Lost = download_records.Sccessful + download_records.Lost


def parse_config():
    parser = argparse.ArgumentParser('Argparser for download url images')
    parser.add_argument('--file_root')
    parser.add_argument('--save_root', default=str())
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_config()
    print('Downlaoder for open dataset in web')

    download(args.file_root, args.save_root)
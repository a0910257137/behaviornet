#!/usr/bin/python3
import numpy as np
import sys
import argparse
import cv2
import os
import json
import tensorflow as tf
import math
from pprint import pprint
from tqdm import tqdm
from box import Box

import subprocess
from glob import glob


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def make_dir(path):
    if not os.path.exists(path):
        os.umask(0)
        os.makedirs(path, mode=0o755)


def run(video_path, save_path, img_size):

    path_list = glob("{}/*png".format(save_path))
    if len(path_list) == 0:
        subprocess.run([
            '/bin/bash', '-O', 'extglob', '-c',
            'ffmpeg -i' + ' {} {}/image%04d.png'.format(video_path, save_path)
        ])
        path_list = glob("{}/*png".format(save_path))

    img_path_list = sorted(list(path_list))
    for img_path in tqdm(img_path_list):
        img = cv2.imread(img_path)
        img = cv2.resize(img, tuple(img_size))
        img = img[..., ::-1]
        img = img.tostring()
        if img_path.split('/')[-1].split('.')[-1] == 'png':
            filename = img_path.split('/')[-1].replace('png', 'tfrecords')
        else:
            filename = img_path.split('/')[-1].replace('jpg', 'tfrecords')
        example = tf.train.Example(features=tf.train.Features(
            feature={'b_images': _bytes_feature(img)}))
        save_dir = os.path.join(save_path, 'video_pool')
        make_dir(save_dir)
        writer = tf.io.TFRecordWriter(os.path.join(save_dir, filename))
        writer.write(example.SerializeToString())
        writer.close()
    print("End of process...")


# ffmpeg -i ./cache/demo-sbr.mp4 ./cache/demo-sbrs/image%04d.png
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--img_size', default=(256, 256), type=tuple)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()
    print("Processing video and saving image on {}".format(args.save_path))
    run(args.video_path, args.save_path, args.img_size)

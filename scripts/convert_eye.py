import tensorflow as tf
import commentjson
from box import Box
from pprint import pprint
import os
from glob import glob
import multiprocessing
import numpy as np
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

threads = multiprocessing.cpu_count()
features = {
    "origin_height": tf.io.FixedLenFeature([], dtype=tf.int64),
    "origin_width": tf.io.FixedLenFeature([], dtype=tf.int64),
    "b_theta": tf.io.FixedLenFeature([], dtype=tf.string),
    "b_images": tf.io.FixedLenFeature([], dtype=tf.string),
    "b_coords": tf.io.FixedLenFeature([], dtype=tf.string)
}


def representative_dataset_gen():
    path = "/aidata/anders/objects/landmarks/eye_wild/total/train/x_train.npy"
    eye_imgs = np.load(path)
    for eye_img in eye_imgs:
        b_eye_img = np.reshape(eye_img, [-1, 26, 34, 3]) / 255
        b_eye_img = b_eye_img.astype(np.float32)
        yield [b_eye_img]


def read_json(path):
    with open(path, mode="r") as f:
        return commentjson.load(f)


cp_dir = '/aidata/anders/objects/landmarks/eye_wild/total/archive_model/max'
model = tf.keras.models.load_model(cp_dir)
inputs = tf.keras.Input(shape=(26, 34, 3), batch_size=2, name='input')
preds = model(inputs, training=False)
model = tf.keras.Model(inputs, preds)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.target_spec.supported_types = [tf.float32]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
# converter.target_spec.supported_types = [tf.int8, tf.uint8]

converter.representative_dataset = representative_dataset_gen
tflite_model = converter.convert()
with open(
        '/aidata/anders/objects/landmarks/eye_wild/total/archive_model/max/FP32.tflite',
        'wb') as f:
    f.write(tflite_model)
tf.lite.experimental.Analyzer.analyze(
    model_path=
    '/aidata/anders/objects/landmarks/eye_wild/total/archive_model/max/FP32.tflite',
    gpu_compatibility=True)

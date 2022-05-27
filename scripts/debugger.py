import tensorflow as tf
from pprint import pprint
import os
from glob import glob
import multiprocessing
import numpy as np
import cv2
import pandas as pd

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
    filenames = glob(
        os.path.join(
            '/aidata/anders/objects/landmarks/demo_test/tf_records/train/*.tfrecords'
        ))
    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=threads)
    ds = iter(ds)
    for i in range(380):
        ds_item = ds.next()
        parse_vals = tf.io.parse_example(ds_item, features)
        b_images = tf.io.decode_raw(parse_vals['b_images'], tf.uint8)
        b_images = tf.reshape(b_images, [-1, 192, 320, 3])
        b_images = b_images / 255
        yield [b_images]


RESULTS_FILE = '/aidata/anders/objects/landmarks/AFLW/archive_model/add_capacity/debugger/debugger_results.csv'
layer_stats = pd.read_csv(RESULTS_FILE)
layer_stats['range'] = 255.0 * layer_stats['scale']
layer_stats['rmse/scale'] = layer_stats.apply(
    lambda row: np.sqrt(row['mean_squared_error']) / row['scale'], axis=1)

heads = list(layer_stats.keys()) + ['rmse/scale']
suspected_layers = list()
#select tensor name

suspected_layers.extend(list(layer_stats[130:140]['tensor_name']))

debug_options = tf.lite.experimental.QuantizationDebugOptions(
    denylisted_nodes=suspected_layers)

#restore model
cp_dir = '/aidata/anders/objects/landmarks/AFLW/archive_model/add_capacity'
model = tf.keras.models.load_model(cp_dir)
inputs = tf.keras.Input(shape=(192, 320, 3), batch_size=1, name='input')
x = model(inputs, training=False)
model = tf.keras.Model(inputs, x)

#convert model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.target_spec.supported_types = [tf.int8, tf.uint8]
converter.representative_dataset = representative_dataset_gen
debugger = tf.lite.experimental.QuantizationDebugger(
    converter=converter,
    debug_dataset=representative_dataset_gen,
    debug_options=debug_options)

selective_quantized_model = debugger.get_nondebug_quantized_model()
#export model
with open(
        '/aidata/anders/objects/landmarks/AFLW/archive_model/add_capacity/INT8.tflite',
        'wb') as f:
    f.write(selective_quantized_model)

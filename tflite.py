import tensorflow as tf
import numpy as np
from models.backbone.hardnet import *
from pprint import pprint
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


def set_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for i in range(len(gpus)):
        tf.config.experimental.set_memory_growth(gpus[i], True)


# set_gpu()
saved_path = "/aidata/anders/objects/tflite"

backbone = HardNet39(input_shape=(192, 320, 3),
                     pooling='avg_pool',
                     kernel_initializer='he_uniform')


@tf.function
def run_model(backbone):
    imgs = tf.constant(0., shape=(1, 192, 320, 3), dtype=tf.float32)
    outputs = backbone(imgs, training=False)
    return outputs


# outputs = run_model(backbone)
# tf.saved_model.save(backbone, '/aidata/anders/objects/tflite')
converter = tf.lite.TFLiteConverter.from_saved_model(
    "/aidata/anders/objects/tflite/base")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
interpreter = tf.lite.Interpreter(model_content=tflite_model)
signatures = interpreter.get_signature_list()
TFLITE_FILE_PATH = '/aidata/anders/objects/tflite/base/model.tflite'
with open(TFLITE_FILE_PATH, 'wb') as f:
    f.write(tflite_model)
interpreter = tf.lite.Interpreter(
    model_path='/aidata/anders/objects/tflite/base/model.tflite')
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

for _ in range(100):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    start_time = time.time()
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    end_time = time.time()
    print("%.5f" % (end_time - start_time))
print(output_data.shape)
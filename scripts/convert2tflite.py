import tensorflow as tf
import os
import numpy as np
import cv2
from glob import glob

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


def representative_dataset_gen():
    path = "/aidata/anders/data_collection/okay/demo_test/imgs"
    img_paths = list(glob(os.path.join(path, "*.jpg")))
    for path in img_paths:
        img = cv2.imread(path)
        # img = cv2.resize(img, (320, 320))
        img = cv2.resize(img, (480, 640))
        # img = img / 255.
        # img = np.reshape(img, (-1, 320, 320, 3))
        img = np.reshape(img, (-1, 480, 640, 3))
        img = img.astype(np.float32)
        yield [img]


class PreProcessModel(tf.keras.Model):

    def __init__(self, size: tuple, *args, **kwargs):
        super(PreProcessModel, self).__init__(*args, **kwargs)
        self.size = size
        self.scale = 1 / 255
        self.resize = tf.image.resize

    def __call__(self, x):
        x = self.resize(images=x,
                        size=self.size,
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x *= self.scale
        return x


# input_width = 1280
# input_height = 720
# inputs = tf.constant(0., shape=(1, input_height, input_width, 3))
# input_shape = (input_height, input_width, 3)
# image_inputs = tf.keras.Input(shape=input_shape, name='image_inputs')
# preprocess = PreProcessModel((320, 320))
# x = preprocess(image_inputs)
cp_dir = "/aidata/anders/data_collection/okay/total/archives/whole/one_branch"
restored_model = tf.keras.models.load_model(cp_dir)
# preds = restored_model(x, training=False)
# finalModel = tf.keras.Model(image_inputs, preds)
# _ = finalModel(inputs)
converter = tf.lite.TFLiteConverter.from_keras_model(restored_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [
#     tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS
# ]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
# converter.inference_input_type = tf.uint8
# converter.target_spec.supported_types = [tf.int8, tf.uint8]
converter.target_spec.supported_types = [tf.float32]
# converter.representative_dataset = representative_dataset_gen
tflite_model = converter.convert()
model_path = os.path.join(cp_dir, "./tflite/mtfd_FP32.tflite")
with open(model_path, 'wb') as f:
    f.write(tflite_model)

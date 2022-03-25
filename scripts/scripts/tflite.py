import os
import cv2
import numpy as np
import commentjson
import tensorflow as tf
import sys
from pathlib import Path
import argparse
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
from behavior_predictor.core.base import Base
from monitor import logger


class OffsetPostModel(tf.keras.Model):
    def __init__(self, pred_model, n_objs, top_k_n, kp_thres, nms_iou_thres,
                 resize_shape, *args, **kwargs):
        super(OffsetPostModel, self).__init__(*args, **kwargs)
        self.pred_model = pred_model
        self.n_objs = n_objs
        self.top_k_n = top_k_n
        self.kp_thres = kp_thres
        self.nms_iou_thres = nms_iou_thres
        self.resize_shape = tf.cast(resize_shape, tf.float32)
        self.origin_shapes = tf.constant([720., 1280.], shape=(1, 2))
        self.base = Base()

    @tf.function
    def call(self, x, training=False):
        obj_heat_map, obj_offset_map, obj_size_maps = x["obj_heat_map"], x[
            "obj_offset_map"], x["obj_size_maps"]
        batch_size = tf.shape(obj_heat_map)[0]
        self.resize_ratio = tf.cast(self.origin_shapes / self.resize_shape,
                                    tf.dtypes.float32)
        outputs = self._obj_detect(batch_size, obj_heat_map, obj_offset_map,
                                   obj_size_maps)
        return outputs

    @tf.function
    def _obj_detect(self, batch_size, hms, offset_maps, size_maps):
        hms = self.base.apply_max_pool(hms)
        b, h, w, c = [tf.shape(hms)[i] for i in range(4)]
        b_coors = self.base.top_k_loc(hms, self.top_k_n, h, w, c)
        b_lnmks = b_coors[:, 1:, ...]
        b_coors = b_coors[:, :1, ...]

        res_c = c - 1
        c = c - res_c
        b_idxs = tf.tile(
            tf.range(0, b, dtype=tf.int32)[:, tf.newaxis, tf.newaxis,
                                           tf.newaxis],
            [1, c, self.top_k_n, 1],
        )
        nose_scores = tf.gather_nd(hms[..., 1],
                                   tf.concat([b_idxs, b_lnmks], axis=-1))
        nose_mask = nose_scores > 0.5
        b_lnmks = b_lnmks[nose_mask]
        b_nose_scores = nose_scores[nose_mask]
        n, d = [tf.shape(b_lnmks)[i] for i in range(2)]
        b_lnmks = tf.reshape(b_lnmks, (batch_size, n, 1, d))
        b_nose_scores = tf.reshape(b_nose_scores, (batch_size, n))
        b_infos = tf.concat([b_idxs, b_coors], axis=-1)
        # only pick bbox
        b_size_vals = tf.gather_nd(size_maps, b_infos)

        b_c_idxs = tf.tile(
            tf.range(0, c, dtype=tf.int32)[tf.newaxis, :, tf.newaxis,
                                           tf.newaxis],
            [b, 1, self.top_k_n, 1])

        b_infos = tf.concat([b_infos, b_c_idxs], axis=-1)
        b_scores = tf.gather_nd(hms, b_infos)

        b_centers = tf.cast(b_coors, tf.float32)
        b_tls = (b_centers - b_size_vals / 2)
        b_brs = (b_centers + b_size_vals / 2)
        # clip value
        b_br_y = b_brs[..., 0]
        b_br_x = b_brs[..., 1]
        b_tls = tf.where(b_tls < 0., 0., b_tls)

        b_br_y = tf.where(b_brs[..., :1] > self.resize_shape[0] - 1.,
                          self.resize_shape[0] - 1., b_brs[..., :1])
        b_br_x = tf.where(b_brs[..., -1:] > self.resize_shape[1] - 1.,
                          self.resize_shape[1] - 1., b_brs[..., -1:])
        b_brs = tf.concat([b_br_y, b_br_x], axis=-1)

        b_bboxes = tf.concat([b_tls, b_brs], axis=-1)

        b_bboxes = self.base.resize_back(b_bboxes, self.resize_ratio)

        b_scores = tf.transpose(b_scores, [0, 2, 1])
        # B N C D
        b_bboxes = tf.concat([b_bboxes, b_scores[..., None]], axis=-1)

        mask = b_scores > self.kp_thres
        index = tf.where(mask == True)
        n = tf.shape(index)[0]
        d = tf.shape(b_bboxes)[-1]
        c_idx = tf.tile(tf.range(d)[None, :, None], [n, 1, 1])
        index = tf.cast(tf.tile(index[:, tf.newaxis, :], [1, d, 1]), tf.int32)
        index = tf.concat([index, c_idx], axis=-1)
        b_offset_vect = self._offset_vec_nose(batch_size, b_lnmks, offset_maps)
        b_lnmks = tf.cast(b_lnmks, tf.float32) * self.resize_ratio[:, None,
                                                                   None, :]
        b_ENM = b_lnmks - b_offset_vect
        b_lnmks = tf.concat([b_ENM[:, :, :2], b_lnmks, b_ENM[:, :, 2:]],
                            axis=-2)

        lnmks = tf.squeeze(b_lnmks, axis=0)
        nose_scores = tf.squeeze(b_nose_scores, axis=0)

        bboxes = b_bboxes[mask]
        box_ouputs = tf.constant(-1., shape=[15, 5])
        lnmk_ouputs = tf.constant(-1., shape=[15, 11])
        n = tf.shape(bboxes)[0]
        box_ouputs = tf.tensor_scatter_nd_update(
            box_ouputs,
            tf.range(n, dtype=tf.int32)[:, None], bboxes)
        m = tf.shape(lnmks)[0]
        lnmks = tf.reshape(lnmks, [-1, 10])
        lnmks = tf.concat([lnmks, nose_scores[:, None]], axis=-1)
        lnmk_ouputs = tf.tensor_scatter_nd_update(
            lnmk_ouputs,
            tf.range(m, dtype=tf.int32)[:, None], lnmks)
        outputs = tf.concat([box_ouputs, lnmk_ouputs], axis=-1)
        return outputs

    @tf.function
    def _offset_vec_nose(self, batch_size, b_lnmks, offset_maps):
        b_nose_lnmks = tf.squeeze(b_lnmks, axis=-2)
        _, n, d = [tf.shape(b_lnmks)[i] for i in range(3)]
        b_idx = tf.tile(
            tf.range(batch_size, dtype=tf.int32)[:, None, None], [1, n, 1])
        b_idx = tf.concat([b_idx, b_nose_lnmks], axis=-1)
        b_offset_vect = tf.gather_nd(offset_maps, b_idx)
        b_offset_vect = tf.reshape(b_offset_vect, [batch_size, n, 4, 2])
        b_lnmks = tf.cast(b_nose_lnmks[:, :, None, :], tf.float32)
        b_offset_vect = b_offset_vect * self.resize_ratio[:, None, None, :]
        return b_offset_vect


def convert(config, save_path, is_convert):
    test_imgs = tf.constant(0., shape=(1, 192, 320, 3), dtype=tf.float32)
    if is_convert:
        config = config['predictor']
        logger.info('Load and Restore centernet offset model as first')
        _model = tf.keras.models.load_model(config['pb_path'])
        _post_model = OffsetPostModel(
            _model, config['n_objs'], config['top_k_n'],
            config['kp_thres'], config['nms_iou_thres'],
            tf.cast(config['resize_size'], tf.float32))

        img_inputs = tf.keras.Input(shape=(192, 320, 3))
        model_outs = _model(img_inputs)
        ouputs = _post_model(model_outs, training=False)
        _post_model = tf.keras.Model(img_inputs, ouputs)
        _ = _post_model(test_imgs)
        # try:
        logger.info('Porcessing tflite')
        converter = tf.lite.TFLiteConverter.from_keras_model(_post_model)
        # Save the model.
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float32]

        tflite_model = converter.convert()
        with open(save_path, 'wb') as f:
            f.write(tflite_model)
        logger.info('sucessful')
        # except:
        #     logger.error('Convert failed in centernet offset tflite')
        # logger.info('Load and Restore centernet offset model as second')
        cls_path = '/aidata/anders/objects/landmarks/eye_wild/total/archive_model/max'
        _eye_model = tf.keras.models.load_model(cls_path)
        batch_size = 2
        img_size = (26, 34, 3)
        img_inputs = tf.keras.Input(shape=img_size, batch_size=batch_size)
        outs = _eye_model(img_inputs, training=False)
        _eye_model = tf.keras.Model(img_inputs, outs)
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(_eye_model)
            # Save the model.
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            save_path = save_path.replace("model1.tflite", "model2.tflite")
            tflite_model = converter.convert()
            with open(save_path, 'wb') as f:
                f.write(tflite_model)
            logger.info('sucessful')
        except:
            logger.error('Convert failed in eye classification tflite')

    else:
        # save_path = save_path.replace("model1.tflite", "model2.tflite")
        interpreter = tf.lite.Interpreter(model_path=save_path)
        interpreter.allocate_tensors()
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]
        test_imgs = tf.constant(0., shape=(1, 26, 34, 3), dtype=tf.float32)
        for _ in range(10000000000000):
            start_time = time.time()
            interpreter.set_tensor(input_index, test_imgs)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_index)
            print("%.10f" % (time.time() - start_time))
        print(predictions)


def parse_config():
    parser = argparse.ArgumentParser('Argparser for model image generate')
    parser.add_argument('--config')
    parser.add_argument(
        '--save_path',
        default="/aidata/anders/objects/tflite/offset/model1.tflite")
    parser.add_argument('--is_convert', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_config()
    print('Result imgs generating')
    print(f"Use following config to produce tensorflow graph: {args.config}.")
    assert os.path.isfile(args.config)
    with open(args.config) as f:
        config = commentjson.loads(f.read())
    convert(config, args.save_path, args.is_convert)

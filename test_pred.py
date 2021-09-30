import tensorflow as tf
import numpy as np


class BehaviorPred:
    # def __init__(self, config=None):
    # self.config = config
    # if self.config is not None:
    #     self.batch = self.config.batch
    #     self.model_dir = self.config.path
    #     model = tf.keras.models.load_model(self.model_dir)
    def post_model(self, preds):
        (pred_bboxes, pred_cls) = preds
        print(pred_bboxes)
        xxxx
        return dets

    def pred(self, post_model):

        feat_bbox_0 = np.load("../nanodet/feat_bbox_0.npy")
        feat_bbox_1 = np.load("../nanodet/feat_bbox_1.npy")
        feat_bbox_2 = np.load("../nanodet/feat_bbox_2.npy")
        feat_cls_1 = np.load("../nanodet/feat_cls_1.npy")
        feat_cls_0 = np.load("../nanodet/feat_cls_0.npy")
        feat_cls_2 = np.load("../nanodet/feat_cls_2.npy")
        pred_bboxes = [feat_bbox_0, feat_bbox_1, feat_bbox_2]
        pred_cls = [feat_cls_1, feat_cls_0, feat_cls_2]
        preds = (pred_bboxes, pred_cls)
        rets = self.post_model(preds)
        # rets = post_model(img, training=False)
        return rets


predictor = BehaviorPred()
rets = predictor.pred(None)

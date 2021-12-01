import tensorflow as tf
from ..utils import ConvBlock
from pprint import pprint

conv_mode = 'sp_conv2d'


class Head(tf.keras.Model):
    def __init__(self, config, *args, **kwargs):
        super(Head, self).__init__(*args, **kwargs)

        self.config = config
        self.head_cfg = self.config.head
        self.pred_config = self.config.head.pred_layer
        self.task = 'landmarks' if 'num_landmarks' in self.pred_config.keys(
        ) else 'heatmaps'
        if self.task == 'landmarks':
            self.flatten_layer = tf.keras.layers.Flatten()
            self.d_model = self.head_cfg.d_model
            # self.init_layer = tf.keras.initializers.HeNormal()
            self.init_layer = tf.keras.initializers.HeUniform()
            self.num_lnmks = self.pred_config.num_landmarks
            # self.euler_conv3x3 = ConvBlock(filters=self.d_model // 2,
            #                                kernel_size=3,
            #                                strides=1,
            #                                conv_mode=conv_mode,
            #                                kernel_initializer=self.init_layer,
            #                                norm_method="bn",
            #                                activation="relu")
            # self.euler_angle_embed = tf.keras.models.Sequential([
            #     tf.keras.layers.Dense(self.d_model,
            #                           activation='relu',
            #                           kernel_initializer=self.init_layer),
            #     tf.keras.layers.Dense(self.d_model,
            #                           activation='relu',
            #                           kernel_initializer=self.init_layer),
            #     tf.keras.layers.Dense(3,
            #                           activation=None,
            #                           kernel_initializer=self.init_layer)
            # ])
            # (x1, y1, x2, y2)
            self.lnmk_conv3x3 = ConvBlock(filters=self.d_model // 2,
                                          kernel_size=3,
                                          strides=1,
                                          conv_mode=conv_mode,
                                          kernel_initializer=self.init_layer,
                                          norm_method="bn",
                                          activation="relu")
            self.lnmk_embed = tf.keras.models.Sequential([
                tf.keras.layers.Dense(self.d_model,
                                      activation='relu',
                                      kernel_initializer=self.init_layer),
                tf.keras.layers.Dense(self.d_model,
                                      activation='relu',
                                      kernel_initializer=self.init_layer),
                tf.keras.layers.Dense(self.num_lnmks * 2,
                                      activation='sigmoid',
                                      kernel_initializer=self.init_layer)
            ])

        elif self.task == 'heatmaps':
            self.conv = {}
            self.out_tran_dims = 32
            # self.f_aug_layer = AugmentationLayer(self.out_tran_dims)
            for k in self.pred_config.keys():
                pred_branch = self.pred_config[k]
                for info in pred_branch:
                    branch_name = info['name']
                    pred_out_dims = info['out_dims']
                    if 'heat' in branch_name:
                        self.conv[branch_name] = [
                            ConvBlock(filters=self.out_tran_dims,
                                      kernel_size=3,
                                      use_bias=False,
                                      norm_method='bn',
                                      activation='relu',
                                      conv_mode=conv_mode),
                            ConvBlock(filters=pred_out_dims,
                                      kernel_size=1,
                                      activation='sigmoid',
                                      use_bias=False,
                                      norm_method=None,
                                      name=branch_name)
                        ]
                    elif 'size' in branch_name:
                        self.conv[branch_name] = [
                            ConvBlock(filters=self.out_tran_dims,
                                      kernel_size=3,
                                      use_bias=False,
                                      norm_method='bn',
                                      activation='relu',
                                      conv_mode=conv_mode),
                            ConvBlock(filters=pred_out_dims,
                                      kernel_size=1,
                                      use_bias=False,
                                      norm_method=None,
                                      name=branch_name)
                        ]
                    elif 'offset' in branch_name:
                        self.conv[branch_name] = [
                            ConvBlock(filters=self.out_tran_dims,
                                      kernel_size=3,
                                      use_bias=True,
                                      norm_method='bn',
                                      activation='relu'),
                            ConvBlock(filters=pred_out_dims,
                                      kernel_size=1,
                                      norm_method=None,
                                      activation='relu',
                                      name=None)
                        ]
                    elif 'embed' in branch_name:
                        self.conv[branch_name] = [
                            tf.keras.layers.AveragePooling2D(pool_size=(3, 3),
                                                             strides=1,
                                                             padding='same'),
                            ConvBlock(filters=self.out_tran_dims,
                                      kernel_size=3,
                                      use_bias=True,
                                      norm_method='bn',
                                      activation='relu'),
                            ConvBlock(filters=pred_out_dims,
                                      kernel_size=1,
                                      norm_method=None,
                                      activation=None,
                                      name=branch_name)
                        ]

    @tf.function
    def call(self, x):
        pred_branches = {}
        if self.task == "landmarks":
            lnmk_embeddings = self.lnmk_embed(
                self.flatten_layer(self.lnmk_conv3x3(x)))
            # euler_angle_embeddings = self.euler_angle_embed(
            #     self.flatten_layer(self.euler_conv3x3(x)))
            # pred_branches = {
            #     "landmarks": lnmk_embeddings,
            #     'euler_angles': euler_angle_embeddings
            # }
            pred_branches = {"landmarks": lnmk_embeddings}

        elif self.task == "heatmaps":
            embedding_tmp = []
            offset_tmp = []
            for k in self.pred_config.keys():
                pred_branch = self.pred_config[k]
                for i, info in enumerate(pred_branch):
                    branch_name = info['name']
                    if 'embed' in branch_name:
                        z = self.conv[branch_name][0](aug_feat)
                        z = self.conv[branch_name][1](z)
                        z = self.conv[branch_name][2](z)
                        embedding_tmp.append(z)
                    elif 'offset' in branch_name:
                        z = self.conv[branch_name][0](x)
                        z = self.conv[branch_name][1](z)
                        offset_tmp.append(z)
                        if i == len(pred_branch) - 1:
                            offset = tf.concat(offset_tmp, axis=-1)
                            aug_feat = self.f_aug_layer(offset, x)
                            pred_branches['offset_maps'] = offset
                    else:
                        z = self.conv[branch_name][0](x)
                        pred_branches[branch_name] = self.conv[branch_name][1](
                            z)
                    if 'heat' in branch_name:
                        pred_branches[branch_name] = tf.clip_by_value(
                            pred_branches[branch_name], 1e-4, 1 - 1e-4)
            # pred_branches['embed_maps'] = tf.concat(embedding_tmp, axis=-1)
        return pred_branches

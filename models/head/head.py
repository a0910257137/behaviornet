import tensorflow as tf
from ..utils import ConvBlock
from pprint import pprint

conv_mode = 'sp_conv2d'
norm_method = 'bn'


class Head(tf.keras.Model):

    def __init__(self, config, *args, **kwargs):
        super(Head, self).__init__(*args, **kwargs)

        self.config = config
        self.head_cfg = self.config.head
        self.pred_config = self.config.head.pred_layer
        self.head_keys = [
            'offset_map_LE', 'offset_map_RE', 'offset_map_LM', 'offset_map_RM'
        ]
        self.task = 'landmarks' if 'num_landmarks' in self.pred_config.keys(
        ) else 'heatmaps'

        self.conv = {}
        self.out_tran_dims = 32
        for k in self.pred_config.keys():
            pred_branch = self.pred_config[k]
            for info in pred_branch:
                branch_name = info['name']
                pred_out_dims = info['out_dims']
                if 'heat' in branch_name:
                    self.conv[branch_name] = [
                        ConvBlock(filters=self.out_tran_dims,
                                  kernel_size=3,
                                  use_bias=True,
                                  norm_method=norm_method,
                                  conv_mode=conv_mode,
                                  activation='relu',
                                  name='heat_conv3x3'),
                        ConvBlock(filters=pred_out_dims,
                                  kernel_size=1,
                                  activation='sigmoid',
                                  use_bias=True,
                                  norm_method=None,
                                  name=branch_name)
                    ]
                elif 'size' in branch_name:
                    self.conv[branch_name] = [
                        ConvBlock(filters=self.out_tran_dims,
                                  kernel_size=3,
                                  use_bias=True,
                                  norm_method=norm_method,
                                  conv_mode=conv_mode,
                                  activation='relu',
                                  name='size_conv3x3'),
                        ConvBlock(filters=pred_out_dims,
                                  kernel_size=1,
                                  use_bias=True,
                                  norm_method=None,
                                  name=branch_name)
                    ]
                elif 'offset' in branch_name:
                    self.conv[branch_name] = [
                        ConvBlock(filters=self.out_tran_dims,
                                  kernel_size=3,
                                  use_bias=True,
                                  norm_method=norm_method,
                                  conv_mode=conv_mode,
                                  activation='relu',
                                  name='offset_conv3x3'),
                        ConvBlock(filters=2,
                                  kernel_size=1,
                                  use_bias=True,
                                  norm_method=None,
                                  activation=None,
                                  name='offset_conv1x1_LE'),
                        ConvBlock(filters=2,
                                  kernel_size=1,
                                  use_bias=True,
                                  norm_method=None,
                                  activation=None,
                                  name='offset_conv1x1_RE'),
                        ConvBlock(filters=2,
                                  kernel_size=1,
                                  use_bias=True,
                                  norm_method=None,
                                  activation=None,
                                  name='offset_conv1x1_LM'),
                        ConvBlock(filters=2,
                                  kernel_size=1,
                                  use_bias=True,
                                  norm_method=None,
                                  activation=None,
                                  name='offset_conv1x1_RM')
                    ]
                elif 'embed' in branch_name:
                    self.conv[branch_name] = [
                        tf.keras.layers.AveragePooling2D(pool_size=(3, 3),
                                                         strides=1,
                                                         padding='same'),
                        ConvBlock(filters=self.out_tran_dims,
                                  kernel_size=3,
                                  use_bias=True,
                                  conv_mode=conv_mode,
                                  norm_method='bn',
                                  activation='relu'),
                        ConvBlock(filters=pred_out_dims,
                                  kernel_size=1,
                                  norm_method=None,
                                  activation=None,
                                  name=branch_name)
                    ]
                elif 'param' in branch_name:
                    self.conv[branch_name] = [
                        ConvBlock(filters=self.out_tran_dims,
                                  kernel_size=3,
                                  use_bias=True,
                                  conv_mode=conv_mode,
                                  norm_method='bn',
                                  activation='relu'),
                        ConvBlock(filters=pred_out_dims,
                                  kernel_size=3,
                                  norm_method=None,
                                  activation=None,
                                  name=branch_name)
                    ]

    @tf.function
    def call(self, x):
        pred_branches = {}
        for k in self.pred_config.keys():
            pred_branch = self.pred_config[k]
            for info in pred_branch:
                branch_name = info['name']
                if 'offset' in branch_name:
                    z = self.conv[branch_name][0](x)
                    for i, key in enumerate(self.head_keys):
                        pred_branches[key] = self.conv[branch_name][i + 1](z)
                elif 'size' in branch_name:
                    z = self.conv[branch_name][0](x)
                    pred_branches[branch_name] = self.conv[branch_name][1](z)
                elif 'param' in branch_name:
                    z = self.conv[branch_name][0](x)
                    pred_branches[branch_name] = self.conv[branch_name][1](z)
                elif 'heat' in branch_name:
                    z = self.conv[branch_name][0](x)
                    z = self.conv[branch_name][1](z)
                    pred_branches[branch_name] = tf.clip_by_value(
                        z, 1e-4, 1 - 1e-4)
        return pred_branches

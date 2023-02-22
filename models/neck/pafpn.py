import tensorflow as tf
from ..utils.conv_module import *
import warnings


class PAFPN(tf.keras.Model):

    def __init__(self,
                 config,
                 end_level=-1,
                 relu_before_extra_convs=False,
                 extra_convs_on_inputs=True,
                 no_norm_on_lateral=False,
                 **kwargs):
        super(PAFPN, self).__init__(extra_convs_on_inputs, **kwargs)
        # print('-' * 100)
        self.in_channels = config.in_channels
        out_channels = config.out_channels
        self.num_outs = config.num_outs
        start_level = config.start_level
        self.add_extra_convs = config.add_extra_convs
        self.num_ins = len(self.in_channels)
        self.no_norm_on_lateral = no_norm_on_lateral

        self.relu_before_extra_convs = relu_before_extra_convs
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert self.num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(self.in_channels)
            assert self.num_outs == end_level - start_level

        self.start_level = start_level
        self.end_level = end_level

        assert isinstance(self.add_extra_convs, (str, bool))
        if isinstance(self.add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert self.add_extra_convs in ('on_input', 'on_lateral',
                                            'on_output')
        elif self.add_extra_convs:  # True
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'
        self.lateral_convs = []
        self.fpn_convs = []

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvBlock(filters=out_channels,
                               kernel_size=1,
                               strides=1,
                               norm_method=None,
                               activation=None,
                               use_bias=True)
            fpn_conv = ConvBlock(filters=out_channels,
                                 kernel_size=3,
                                 strides=1,
                                 norm_method=None,
                                 activation=None,
                                 use_bias=True)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.downsample_convs, self.pafpn_convs = [], []
        for i in range(self.start_level + 1, self.backbone_end_level):
            d_conv = ConvBlock(filters=out_channels,
                               kernel_size=3,
                               strides=2,
                               norm_method=None,
                               activation=None,
                               use_bias=True)
            pafpn_conv = ConvBlock(filters=out_channels,
                                   kernel_size=3,
                                   strides=1,
                                   norm_method=None,
                                   activation=None,
                                   use_bias=True)
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)
        # add extra conv layers (e.g., RetinaNet)

        extra_levels = self.num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvBlock(filters=out_channels,
                                           kernel_size=3,
                                           strides=2,
                                           norm_method=None,
                                           activation=None,
                                           use_bias=True)
                self.fpn_convs.append(extra_fpn_conv)

        self.resize = tf.image.resize
        self.max_pool2d = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                                    strides=1,
                                                    padding='same')
        self.relu = self.act = tf.keras.layers.Activation(activation='relu')

    def call(self, x):
        assert len(x) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(x[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # build top-down path
        used_backbone_levels = len(laterals)

        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[1:3]

            laterals[i - 1] += self.resize(
                images=laterals[i],
                size=prev_shape,
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            )

        # build outputs
        # part 1: from original levels
        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            inter_outs[i + 1] += self.downsample_convs[i](inter_outs[i])
        outs = []
        outs.append(inter_outs[0])
        outs.extend([
            self.pafpn_convs[i - 1](inter_outs[i])
            for i in range(1, used_backbone_levels)
        ])
        # part 3: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(self.max_pool2d(outs[-1]))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    orig = x[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                elif self.add_extra_convs == 'on_lateral':
                    outs.append(self.fpn_convs[used_backbone_levels](
                        laterals[-1]))
                elif self.add_extra_convs == 'on_output':
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                else:
                    raise NotImplementedError
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](self.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

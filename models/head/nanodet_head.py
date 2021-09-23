import tensorflow as tf
from .gfl_base import GFLBase
from ..utils.conv_module import ConvBlock
from pprint import pprint


class NanoDetHead(GFLBase):
    def __init__(self, config, **kwargs):
        super(NanoDetHead, self).__init__(config, **kwargs)
        self.config = config
        self.stacked_convs = self.config.head.stacked_convs
        self.strides = self.config.head.strides
        self.feat_channels = 128
        self.reg_max = self.config.head.reg_max
        self.cls_convs = []
        self.reg_convs = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvBlock(filters=self.feat_channels,
                          kernel_size=3,
                          stride=1,
                          activation='hs'))
            self.reg_convs.append(
                ConvBlock(filters=self.feat_channels,
                          kernel_size=3,
                          stride=1,
                          activation='hs'))
        self.gfl_cls = ConvBlock(filters=self.cls_out_channels,
                                 kernel_size=3,
                                 stride=1,
                                 activation='hs')
        self.gfl_reg = ConvBlock(filters=4 * (self.reg_max + 1),
                                 kernel_size=3,
                                 stride=1,
                                 activation='hs')

    def call(self):
        return

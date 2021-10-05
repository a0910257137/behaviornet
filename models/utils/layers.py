import tensorflow as tf


class MBConvBlock(tf.keras.layers.Layer):
    """A class of MBConv: Mobile Inverted Residual Bottleneck.

  Attributes:
    endpoints: dict. A list of internal tensors.
  """
    def __init__(self, input_filter, se_ratio, local_pooling=False, name=None):
        """Initializes a MBConv block.

    Args:
      block_args: BlockArgs, arguments to create a Block.
      mconfig: GlobalParams, a set of global parameters.
      name: layer name.
    """
        super().__init__(name=name)
        self._local_pooling = local_pooling
        self.se_ratio = se_ratio
        self._act = 'silu'
        self._has_se = (self.se_ratio is not None and 0 < self.se_ratio <= 1)
        self.endpoints = None
        # Builds the block accordings to arguments.
        self._build()

    def _build(self):
        """Builds block according to the arguments."""

        if self._block_args.expand_ratio != 1:
            tf.keras.layers.Conv2D()

        mconfig = self._mconfig
        filters = self._block_args.input_filters * self._block_args.expand_ratio
        kernel_size = self._block_args.kernel_size

        # Expansion phase. Called if not using fused convolutions and expansion
        # phase is necessary.
        if self._block_args.expand_ratio != 1:
            self._expand_conv = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=1,
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                data_format=self._data_format,
                use_bias=False,
                name=get_conv_name())
            self._norm0 = utils.normalization(mconfig.bn_type,
                                              axis=self._channel_axis,
                                              momentum=mconfig.bn_momentum,
                                              epsilon=mconfig.bn_epsilon,
                                              groups=mconfig.gn_groups,
                                              name=get_norm_name())

        # Depth-wise convolution phase. Called if not using fused convolutions.
        self._depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=self._block_args.strides,
            depthwise_initializer=conv_kernel_initializer,
            padding='same',
            data_format=self._data_format,
            use_bias=False,
            name='depthwise_conv2d')

        self._norm1 = utils.normalization(mconfig.bn_type,
                                          axis=self._channel_axis,
                                          momentum=mconfig.bn_momentum,
                                          epsilon=mconfig.bn_epsilon,
                                          groups=mconfig.gn_groups,
                                          name=get_norm_name())

        if self._has_se:
            num_reduced_filters = max(
                1,
                int(self._block_args.input_filters *
                    self._block_args.se_ratio))
            self._se = SE(self._mconfig,
                          num_reduced_filters,
                          filters,
                          name='se')
        else:
            self._se = None

        # Output phase.
        filters = self._block_args.output_filters
        self._project_conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=1,
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            data_format=self._data_format,
            use_bias=False,
            name=get_conv_name())
        self._norm2 = utils.normalization(mconfig.bn_type,
                                          axis=self._channel_axis,
                                          momentum=mconfig.bn_momentum,
                                          epsilon=mconfig.bn_epsilon,
                                          groups=mconfig.gn_groups,
                                          name=get_norm_name())

    def residual(self, inputs, x, training, survival_prob):
        if (self._block_args.strides == 1 and self._block_args.input_filters
                == self._block_args.output_filters):
            # Apply only if skip connection presents.
            if survival_prob:
                x = utils.drop_connect(x, training, survival_prob)
            x = tf.add(x, inputs)

        return x

    def call(self, inputs, training, survival_prob=None):
        """Implementation of call().

    Args:
      inputs: the inputs tensor.
      training: boolean, whether the model is constructed for training.
      survival_prob: float, between 0 to 1, drop connect rate.

    Returns:
      A output tensor.
    """
        logging.info('Block %s input shape: %s (%s)', self.name, inputs.shape,
                     inputs.dtype)
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._act(self._norm0(self._expand_conv(x), training=training))
            logging.info('Expand shape: %s', x.shape)

        x = self._act(self._norm1(self._depthwise_conv(x), training=training))
        logging.info('DWConv shape: %s', x.shape)

        if self._mconfig.conv_dropout and self._block_args.expand_ratio > 1:
            x = tf.keras.layers.Dropout(self._mconfig.conv_dropout)(
                x, training=training)

        if self._se:
            x = self._se(x)

        self.endpoints = {'expansion_output': x}

        x = self._norm2(self._project_conv(x), training=training)
        x = self.residual(inputs, x, training, survival_prob)

        logging.info('Project shape: %s', x.shape)
        return x


class FusedMBConvBlock(MBConvBlock):
    """Fusing the proj conv1x1 and depthwise_conv into a conv2d."""
    def _build(self):
        """Builds block according to the arguments."""
        # pylint: disable=g-long-lambda
        bid = itertools.count(0)
        get_norm_name = lambda: 'tpu_batch_normalization' + ('' if not next(
            bid) else '_' + str(next(bid) // 2))
        cid = itertools.count(0)
        get_conv_name = lambda: 'conv2d' + ('' if not next(cid) else '_' + str(
            next(cid) // 2))
        # pylint: enable=g-long-lambda

        mconfig = self._mconfig
        block_args = self._block_args
        filters = block_args.input_filters * block_args.expand_ratio
        kernel_size = block_args.kernel_size
        if block_args.expand_ratio != 1:
            # Expansion phase:
            self._expand_conv = tf.keras.layers.Conv2D(
                filters,
                kernel_size=kernel_size,
                strides=block_args.strides,
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                use_bias=False,
                name=get_conv_name())
            self._norm0 = utils.normalization(mconfig.bn_type,
                                              axis=self._channel_axis,
                                              momentum=mconfig.bn_momentum,
                                              epsilon=mconfig.bn_epsilon,
                                              groups=mconfig.gn_groups,
                                              name=get_norm_name())

        if self._has_se:
            num_reduced_filters = max(
                1, int(block_args.input_filters * block_args.se_ratio))
            self._se = SE(mconfig, num_reduced_filters, filters, name='se')
        else:
            self._se = None
        # Output phase:
        filters = block_args.output_filters
        self._project_conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size=1 if block_args.expand_ratio != 1 else kernel_size,
            strides=1 if block_args.expand_ratio != 1 else block_args.strides,
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False,
            name=get_conv_name())
        self._norm1 = utils.normalization(mconfig.bn_type,
                                          axis=self._channel_axis,
                                          momentum=mconfig.bn_momentum,
                                          epsilon=mconfig.bn_epsilon,
                                          groups=mconfig.gn_groups,
                                          name=get_norm_name())

    def call(self, inputs, training, survival_prob=None):
        """Implementation of call().

    Args:
      inputs: the inputs tensor.
      training: boolean, whether the model is constructed for training.
      survival_prob: float, between 0 to 1, drop connect rate.

    Returns:
      A output tensor.
    """
        logging.info('Block %s  input shape: %s', self.name, inputs.shape)
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._act(self._norm0(self._expand_conv(x), training=training))
        logging.info('Expand shape: %s', x.shape)

        self.endpoints = {'expansion_output': x}

        if self._mconfig.conv_dropout and self._block_args.expand_ratio > 1:
            x = tf.keras.layers.Dropout(self._mconfig.conv_dropout)(x,
                                                                    training)

        if self._se:
            x = self._se(x)

        x = self._norm1(self._project_conv(x), training=training)
        if self._block_args.expand_ratio == 1:
            x = self._act(x)  # add act if no expansion.

        x = self.residual(inputs, x, training, survival_prob)
        logging.info('Project shape: %s', x.shape)
        return x
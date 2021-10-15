import tensorflow as tf
from .kernel_initializers import KernelInitializers
from ..utils.conv_module import ConvBlock
from ..utils.conv_module import residual_block, conv_1
from box import Box


class Evo(tf.keras.Model):
    def __init__(self, config, *args, **kwargs):
        super(Evo, self).__init__(*args, **kwargs)

    def call(self, inputs, training=None):
        return


def evo(data_in, soft_map_size, settings, stem_dim, name, training):
    fmap = data_in
    with tf.variable_scope(name):
        for set_idx, setting in enumerate(settings):
            if set_idx == 0:
                blk_filters_in = stem_dim
            else:
                blk_filters_in = settings[set_idx - 1]['out_dim']
            for blk_idx in range(setting['repeat']):
                if blk_idx > 0:
                    strides = 1
                    blk_filters_in = setting['out_dim']
                else:
                    strides = setting['stride']
                blk_name = '%i_res_%i' % (set_idx, blk_idx)
                fmap = residual_block(fmap,
                                      setting['out_dim'],
                                      name=blk_name,
                                      training=training,
                                      strides=strides,
                                      kernel_size=setting['kernel_size'])
                # fmap = inverted_resblk(fmap,
                #                        training=training,
                #                        name=blk_name,
                #                        filters_in=blk_filters_in,
                #                        filters_out=setting['out_dim'],
                #                        se_ratio=setting['se_ratio'],
                #                        strides=setting['stride'],
                #                        kernel_size=setting['kernel_size'],
                #                        id_skip=setting['id_skip'])
        for up_idx in range(3):
            fmap = tf.layers.conv2d_transpose(fmap,
                                              128,
                                              kernel_size=[2, 2],
                                              padding='valid',
                                              strides=(2, 2),
                                              name='upsamp_%i' % up_idx)
            if up_idx == 1:
                spatial_atten = self_atten_layer(fmap, 128, 'spatial')
                channel_atten = channel_atten_layer(fmap, 'channel')
                fmap = conv_1(spatial_atten + channel_atten,
                              128,
                              'atten_conv',
                              activation=None,
                              norm_method=None)
    return fmap


def inverted_resblk(inputs,
                    training,
                    activation='swish',
                    drop_rate=0.,
                    name='',
                    filters_in=16,
                    filters_out=16,
                    kernel_size=3,
                    strides=1,
                    expand_ratio=1,
                    se_ratio=0.,
                    id_skip=True,
                    project=True,
                    regularizer=None):

    CONV_KERNEL_INITIALIZER = {
        'class_name': 'VarianceScaling',
        'config': {
            'scale': 2.0,
            'mode': 'fan_out',
            'distribution': 'truncated_normal'
        }
    }
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    actis = {'relu': tf.nn.relu, 'swish': tf.nn.swish}
    # Expansion phase
    filters = int(filters_in * expand_ratio)
    if expand_ratio != 1:
        x = tf.layers.conv2d(inputs,
                             filters,
                             1,
                             padding='same',
                             use_bias=False,
                             kernel_initializer=kernel_initializer,
                             name=name + 'expand_conv')
        x = tf.layers.batch_normalization(x, training=training)
        x = actis[activation](x)
    else:
        x = inputs

    # Depthwise Convolution
    # x = tf.layers.conv2d(x,
    #                      filters,
    #                      kernel_size,
    #                      strides=strides,
    #                      activation=actis[activation],
    #                      padding='same',
    #                      use_bias=False,
    #                      kernel_initializer=kernel_initializer,
    #                      name=name + 'conv')
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        depthwise_initializer=CONV_KERNEL_INITIALIZER,
        kernel_regularizer=regularizer,
        name=name + 'dwconv')(x)
    x = tf.layers.batch_normalization(x, training=training)
    x = actis[activation](x)

    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = tf.keras.layers.GlobalAveragePooling2D(name=name +
                                                    'se_squeeze')(x)
        se = tf.reshape(se, [-1, 1, 1, filters], name=name + 'se_reshape')
        se = tf.layers.conv2d(se,
                              filters_se,
                              1,
                              activation=actis[activation],
                              padding='same',
                              use_bias=False,
                              kernel_initializer=kernel_initializer,
                              name=name + 'se_reduce')
        se = tf.layers.conv2d(se,
                              filters,
                              1,
                              activation=tf.nn.sigmoid,
                              padding='same',
                              use_bias=False,
                              kernel_initializer=kernel_initializer,
                              name=name + 'se_expand')
        x = tf.keras.layers.multiply([x, se], name=name + 'se_excite')

    # Output phase
    if project:
        x = tf.layers.conv2d(x,
                             filters_out,
                             1,
                             padding='same',
                             use_bias=False,
                             kernel_initializer=kernel_initializer,
                             name=name + 'project_conv')
        x = tf.layers.batch_normalization(x, training=training)

    if id_skip and strides == 1 and filters_in == filters_out and project:
        x = x + inputs
    return x


def residual_block(
        data_in,
        out_dim,
        name,
        training,
        strides=1,
        rate=1,
        kernel_size=3,
        identity_skip=False,
        use_bottle=False,
        norm_method='bn',
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer()):
    normalization = tf.identity
    if norm_method == 'bn':
        normalization = partial(tf.layers.batch_normalization,
                                training=training)
    elif norm_method == 'in':
        normalization = tf.contrib.layers.instance_norm
    elif norm_method == 'gn':
        normalization = group_norm

    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    with tf.variable_scope('residual_' + name):
        # skip
        with tf.name_scope('skip_path'):
            if identity_skip:
                skip = tf.identity(data_in)
            else:
                skip = tf.layers.conv2d(data_in,
                                        out_dim,
                                        kernel_size=1,
                                        strides=strides,
                                        use_bias=False,
                                        padding='same',
                                        kernel_regularizer=regularizer,
                                        kernel_initializer=kernel_initializer)
                skip = normalization(skip)

        # main
        with tf.name_scope('main_path'):
            if use_bottle:
                out = tf.layers.conv2d(data_in,
                                       out_dim / 2,
                                       kernel_size=1,
                                       strides=strides,
                                       use_bias=False,
                                       padding='same',
                                       kernel_regularizer=regularizer,
                                       kernel_initializer=kernel_initializer)
                out = normalization(out)
                out = tf.nn.relu6(out)
                out = tf.layers.conv2d(out,
                                       out_dim / 2,
                                       kernel_size=kernel_size,
                                       dilation_rate=2,
                                       use_bias=False,
                                       padding='same',
                                       kernel_regularizer=regularizer,
                                       kernel_initializer=kernel_initializer)
                out = normalization(out)
                out = tf.nn.relu6(out)
                out = tf.layers.conv2d(out,
                                       out_dim,
                                       kernel_size=1,
                                       strides=strides,
                                       use_bias=False,
                                       padding='same',
                                       kernel_regularizer=regularizer,
                                       kernel_initializer=kernel_initializer)
            else:
                out = tf.layers.conv2d(data_in,
                                       out_dim,
                                       kernel_size=kernel_size,
                                       strides=strides,
                                       use_bias=False,
                                       padding='same',
                                       kernel_regularizer=regularizer,
                                       kernel_initializer=kernel_initializer)

                out = normalization(out)
                out = tf.nn.relu6(out)
                out = tf.layers.conv2d(out,
                                       out_dim,
                                       kernel_size=kernel_size,
                                       use_bias=False,
                                       padding='same',
                                       kernel_regularizer=regularizer,
                                       kernel_initializer=kernel_initializer)

            out = normalization(out)
        out = tf.math.add_n([skip, out], name='merge')

        out = tf.nn.relu6(out)
    return out


def aspp_layer(data_in, out_dims, is_train):
    with tf.variable_scope('aspp_layer'):
        # conv1

        out_1 = conv_1(data_in, filters=out_dims, name='conv1')
        out_1 = tf.layers.batch_normalization(out_1, training=is_train)
        # dilate conv6
        aout_6 = tf.layers.conv2d(data_in,
                                  filters=out_dims,
                                  kernel_size=3,
                                  dilation_rate=6,
                                  activation=tf.nn.relu6,
                                  padding='same')
        aout_6 = tf.layers.batch_normalization(aout_6, training=is_train)

        # dilate conv12
        aout_12 = tf.layers.conv2d(data_in,
                                   filters=out_dims,
                                   kernel_size=3,
                                   dilation_rate=12,
                                   activation=tf.nn.relu6,
                                   padding='same')
        aout_12 = tf.layers.batch_normalization(aout_12, training=is_train)

        # dilate conv18
        aout_18 = tf.layers.conv2d(data_in,
                                   filters=out_dims,
                                   kernel_size=3,
                                   dilation_rate=18,
                                   activation=tf.nn.relu6,
                                   padding='same')
        aout_18 = tf.layers.batch_normalization(aout_18, training=is_train)

        # img pooling
        img_pool = tf.reduce_mean(data_in, [1, 2],
                                  name='global_average_pooling',
                                  keepdims=True)
        img_pool = conv_1(img_pool, filters=out_dims, name='gap_conv_1')
        img_pool = tf.image.resize_bilinear(img_pool,
                                            tf.shape(data_in)[1:3],
                                            name='up_sampling')
        img_pool = tf.layers.batch_normalization(img_pool, training=is_train)
        concat_list = [out_1, aout_6, aout_12, aout_18, img_pool]

        aout = tf.concat(concat_list, axis=-1)
        out = conv_1(aout, filters=out_dims, name='out_conv1')
        out = tf.layers.batch_normalization(out, training=is_train)
        return out


def Evo(input_shape, pooling, kernel_initializer):
    kernel_initializer = KernelInitializers().get_initializer(
        kernel_initializer)
    hardnet = Evo(pooling=pooling,
                  arch=39,
                  kernel_initializer=kernel_initializer)
    image_inputs = tf.keras.Input(shape=input_shape, name='image_inputs')
    fmaps = hardnet(image_inputs)

    return tf.keras.Model(image_inputs, fmaps, name='backbone')
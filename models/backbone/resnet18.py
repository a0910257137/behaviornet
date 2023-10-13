import tensorflow as tf
from .kernel_initializers import KernelInitializers
from ..utils.conv_module import ConvBlock

from pprint import pprint


class BasicBlock(tf.keras.layers.Layer):

    def __init__(self,
                 filter_num,
                 stride=1,
                 kernel_initializer=tf.keras.initializers.HeUniform()):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filter_num,
            kernel_size=(3, 3),
            strides=stride,
            padding="same",
            kernel_initializer=kernel_initializer)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filter_num,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            kernel_initializer=kernel_initializer)
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(
                tf.keras.layers.Conv2D(filters=filter_num,
                                       kernel_size=(1, 1),
                                       strides=stride,
                                       kernel_initializer=kernel_initializer))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output


class BottleNeck(tf.keras.layers.Layer):

    def __init__(self,
                 filter_num,
                 stride=1,
                 kernel_initializer=tf.keras.initializers.HeUniform()):
        super(BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filter_num,
            kernel_size=(1, 1),
            strides=1,
            padding='same',
            kernel_initializer=kernel_initializer)

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filter_num,
            kernel_size=(3, 3),
            strides=stride,
            padding='same',
            kernel_initialize=kernel_initializer)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(
            filters=filter_num * 4,
            kernel_size=(1, 1),
            strides=1,
            padding='same',
            kernel_initialize=kernel_initializer)
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.downsample = tf.keras.Sequential()
        self.downsample.add(
            tf.keras.layers.Conv2D(filters=filter_num * 4,
                                   kernel_size=(1, 1),
                                   strides=stride,
                                   kernel_initialize=kernel_initializer))
        self.downsample.add(tf.keras.layers.BatchNormalization())

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output


def make_basic_block_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=1))

    return res_block


def make_bottleneck_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BottleNeck(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BottleNeck(filter_num, stride=1))

    return res_block


class ResNetTypeI(tf.keras.Model):

    def __init__(self, layer_params, kernel_initializer, *args, **kwargs):
        super(ResNetTypeI, self).__init__(*args, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(7, 7),
            strides=2,
            padding="same",
            kernel_initializer=kernel_initializer)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_basic_block_layer(filter_num=64,
                                             blocks=layer_params[0])
        self.layer2 = make_basic_block_layer(filter_num=128,
                                             blocks=layer_params[1],
                                             stride=2)
        self.layer3 = make_basic_block_layer(filter_num=256,
                                             blocks=layer_params[2],
                                             stride=2)
        self.layer4 = make_basic_block_layer(filter_num=512,
                                             blocks=layer_params[3],
                                             stride=2)

    @tf.function
    def call(self, inputs, training=None):
        output = []
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        output.append(x)
        x = self.layer2(x, training=training)
        output.append(x)
        x = self.layer3(x, training=training)
        output.append(x)
        x = self.layer4(x, training=training)
        output.append(x)
        return tuple(output)


def resnet18(config, input_shape, kernel_initializer):
    from keras_flops import get_flops
    kernel_initializer = KernelInitializers().get_initializer(
        kernel_initializer)
    model = ResNetTypeI(layer_params=[2, 2, 2, 2],
                        kernel_initializer=kernel_initializer)
    image_inputs = tf.keras.Input(shape=input_shape, name='image_inputs')
    fmaps = model(image_inputs)
    return tf.keras.Model(image_inputs, fmaps)
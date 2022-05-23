"""
Misc Utility functions
"""
import os
import logging
import datetime
import commentjson
import tensorflow as tf
import commentjson
from .callbacks import EmbeddingMap, LossAndErrorPrintingCallback, CheckpointManagerCallback, Histogram
from monitor import logger


def get_callbacks(config, model, optimizer, train_datasets, test_datasets):
    callbacks = []
    train_writer = tf.summary.create_file_writer(
        os.path.join(config.summary.log_dir, 'train'))
    eval_writer = tf.summary.create_file_writer(
        os.path.join(config.summary.log_dir, 'validation'))
    writers = {'train': train_writer, 'validation': eval_writer}
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_dir = config.model_path
    manager = tf.train.CheckpointManager(checkpoint,
                                         checkpoint_dir,
                                         max_to_keep=5)

    saver_callback = CheckpointManagerCallback(checkpoint,
                                               manager,
                                               model,
                                               directory=checkpoint_dir,
                                               period=1)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=config.summary.log_dir,
        write_graph=False,
        write_images=False,
        update_freq='batch',
        histogram_freq=0,
        profile_batch=0)
    histogram = Histogram(config=config, writers=writers, update_freq=1000)
    # embedding_map = EmbeddingMap(config=config,
    #                              writers=writers,
    #                              train_datasets=train_datasets,
    #                              test_datasets=test_datasets,
    #                              update_freq=1000)
    # cosine_decay_scheduler = WarmUpCosineDecayScheduler(
    #     config.learn_rate, config.epochs, train_datasets)
    callbacks.append([
        saver_callback, tensorboard_callback, histogram,
        LossAndErrorPrintingCallback()
    ])
    return callbacks


def get_logger():
    logger = logging.getLogger("ptsemseg")
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    logger.setLevel(logging.INFO)
    return logger


def load_configger(config_path):
    if not os.path.isfile(config_path):
        raise FileNotFoundError('File %s does not exist.' % config_path)
    with open(config_path, 'r') as fp:
        config = commentjson.loads(fp.read())
    config = AttrDict(config)
    config = set_data_config(config)
    config = set_model_config(config)
    model_config = config.set_immutable()
    return config


def set_data_config(config):
    config.data_reader.epochs = config.epochs
    # add tasks keys in data_reader
    config.data_reader.train_batch_size = config.train_batch_size
    config.data_reader.test_batch_size = config.test_batch_size
    config.data_reader.model_name = config.models.model_name

    # config.data_reader.num_landmarks = config.models.head.pred_layer.num_landmarks
    return config


def set_model_config(config):
    config.models.max_obj_num = config.data_reader.max_obj_num
    config.models.resize_size = config.data_reader.resize_size
    config.models.batch_size = config.train_batch_size
    config.models.train_batch_size = config.train_batch_size
    config.models.test_batch_size = config.test_batch_size
    config.models.lr = config.learn_rate
    return config


class AttrDict(dict):
    IMMUTABLE = '__immutable__'

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)

        self.__dict__[AttrDict.IMMUTABLE] = False

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        elif key in self:
            return self[key]
        else:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        if self.__dict__[AttrDict.IMMUTABLE]:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but AttrDict is immutable'.
                format(key, value))

        if isinstance(value, dict):
            value = AttrDict(value)

        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

    def set_immutable(self, is_immutable=True):
        self.__dict__[AttrDict.IMMUTABLE] = is_immutable

        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.set_immutable(is_immutable)

        for v in self.values():
            if isinstance(v, AttrDict):
                v.set_immutable(is_immutable)

    def is_immutable(self):
        return self.__dict__[AttrDict.IMMUTABLE]

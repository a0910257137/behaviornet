import os
import math
import tensorflow as tf
import argparse
from box import Box
from monitor import logger
from dataset.general import GeneralDataset
from system.restore import Restore
from utils.sys_tools import set_gpu
from models.model_factory import ModelFactory

from utils.tools import get_callbacks, load_configger
from pprint import pprint
from utils.tools import *
from utils.io import *


def train(config, is_restore):
    mirrored_strategy = tf.distribute.MirroredStrategy()
    general_dataset = GeneralDataset(config.data_reader)
    # Read in Training Data
    datasets = general_dataset.get_datasets(mirrored_strategy)
    train_datasets, test_datasets = datasets['train'], datasets['test']
    with mirrored_strategy.scope():
        model, optimizer = ModelFactory(config.models, config.model_path,
                                        config.learn_rate).build_model()
        if is_restore:
            model = Restore(config.model_path).build_restoration(
                model, excluded_layers)
    callbacks = get_callbacks(config, model, optimizer, train_datasets,
                              test_datasets)
    model.fit(train_datasets,
              validation_data=test_datasets,
              epochs=config.epochs,
              workers=10,
              use_multiprocessing=True,
              callbacks=callbacks)


def argparser():
    parser = argparse.ArgumentParser(
        description='Keywords spotting entry point')
    parser.add_argument('--gpus', help='Use gpus for training or not')
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file to use",
    )
    parser.add_argument('--restore',
                        action='store_true',
                        help='Restore pre-trained model in config')
    parser.add_argument('--excluded_layers',
                        nargs='+',
                        help='Exclude layers with input keywords when restore')
    return parser.parse_args()


if __name__ == '__main__':
    args = argparser()
    config = read_commentjson(args.config)
    # config = Box(config)
    set_gpu(args.gpus)
    logger.info(f'Use config: {args.config} to train kws')
    config = load_configger(args.config)
    train(config, args.restore)

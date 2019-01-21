# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 11:16:32 2018

@author: dumle
"""
from datetime import datetime
import os

from models.model_thomas import TemporalClustering, ClusteringLayer_temporal
from utils import factory
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utilities import get_args, logger


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([os.path.join('/scratch/users/thomaslj/autogen_files', config.model.bash_name, config.callbacks.tensorboard_log_dir),
                 os.path.join('/scratch/users/thomaslj/autogen_files', config.model.bash_name, config.callbacks.checkpoint_dir)])

    print('Create the data generator.')
    logger('Creating data generators ...'.format(datetime.now()))
    data_loader = {'train': factory.create("data_loader."+config.data_loader.name)(config, subset='train', shuffle=True),
                   'test': factory.create("data_loader."+config.data_loader.name)(config, subset='test'),
                          'eval': factory.create("data_loader." + config.data_loader.name)(config, subset='eval')}

    print('Create the model.')
    model = TemporalClustering(config)
    logger('Creating the trainer ...'.format(datetime.now()))
    if config.model.num_gpus > 1:
        trainer = factory.create("trainers."+config.trainer.name)(model.parallel_model, data_loader, config)
    else:
        trainer = factory.create("trainers."+config.trainer.name)(model.model, data_loader, config)

    print('Start training the model.')
    trainer.train()
    print('predicting the autoencoder')
    trainer.predict_autoencoder()

    print('Computing the clustering layer.')
    data_loader['train'].model_type = data_loader['test'].model_type = data_loader['eval'].model_type = 'clustering'
    trainer.data_groups['train_data'].model_type = trainer.data_groups['test_data'].model_type = \
        trainer.data_groups['eval_data'].model_type = trainer.eval_data.model_type = trainer.test_data.model_type = \
        trainer.test_data.model_type = 'clustering'
    clust_layer = trainer.clustering_layer()

if __name__ == '__main__':
    main()
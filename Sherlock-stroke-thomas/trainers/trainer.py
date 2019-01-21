# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 10:49:23 2018

@author: dumle
"""

import os

from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

from base.base_train import BaseTrain
from data_loader.DataGenerator import TrainGenerator, TestGenerator, EvalGenerator
from models.model_thomas import TemporalClustering, ClusteringLayer_temporal
from utils.config import process_config
import numpy as np

class TemporalTrainer(BaseTrain):
    def __init__(self, model, data, config):
        super().__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.schedule_fn = self.step_decay
        self.init_callbacks()

    def step_decay(self, epoch, learning_rate):
        lr = self.config.model.initial_lr * (self.config.callbacks.learningrate_decay_rate ** (
                    epoch // self.config.callbacks.learningrate_decay_epochs))
        return lr

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join('/scratch/users/thomaslj/autogen_files', self.config.model.bash_name, self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5'),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=os.path.join('/scratch/users/thomaslj/autogen_files', self.config.model.bash_name, self.config.callbacks.tensorboard_log_dir),
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

        self.callbacks.append(
            EarlyStopping(
                monitor=self.config.callbacks.earlystopping_monitor,
                patience=self.config.callbacks.earlystopping_patience,
                verbose=self.config.callbacks.earlystopping_verbose,
            )
        )


    def train(self):
        history = self.model.fit_generator(
            self.train_data,
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            callbacks=self.callbacks,
            validation_data=self.eval_data,
            workers=self.config.trainer.num_workers,
            use_multiprocessing=self.config.trainer.use_multiprocessing,
            max_queue_size=self.config.trainer.max_queue_size
        )
        self.loss.extend(history.history['loss'])
        # self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        # self.val_acc.extend(history.history['val_acc'])

    def predict_autoencoder(self):
        # predictions for train, eval and test data:
        self.predictions = []
        for data in [self.eval_data, self.test_data]:
            self.predictions.append(self.model.predict_generator(data,
                verbose=self.config.trainer.verbose_training,
                max_queue_size=self.config.trainer.max_queue_size))
        # saving the generated signals for subject ID 0 and epoch ID 0:
        self.sigs = []
        for x in self.predictions:
            self.sigs.append(x[0,:,0,:])
        np.save(os.path.join('/scratch/users/thomaslj/autogen_files', self.config.model.bash_name,
                             self.config.callbacks.checkpoint_dir, 'autoencoder_signals'), self.sigs)

        # extracting the test signal from subject 0, epoch 0:
        x_test = []
        for idx, (x, y) in enumerate(self.test_data):
            # if idx > 0:
            #     break
            x_test.append(x[0,:,0,:])
        np.save(os.path.join('/scratch/users/thomaslj/autogen_files', self.config.model.bash_name,
                             self.config.callbacks.checkpoint_dir, 'test_signal'), x_test)

        # extracting the eval signal from subject 0, epoch 0:
        x_eval = []
        for idx, (x, y) in enumerate(self.eval_data):
            # if idx > 0:
            #     break
            x_eval.append(x[0,:,0,:])
        np.save(os.path.join('/scratch/users/thomaslj/autogen_files', self.config.model.bash_name,
                             self.config.callbacks.checkpoint_dir, 'eval_signal'), x_eval)

    def clustering_layer(self):
        clust_series = ClusteringLayer_temporal.cluster(self, self.train_data, self.model)
        np.save(os.path.join('/scratch/users/thomaslj/autogen_files', self.config.model.bash_name,
                             self.config.callbacks.checkpoint_dir, 'clust_probability'), clust_series)



if __name__ == '__main__':

    config = process_config('/home/users/thomaslj/stroke-thomas/config/param_configs.json')
    data = {'train': TrainGenerator(config),
            'test': TestGenerator(config),
            'eval': EvalGenerator(config)}
    model = TemporalClustering(config)

    trainer = TemporalTrainer(model, data, config)
    print(trainer)
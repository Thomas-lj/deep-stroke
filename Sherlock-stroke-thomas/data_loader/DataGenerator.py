# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 14:06:39 2018

@author: dumle
"""
import os
import pdb

import h5py
import numpy as np
import pandas as pd
from keras.utils import Sequence

from utils.config import process_config


#from os import listdir

class DataGenerator(Sequence): 
    
    def __init__(self, config, subset = 'train', shuffle=False):
        'Initialization of generator'
        self.batch_size = config.data_loader.batch_size[subset]*config.model.num_gpus
#        self.class_weight = {int(k): v for k, v in self.config.trainer.class_weight.items()}
        self.data_dir = config.data_loader.data_dir
        self.data = config.data_loader.data[subset]
        self.df_dir = config.data_loader.df_file
#        self.data_format = data_format
#        self.input_shape = input_shape
        self.channels_to_load = config.data_loader.channels_to_load
        self.number_epochs = config.data_loader.number_epochs        
#        self.num_channels = len(self.channels_to_load)
        self.num_classes = config.data_loader.num_classes
        self.shuffle = shuffle
        self.collect_and_prune_df()
        self.num_subjects = len(self.df)
        self.len_hypnograms = self.df.Length.values
        self.subset = subset
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        nb = int(np.floor(sum(self.len_hypnograms) / self.batch_size))
        return nb
    
    def __getitem__(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch
      
      current_batch_indices = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    
      # Find list of IDs
      #list_IDs_temp = [self.list_IDs[k] for k in indexes]
    
      # Generate data
#      X, y, weights = self.__data_generation(current_batch_indices)
      # To do: implement a double version of X for fit_generator
      X, y = self.__data_generation(current_batch_indices)
      
      return X, X

    def collect_and_prune_df(self):
        """Collects the cohort dataframes and extracts current cohort and subset"""
        cohorts = [x[0] for x in self.data]
        subsets = [x[1] for x in self.data]
        df = pd.concat(
            [pd.read_csv(os.path.join(self.df_dir), index_col=0) for cohort in cohorts],
            ignore_index=True)
        
        # Prune for subsets
        self.df = []
        for c, s in zip(cohorts, subsets):
            if s == 'train':
                self.df.append(df.loc[(df.Cohort == c) & (df.Fold1 == s)])
            if s == 'eval':
                self.df.append(df.loc[(df.Cohort == c) & (df.Fold1 == s)])
            if s == 'test':
                self.df.append(df.loc[(df.Cohort == c) & (df.Fold1 == s)])
        self.df = pd.concat(self.df, ignore_index=True)
        self.df = self.df.dropna(axis=0, how = 'any')
    
    def on_epoch_end(self):
        """Updates the indices of the data after each epoch"""
        self.indexes = [(i, j) for i in np.arange(self.num_subjects) for j in
                        np.arange(self.len_hypnograms[i])]
        # self.indexes = np.arange(self.num_subjects)
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
    def load_h5data_epochs(self, subject_id, epoch_id):
        'Loads the h5 files with the wanted channels to load'
        # loads h5 files of varying number of epochs. One epoch is 5 min:
        with h5py.File(os.path.join(self.data_dir, subject_id + '.hpf5'), "r") as f:
            dset = {k: np.array(f[k][epoch_id, :][np.newaxis, :]) for k in self.channels_to_load}
            darray = np.array(list(dset.values()))
            #label = 1 is stroke, 0 is healthy
            attributes = f.attrs["group"]
        return darray, attributes 
            
        
    def __data_generation(self, batch_files):

        X = [[]] * self.batch_size
        Y = [[]] * self.batch_size
        # X = np.empty((self.batch_size, *self.input_shape))
        # y = np.empty((self.batch_size, self.num_classes), dtype=int)

        # Generate data
        for i, (subject_id, epoch_id) in enumerate(batch_files):
            try:
                fileID = self.df.FileID.iloc[subject_id].lower()
            except:
                pdb.set_trace()
            X[i], Y[i] = self.load_h5data_epochs(fileID, epoch_id)
#            Y[i] = Y[i][2] 
#        return np.stack([x for x in X if x != []], axis=0), np.stack([y for y in Y if y != []], axis=0).astype(int)
        return np.stack([x for x in X if x != []], axis=0), Y

class TrainGenerator(DataGenerator):

    def __init__(self, config):
        super(TrainGenerator, self).__init__(config, subset='train', shuffle=True)


class EvalGenerator(DataGenerator):

    def __init__(self, config):
        super(EvalGenerator, self).__init__(config, subset='eval', shuffle=True)


class TestGenerator(DataGenerator):

    def __init__(self, config):
        super(TestGenerator, self).__init__(config, subset='test', shuffle=True)    
    
if __name__ == '__main__':

    config = process_config(r"C:\Users\dumle\OneDrive\Dokumenter\GitHub\stroke-thomas\config\param_configs.json")
    train_data = TrainGenerator(config)
    test_data = TestGenerator(config)
    eval_data = EvalGenerator(config)

    train_gen = DataGenerator(config)
    print(len(train_gen))
    import time
    start_time = time.time()
    number_batches = len(train_gen)
    for i, (x, y) in enumerate(train_gen):
        if i % 50 == 0:
            print('Iteration {} of {}'.format(i, number_batches))
    end_time = time.time()
    print('Total time to run through dataset: {}'.format(end_time - start_time))
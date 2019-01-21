# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 11:14:27 2018

@author: dumle
"""

class BaseTrain(object):
    def __init__(self, model, data, config):
        self.model = model
        self.train_data = data['train']
        self.test_data = data['test']
        self.eval_data = data['eval']
        self.config = config
        self.data_groups = {'train_data': self.train_data,
                    'test_data': self.test_data,
                    'eval_data': self.eval_data}
    def train(self):
        raise NotImplementedError
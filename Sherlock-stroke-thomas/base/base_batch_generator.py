from keras.utils import Sequence


class BaseBatchGenerator(Sequence):

    def __init__(self, config):
        self.config = config

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

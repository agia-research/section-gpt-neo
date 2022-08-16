import abc


class ShrinkMethod(metaclass=abc.ABCMeta):

    def __init__(self, meta, logger):
        self.meta = meta
        self.logger = logger

    @abc.abstractmethod
    def tokenize(self, args, text, body_size, encoder, pad, encoded_pad):
        pass

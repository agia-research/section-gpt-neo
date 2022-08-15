import abc


class ShrinkMethod(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def tokenize(self, args, text, body_size, encoder, pad, encoded_pad):
        pass

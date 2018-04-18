import tensorflow as tf
import abc

class Pooling():
    def __init__(self, encoder_outputs, mask=None):
        self.encoder_outputs = encoder_outputs
        self.mask = mask

    def __call__(self):
        return self._create()

    @abc.abstractmethod
    def _create(self):
        raise NotImplementedError("Must be implemented by child class")



class MeanPooling(Pooling):
    def _create(self):
        if self.mask is not None:
            return tf.reduce_sum(self.encoder_outputs * self.mask, axis=1)/tf.reduce_sum(self.mask, axis=1)
        else:
            return tf.reduce_mean(self.encoder_outputs, axis=1)


class AttentionPooling(Pooling):
    pass
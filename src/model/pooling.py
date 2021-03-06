import tensorflow as tf
import abc

class Pooling(object):
    def __init__(self, inputs, mask=None):
        self.inputs = inputs
        self.mask = mask

    def __call__(self):
        return self._create()

    @abc.abstractmethod
    def _create(self):
        raise NotImplementedError("Must be implemented by child class")



class MeanPooling(Pooling):
    def _create(self):
        if self.mask is not None:
            return tf.reduce_sum(self.inputs * tf.expand_dims(self.mask,-1), axis=1) \
                   / tf.reduce_sum(tf.expand_dims(self.mask,-1), axis=1)
        else:
            return tf.reduce_mean(self.inputs, axis=1)


class AttentionPooling(Pooling):
    def __init__(self, inputs, mask=None):
        Pooling.__init__(self, inputs, mask)

    def _create(self, ):
        output, self.attn_alphas = attention(self.inputs, self.mask)
        return output


class AttentionWithContextPooling(Pooling):

    def __init__(self, inputs, attention_size, mask=None):
        Pooling.__init__(self, inputs, mask)
        self.attention_size=attention_size


    def _create(self):
        output,self.attn_alphas = attention_with_context(self.inputs, self.attention_size, self.mask)
        return output


def attention(inputs, mask):
    hidden_size = inputs.shape[2]
    w_omega = tf.get_variable("w_omega", shape=(hidden_size, ))
    b_omega = tf.get_variable("b_omega", shape=())
    with tf.name_scope('attention'):
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
        v.set_shape((inputs.shape[0], inputs.shape[1]))
        v = tf.expand_dims(v, -1)

    alphas = tf.nn.softmax(v, name="alphas")
    output = tf.reduce_sum(inputs * alphas * tf.expand_dims(mask,-1), 1)

    return output, alphas


def attention_with_context(inputs, attention_size, mask):
    hidden_size = inputs.shape[2]
    w_omega = tf.get_variable("w_omega", shape=(hidden_size, attention_size))
    b_omega = tf.get_variable("b_omega", shape=(attention_size, ))
    u_omega = tf.get_variable("u_omega", shape=(attention_size, ))
    with tf.name_scope('attention'):
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
        v.set_shape((inputs.shape[0],inputs.shape[1],attention_size))

    vu = tf.tensordot(v, u_omega, axes=1,name="vu")
    vu.set_shape((inputs.shape[0], inputs.shape[1]))
    alphas = tf.nn.softmax(vu, name="alphas")
    alphas = tf.expand_dims(alphas, -1)
    output = tf.reduce_sum(inputs * alphas *tf.expand_dims(mask,-1), 1)

    return output, alphas
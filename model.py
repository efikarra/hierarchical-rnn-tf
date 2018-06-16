import tensorflow as tf
import model_helper
import abc
import numpy as np


class BaseModel(object):
    def __init__(self, hparams, mode, iterator):
        self.iterator = iterator
        self.n_classes = hparams.n_classes
        self.batch_size = iterator.batch_size
        self.mode = mode

        # Set weights initializer
        initializer = model_helper.get_initializer(hparams.init_op, hparams.random_seed, hparams.init_weight)
        tf.get_variable_scope().set_initializer(initializer)

        # build tensorflow graph of the main model.
        self.logits, loss = self.build_network(hparams)
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.train_loss = loss
        elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
            self.eval_loss = loss
        # transition parameters are not None only for NNs with CRF on top.
        self.transition_params = self._get_trans_params()
        self.predictions = {
            "probabilities": self.compute_probabilities(self.logits),
            "labels": tf.cast(self.compute_labels(self.logits), tf.int32)
        }
        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            self.accuracy = self.compute_accuracy(self.predictions["labels"])

        ## Learning rate
        print("  start_decay_step=%d, learning_rate=%g, decay_steps %d,"
              " decay_factor %g" % (hparams.start_decay_step, hparams.learning_rate,
                                    hparams.decay_steps, hparams.decay_factor))
        self.global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()
        # Gradients and sgd update operation for model training.
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            # Optimizer
            if hparams.optimizer == "sgd":
                # perform SGD with a learning rate with exponential decay
                self.learning_rate = tf.cond(
                    self.global_step < hparams.start_decay_step,
                    lambda: tf.constant(hparams.learning_rate),
                    lambda: tf.train.exponential_decay(
                        hparams.learning_rate,
                        (self.global_step - hparams.start_decay_step),
                        hparams.decay_steps,
                        hparams.decay_factor,
                        staircase=True),
                    name="learning_rate")
                opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                tf.summary.scalar("lr", self.learning_rate)
            elif hparams.optimizer == "adam":
                self.learning_rate = tf.constant(hparams.learning_rate)
                opt = tf.train.AdamOptimizer(self.learning_rate)
            # compute the gradients of train_loss w.r.t to the model trainable parameters.
            # if colocate_gradients_with_ops is true, the gradients will be computed in the same gpu/cpu device with the
            # original (forward-pass) operator
            gradients = tf.gradients(self.train_loss, params,
                                     colocate_gradients_with_ops=hparams.colocate_gradients_with_ops)
            # clip gradients below a threshold to avoid explosion
            clipped_grads, grad_norm_summary, grad_norm = model_helper.gradient_clip(gradients,
                                                                                     max_gradient_norm=hparams.max_gradient_norm)
            self.grad_norm = grad_norm
            # ask the optimizer to apply the processed gradients. We give as argument a list of pairs (gradient,variable).
            self.update = opt.apply_gradients(
                zip(clipped_grads, params), global_step=self.global_step
            )
            self.train_summary = tf.summary.merge([
                                                      tf.summary.scalar("lr", self.learning_rate),
                                                      tf.summary.scalar("train_loss",
                                                                        self.train_loss), ] + grad_norm_summary
                                                  )
        # Saver. As argument, we give the variables that are going to be saved and restored.
        # The Saver op will save the variables of the graph within which it is defined. All graphs (train/eval/infer)
        # have a Saver operator.
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)
        # Print trainable variables
        print("# Trainable variables")
        for param in params:
            print("  %s, %s" % (param.name, str(param.get_shape())))
        total_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print("Total number of parameters: %d" % total_params)

    # this method is overwritten from models with CRF on top to return trans parameters.
    def _get_trans_params(self):
        return tf.no_op()

    def output_layer(self, hparams, outputs):
        with tf.variable_scope("output_layer"):
            out_layer = tf.layers.Dense(hparams.n_classes, use_bias=False, name="output_layer")
            logits = out_layer(outputs)
        return logits

    def train(self, sess, options=None, run_metadata=None):
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        return sess.run([self.update,
                         self.train_loss,
                         self.train_summary,
                         self.global_step,
                         self.learning_rate,
                         self.batch_size,
                         self.accuracy,
                         self.transition_params],
                        options=options,
                        run_metadata=run_metadata
                        )

    def eval(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.EVAL
        return sess.run([self.eval_loss, self.accuracy, self.batch_size, self.predictions])

    def predict(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        return sess.run(self.predictions)

    @abc.abstractmethod
    def build_network(self, hparams):
        pass

    @abc.abstractmethod
    def compute_labels(self, logits):
        pass

    @abc.abstractmethod
    def compute_probabilities(self, logits):
        pass

    @abc.abstractmethod
    def compute_loss(self, logits):
        pass

    @abc.abstractmethod
    def compute_accuracy(self, labels):
        pass


class FlatModel(BaseModel):
    """This class implements a non hierarchical utterance classifier"""

    def build_network(self, hparams):
        print ("Creating %s graph" % self.mode)
        dtype = tf.float32
        with tf.variable_scope("flat_model", dtype=dtype):
            # self.iterator.input is of shape batch_size x
            input_emb = self.encoder(hparams, self.iterator.input)
            logits = self.output_layer(hparams, input_emb)
            # compute loss
            if self.mode == tf.contrib.learn.ModeKeys.INFER:
                loss = None
            else:
                loss = self.compute_loss(logits)
        return logits, loss

    def compute_loss(self, logits):
        target_output = self.iterator.target
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=logits)
        loss = tf.reduce_mean(crossent)
        return loss

    def compute_accuracy(self, labels):
        target_output = self.iterator.target
        correct_pred = tf.equal(labels, target_output)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy

    def compute_labels(self, logits):
        return tf.argmax(self.compute_probabilities(logits), len(logits.get_shape()) - 1)

    def compute_probabilities(self, logits):
        return tf.nn.softmax(logits)

    @abc.abstractmethod
    def encoder(self, hparams, input):
        pass


class FFN(FlatModel):
    """This class implements a non hierarchical utterance classifier which encodes the input utterance using a FFN network."""

    def encoder(self, hparams, input):
        with tf.variable_scope("ffn"):
            input_emb = model_helper.ffn(input, layers=hparams.uttr_layers, units_list=hparams.uttr_units, bias=True,
                                         hid_to_out_dropouts=hparams.uttr_hid_to_out_dropout,
                                         activations=hparams.uttr_activation, mode=self.mode)
            return input_emb


class RNN(FlatModel):
    """This class implements a non hierarchical utterance classifier which encodes the input utterance using a RNN network."""

    def init_embeddings(self, hparams):
        self.input_embedding, self.input_emb_init, self.input_emb_placeholder = model_helper.create_embeddings \
            (vocab_size=self.vocab_size,
             emb_size=hparams.input_emb_size,
             emb_trainable=hparams.input_emb_trainable,
             emb_pretrain=hparams.input_emb_pretrain)

    def encoder(self, hparams, input):
        self.vocab_size = hparams.vocab_size
        # Create embedding layer
        self.init_embeddings(hparams)
        emb_inp = tf.nn.embedding_lookup(self.input_embedding, input)
        with tf.variable_scope("utterance_rnn") as scope:
            # rnn_outputs.shape = (batch_size, max_uttr_length, uttr_units) or
            # (batch_size, num_utterances, 2*uttr_units) for bi-directional rnn.
            rnn_outputs, last_hidden_sate = model_helper.rnn_network(emb_inp, scope.dtype,
                                                                     hparams.uttr_rnn_type, hparams.uttr_unit_type,
                                                                     hparams.uttr_units, hparams.uttr_layers,
                                                                     hparams.uttr_hid_to_out_dropout,
                                                                     self.iterator.input_uttr_length,
                                                                     hparams.forget_bias, hparams.uttr_activation, self.mode)

            # pool the rnn hidden states to build a representation of each utterance.
            # Pooling methods supported: Last hidden state, Mean pooling, Attention pooling.
            # utterances_embs.shape = (batch_size, uttr_units)
            utterances_embs, self.attn_alphas = model_helper.pool_rnn_output(hparams.uttr_pooling, rnn_outputs,
                                                                             last_hidden_sate,
                                                                             self.iterator.input_uttr_length,
                                                                             hparams.uttr_attention_size)
        return utterances_embs


class CNN(FlatModel):
    """This class implements a non hierarchical utterance classifier which encodes the input utterance using a CNN network."""

    def init_embeddings(self, hparams):
        self.input_embedding, self.input_emb_init, self.input_emb_placeholder = model_helper.create_embeddings \
            (vocab_size=self.vocab_size,
             emb_size=hparams.input_emb_size,
             emb_trainable=hparams.input_emb_trainable,
             emb_pretrain=hparams.input_emb_pretrain)

    def encoder(self, hparams, input):
        self.vocab_size = hparams.vocab_size
        # Create embedding layer
        self.init_embeddings(hparams)
        emb_inp = tf.nn.embedding_lookup(self.input_embedding, input)
        emb_inp = tf.expand_dims(emb_inp, -1)
        with tf.variable_scope("utterance_cnn"):
            filter_sizes = [(filter_size, hparams.input_emb_size) for filter_size in hparams.filter_sizes]
            cnn_outputs = model_helper.cnn(emb_inp, self.iterator.input_uttr_length, filter_sizes,
                                           hparams.num_filters, hparams.stride,
                                           hparams.uttr_activation[0], hparams.uttr_hid_to_out_dropout[0],
                                           self.mode, hparams.padding)
        return cnn_outputs

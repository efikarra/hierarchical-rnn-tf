import tensorflow as tf
import model_helper
import abc

class HModel(object):
    """Base class for Hierarchical Model
    Args:
      hparams: Hyperparameter configurations.
      mode: TRAIN | EVAL | INFER
      iterator: Dataset Iterator that feeds data.
      input_vocab_table: Lookup table mapping input words to ids."""
    def __init__(self, hparams, mode, iterator, input_vocab_table=None):
        self.iterator=iterator
        self.input_vocab_table = input_vocab_table
        self.n_classes = hparams.n_classes
        self.vocab_size = hparams.vocab_size
        self.batch_size = iterator.batch_size
        self.mode = mode

        self.max_sess_length = tf.shape(self.iterator.input)[1]
        self.max_uttr_length = tf.shape(self.iterator.input)[2]


        # Initializer
        initializer = model_helper.get_initializer(hparams.init_op, hparams.random_seed, hparams.init_weight)
        tf.get_variable_scope().set_initializer(initializer)
        # Build graph of hierarchical model
        res = self.build_network(hparams)
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.train_loss = res[1]
        elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
            self.eval_loss = res[1]
        if self.mode != tf.contrib.learn.ModeKeys.TRAIN:
            # Generate predictions (for INFER and EVAL mode)
            self.logits = res[0]
            self.predictions = tf.nn.softmax(self.logits)
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
        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            self.logits = res[0]
            correct_pred = tf.equal(tf.argmax(tf.nn.softmax(self.logits), len(self.logits.get_shape())-1),
                                    tf.cast(self.iterator.target, tf.int64))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        # Saver. As argument, we give the variables that are going to be saved and restored.
        # The Saver op will save the variables of the graph within it is defined. All graphs (train/eval/predict) have
        # have a Saver operator.
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)
        # print trainable params
        # Print trainable variables
        print("# Trainable variables")
        for param in params:
            print("  %s, %s" % (param.name, str(param.get_shape())))
        import numpy as np
        total_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print("Total number of parameters: %d" % total_params)


    def compute_loss(self, logits):
        target_output = self.iterator.target
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=logits)
        target_weights = tf.sequence_mask(self.iterator.input_sess_length, target_output.shape[1].value, dtype=logits.dtype)
        loss = tf.reduce_mean(crossent * target_weights)
               #/ tf.to_float(self.batch_size)
        return loss


    def output_layer(self, hparams, rnn_outputs):
        with tf.variable_scope("output_layer"):
            out_layer = tf.layers.Dense(hparams.n_classes, use_bias=hparams.out_bias, name="output_layer")
            logits = out_layer(rnn_outputs)
        return logits


    def build_network(self, hparams):
        print ("Creating %s graph" % self.mode)
        dtype = tf.float32
        with tf.variable_scope("h_model",dtype = dtype) as scope:
            # utterances representation: utterances_embs.shape = [batch_size*num_utterances, uttr_units] or for bi:
            # [batch_size*num_utterances, uttr_units*2]
            utterances_embs=self.utterance_encoder(hparams, self.iterator.input)
            # session rnn outputs: session_rnn_outputs.shape = [batch_size, max_sess_length, sess_units] or for bi:
            # [batch_size, max_sess_length, sess_units*2]
            session_rnn_outputs = self.session_encoder(hparams, utterances_embs)
            logits = self.output_layer(hparams, session_rnn_outputs)
            # compute loss
            if self.mode == tf.contrib.learn.ModeKeys.INFER:
                loss = None
            else:
                loss = self.compute_loss(logits)
            return logits, loss


    @abc.abstractmethod
    def utterance_encoder(self, hparams, input_emb):
        """All sub-classes should implement this method."""
        pass

    @abc.abstractmethod
    def session_encoder(self, hparams, utterances_embs):
        """All sub-classes should implement this method."""
        pass

    def train(self, sess, options=None, run_metadata=None):
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        return sess.run([self.update,
                         self.train_loss,
                         self.train_summary,
                         self.global_step,
                         self.learning_rate,
                         self.batch_size,
                         self.accuracy],
                        options=options,
                        run_metadata=run_metadata
                        )

    def eval(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.EVAL
        return sess.run([self.eval_loss, self.accuracy, self.batch_size])


    def predict(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        return sess.run(self.predictions)


class H_RNN(HModel):
    """Hierarchical Model with RNN in the session level"""
    def session_encoder(self, hparams, utterances_embs):
        with tf.variable_scope("session_rnn") as scope:
            # reshape_utterances_embs.shape = [batch_size,  max_sess_length, uttr_units * 2] or
            # [batch_size, max_sess_length, uttr_units]
            reshape_utterances_embs = tf.reshape(utterances_embs, shape=[self.batch_size, self.max_sess_length,
                                                                         utterances_embs.get_shape()[-1]])
            rnn_outputs, last_hidden_sate = self.rnn_network(reshape_utterances_embs, scope.dtype,
                                                             hparams.sess_rnn_type, hparams.sess_unit_type,
                                                             hparams.sess_units, hparams.sess_layers,
                                                             hparams.sess_in_to_hid_dropout,
                                                             self.iterator.input_sess_length,
                                                             hparams.forget_bias, hparams.sess_time_major)
        return rnn_outputs

    def rnn_network(self, inputs, dtype, rnn_type,
                    unit_type, num_units, num_layers, in_to_hid_dropout, sequence_length, forget_bias, time_major):
        if rnn_type == 'uni':
            rnn_outputs, last_hidden_sate = self.rnn(inputs, dtype, unit_type, num_units, num_layers, in_to_hid_dropout,
                                                     sequence_length, forget_bias, time_major)
        elif rnn_type == 'bi':
            num_bi_layers = int(num_layers * 2)
            rnn_outputs, last_hidden_sate = self.bidirectional_rnn(inputs, dtype, unit_type, num_units, num_bi_layers,
                                                                   in_to_hid_dropout, sequence_length, forget_bias, time_major)
            last_hidden_sate = tf.concat(last_hidden_sate, -1)
        else:
            raise ValueError("Unknown encoder_type %s" % rnn_type)
        return rnn_outputs, last_hidden_sate


    def rnn(self, inputs, dtype, unit_type, num_units, num_layers, in_to_hid_dropout, sequence_length, forget_bias,
            time_major):
        cell = model_helper.create_rnn_cell(unit_type, num_units, num_layers,
                                            forget_bias, in_to_hid_dropout, self.mode)
        # encoder_state --> a Tensor of shape `[batch_size, cell.state_size]` or a list of such Tensors for many layers
        # the sequence_length achieves 1) performance 2) correctness. The RNN calculations stop when the true seq.
        # length is reached for each sequence. All outputs (hidden states) past the true seq length are set to 0.
        # The hidden state of the true last timestep is returned as last_hidden_state.s
        # he shape format of the inputs and outputs Tensors. If true, these Tensors must be shaped [max_time, batch_size
        # , depth]. If false, these Tensors must be shaped [batch_size, max_time, depth]. Using time_major = True is
        # a bit more efficient
        rnn_outputs, last_hidden_sate = tf.nn.dynamic_rnn(cell, inputs,
                                                          dtype=dtype,
                                                          sequence_length=sequence_length,
                                                          time_major=time_major)
        return rnn_outputs, last_hidden_sate


    def bidirectional_rnn(self, inputs, dtype, unit_type, num_units, num_bi_layers, in_to_hid_dropout,
                          sequence_length, forget_bias, time_major):
        # Construct forward and backward cells.
        #each one has num_bi_layers layers. Each layer has num_units.
        fw_cell = model_helper.create_rnn_cell(unit_type, num_units, num_bi_layers,
                                               forget_bias, in_to_hid_dropout, self.mode)
        bw_cell = model_helper.create_rnn_cell(unit_type, num_units, num_bi_layers,
                                               forget_bias, in_to_hid_dropout, self.mode)

        # initial_state_fw, initial_state_bw are initialized to 0
        # bi_outputs is a tuple (output_fw, output_bw) containing the forward and the backward rnn output Tensor
        # bi_state is a tuple (output_state_fw, output_state_bw) with the forward and the backward final states.
        # Each state has num_units.
        # num_bi_layers>1, we have a list of num_bi_layers tuples.
        bi_outputs,bi_state = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            inputs,
            dtype=dtype,
            sequence_length=sequence_length,
            time_major=time_major)

        # return fw and bw outputs,i.e., ([h1_fw;h1_bw],...,[hT_fw;hT_bw]) concatenated.
        return tf.concat(bi_outputs,-1),bi_state


class H_RNN_FFN(H_RNN):
    """Hierarchical Model with RNN in the session level and FFN in the utterance level."""
    def utterance_encoder(self, hparams, inputs):
        utterances_embs = self.ffn(inputs, hparams)
        return utterances_embs

    def ffn(self, inputs, hparams):
        layer_input = inputs
        for l in range(hparams.uttr_layers):
            layer_output = tf.layers.Dense(hparams.uttr_units[l], use_bias=hparams.out_bias, name="output_layer")(layer_input)
            layer_output = tf.layers.dropout(layer_output, hparams.uttr_ffn_dropouts[l])
            layer_input = layer_output
        return layer_input


class H_RNN_RNN(H_RNN):
    """Hierarchical Model with RNN in the session level and RNN in the utterance level."""
    def init_embeddings(self, hparams):
        self.input_embedding, self.input_emb_init, self.input_emb_placeholder = model_helper.create_embeddings \
            (vocab_size=self.vocab_size,
             emb_size=hparams.input_emb_size,
             emb_trainable=hparams.input_emb_trainable,
             emb_pretrain=hparams.input_emb_pretrain)


    def utterance_encoder(self, hparams, inputs):
        # Create embedding layer
        self.init_embeddings(hparams)
        emb_inp = tf.nn.embedding_lookup(self.input_embedding, inputs)
        with tf.variable_scope("utterance_rnn") as scope:
            # reshape_input_emb.shape = [batch_size*num_utterances, uttr_max_len, embed_dim]
            reshape_input_emb = tf.reshape(emb_inp, [-1, self.max_uttr_length, hparams.input_emb_size])
            reshape_uttr_length = tf.reshape(self.iterator.input_uttr_length, [-1])
            rnn_outputs, last_hidden_sate = self.rnn_network(reshape_input_emb, scope.dtype,
                                                             hparams.uttr_rnn_type, hparams.uttr_unit_type,
                                                             hparams.uttr_units, hparams.uttr_layers,
                                                             hparams.uttr_in_to_hid_dropout,
                                                             reshape_uttr_length,
                                                             hparams.forget_bias, hparams.uttr_time_major)
            # utterances_embs.shape = [batch_size*num_utterances, uttr_units] or
            # [batch_size*num_utterances, 2*uttr_units]
            utterances_embs = model_helper.pool_rnn_output(hparams.uttr_pooling, rnn_outputs, last_hidden_sate)
        return utterances_embs


class H_RNN_CNN(H_RNN):
    """Hierarchical Model with RNN in the session level and CNN in the utterance level."""
    def utterance_encoder(self, hparams, inputs):
        # Create embedding layer
        self.input_embedding, self.input_emb_init, self.input_emb_placeholder = model_helper.create_embeddings \
            (vocab_size=self.vocab_size,
             emb_size=hparams.input_emb_size,
             emb_trainable=hparams.input_emb_trainable,
             emb_pretrain=self.input_emb_pretrain)
        emb_inp = tf.nn.embedding_lookup(self.input_embedding, inputs)
        with tf.variable_scope("utterance_cnn") as scope:
            # reshape_input_emb.shape = [batch_size*num_utterances, uttr_max_len,embed_dim]
            reshape_input_emb = tf.reshape(emb_inp, [-1, self.max_uttr_length, hparams.input_emb_size])

    def cnn(self, inputs, dtype, hparams):
        pass







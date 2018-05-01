import tensorflow as tf
import model_helper
import abc
import model


class HModel(model.BaseModel):

    def compute_loss(self, logits):
        target_output = self.iterator.target
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=logits)
        target_weights = tf.sequence_mask(self.iterator.input_sess_length, target_output.shape[1].value,
                                          dtype=logits.dtype)
        loss = tf.reduce_mean(crossent * target_weights)
        # / tf.to_float(self.batch_size)
        return loss

    def compute_predictions(self, logits):
        return tf.nn.softmax(self.logits)


    def build_network(self, hparams):
        print ("Creating %s graph" % self.mode)
        dtype = tf.float32
        with tf.variable_scope("h_model",dtype = dtype) as scope:
            # reshape_input_emb.shape = [batch_size*num_utterances, uttr_max_len, embed_dim]
            reshape_input = tf.reshape(self.iterator.input, [-1, model_helper.get_tensor_dim(self.iterator.input,-1)])
            # utterances representation: utterances_embs.shape = [batch_size*num_utterances, uttr_units] or for bi:
            # [batch_size*num_utterances, uttr_units*2]
            utterances_embs=self.utterance_encoder(hparams, reshape_input)
            # reshape_utterances_embs.shape = [batch_size,  max_sess_length, uttr_units * 2] or
            # [batch_size, max_sess_length, uttr_units]
            reshape_utterances_embs = tf.reshape(utterances_embs, shape=[self.batch_size, model_helper.get_tensor_dim(self.iterator.input,1),
                                                                         utterances_embs.get_shape()[-1]])
            # session rnn outputs: session_rnn_outputs.shape = [batch_size, max_sess_length, sess_units] or for bi:
            # [batch_size, max_sess_length, sess_units*2]
            session_rnn_outputs = self.session_encoder(hparams, reshape_utterances_embs)
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
        return sess.run([self.eval_loss, self.accuracy, self.batch_size, self.predictions])


    def predict(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        return sess.run(self.predictions)



class H_RNN(HModel):
    """Hierarchical Model with RNN in the session level"""
    def session_encoder(self, hparams, utterances_embs):
        with tf.variable_scope("session_rnn") as scope:
            rnn_outputs, last_hidden_sate = model_helper.rnn_network(utterances_embs, scope.dtype,
                                                             hparams.sess_rnn_type, hparams.sess_unit_type,
                                                             hparams.sess_units, hparams.sess_layers,
                                                             hparams.sess_in_to_hid_dropout,
                                                             self.iterator.input_sess_length,
                                                             hparams.forget_bias, hparams.sess_time_major,
                                                                     hparams.sess_activation, self.mode)
        return rnn_outputs


class H_RNN_FFN(H_RNN):
    """Hierarchical Model with RNN in the session level and FFN in the utterance level."""
    def utterance_encoder(self, hparams, inputs):
        utterances_embs = model_helper.ffn(self.iterator.input, layers=hparams.uttr_layers, units_list=hparams.uttr_units, bias=True,
                                     uttr_in_to_hid_dropouts=hparams.uttr_in_to_hid_dropout,
                                     activations=hparams.uttr_activation, mode=self.mode)
        return utterances_embs



class H_RNN_RNN(H_RNN):
    """Hierarchical Model with RNN in the session level and RNN in the utterance level."""
    def init_embeddings(self, hparams):
        self.input_embedding, self.input_emb_init, self.input_emb_placeholder = model_helper.create_embeddings \
            (vocab_size=self.vocab_size,
             emb_size=hparams.input_emb_size,
             emb_trainable=hparams.input_emb_trainable,
             emb_pretrain=hparams.input_emb_pretrain)


    def utterance_encoder(self, hparams, inputs):
        self.vocab_size = hparams.vocab_size
        self.max_uttr_length = tf.shape(self.iterator.input)[2]
        # Create embedding layer
        self.init_embeddings(hparams)
        emb_inp = tf.nn.embedding_lookup(self.input_embedding, inputs)
        with tf.variable_scope("utterance_rnn") as scope:
            reshape_uttr_length = tf.reshape(self.iterator.input_uttr_length, [-1])
            rnn_outputs, last_hidden_sate = model_helper.rnn_network(emb_inp, scope.dtype,
                                                             hparams.uttr_rnn_type, hparams.uttr_unit_type,
                                                             hparams.uttr_units, hparams.uttr_layers,
                                                             hparams.uttr_in_to_hid_dropout,
                                                             reshape_uttr_length,
                                                             hparams.forget_bias, hparams.uttr_time_major,
                                                                     hparams.uttr_activation, self.mode)
            # utterances_embs.shape = [batch_size*num_utterances, uttr_units] or
            # [batch_size*num_utterances, 2*uttr_units]
            utterances_embs, self.attn_alphas  = model_helper.pool_rnn_output(hparams.uttr_pooling, rnn_outputs,
                                                                              last_hidden_sate, reshape_uttr_length, hparams.uttr_attention_size)
        return utterances_embs


class H_RNN_CNN(H_RNN):
    """Hierarchical Model with RNN in the session level and CNN in the utterance level."""
    def utterance_encoder(self, hparams, inputs):
        self.vocab_size = hparams.vocab_size
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



class H_RNN_FFN_CRF(H_RNN_FFN):

    def compute_loss(self, logits):
        target_output = self.iterator.target
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(logits, target_output, self.iterator.input_sess_length)
        return tf.reduce_mean(-log_likelihood)

    def compute_predictions(self, logits):
        viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(self.logits, self.transition_params,
                                                                    self.iterator.input_sess_length)
        predictions = tf.convert_to_tensor(viterbi_sequence)  # , np.float32)
        return predictions


class H_RNN_RNN_CRF(H_RNN_RNN):

    def compute_loss(self, logits):
        target_output = self.iterator.target
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(logits, target_output, self.iterator.input_sess_length)
        return tf.reduce_mean(-log_likelihood)

    def compute_predictions(self, logits):
        viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(self.logits, self.transition_params,
                                                                    self.iterator.input_sess_length)
        predictions = tf.convert_to_tensor(viterbi_sequence)  # , np.float32)
        return predictions


class H_RNN_RNN_CNN(H_RNN_CNN):

    def compute_loss(self, logits):
        target_output = self.iterator.target
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(logits, target_output, self.iterator.input_sess_length)
        return tf.reduce_mean(-log_likelihood)

    def compute_predictions(self, logits):
        viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(self.logits, self.transition_params,
                                                                    self.iterator.input_sess_length)
        predictions = tf.convert_to_tensor(viterbi_sequence)  # , np.float32)
        return predictions








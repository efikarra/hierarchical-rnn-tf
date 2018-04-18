import collections
import tensorflow as tf
import time
import iterator_utils
import vocab_utils
import numpy as np
import pooling


class TrainModel(collections.namedtuple("TrainModel",("graph", "model", "iterator"))):
    pass


class EvalModel(collections.namedtuple("EvalModel",("graph", "model", "input_file_placeholder", "output_file_placeholder", "iterator"))):
    pass


class InferModel(collections.namedtuple("EvalModel",("graph", "model", "input_file_placeholder", "iterator"))):
    pass


def create_train_model(model_creator, hparams, input_path, target_path, mode):
    graph = tf.Graph()
    with graph.as_default() , tf.container("train"):
        input_vocab_table = vocab_utils.create_vocab_table(hparams.vocab_path)
        input_dataset = tf.data.TextLineDataset(input_path)
        output_dataset = tf.data.TextLineDataset(target_path)
        iterator = iterator_utils.get_iterator_hierarchical(input_dataset, output_dataset, input_vocab_table,
                                                   batch_size=hparams.batch_size, random_seed=hparams.random_seed,
                                                   pad=hparams.pad, sess_max_len=hparams.sess_max_len)
        model = model_creator(hparams, mode, iterator, input_vocab_table=input_vocab_table)
        return TrainModel(graph, model, iterator)


def create_eval_model(model_creator, hparams, mode):
    graph = tf.Graph()
    with graph.as_default(), tf.container("eval"):
        # create a table to map words to vocab ids.
        input_vocab_table = vocab_utils.create_vocab_table(hparams.vocab_path)
        # define a placeholder for the input dataset.
        # we will dynamically initialize this placeholder with a file name during validation.
        # The reason for this is that during validation, we may want to evaluate our trained model on different datasets.
        input_file_placeholder= tf.placeholder(shape=(),dtype=tf.string)
        input_dataset = tf.data.TextLineDataset(input_file_placeholder)
        output_file_placeholder= tf.placeholder(shape=(), dtype=tf.string)
        output_dataset = tf.data.TextLineDataset(output_file_placeholder)

        iterator = iterator_utils.get_iterator_hierarchical(input_dataset, output_dataset, input_vocab_table,
                                               batch_size=hparams.eval_batch_size, random_seed=hparams.random_seed,
                                               pad=hparams.pad, sess_max_len=hparams.sess_max_len)
        model = model_creator(hparams, mode, iterator, input_vocab_table=input_vocab_table)
        return EvalModel(graph, model, input_file_placeholder, output_file_placeholder, iterator)


def create_infer_model(model_creator, hparams, mode):
    graph = tf.Graph()
    with graph.as_default(), tf.container("predict"):
        input_vocab_table = vocab_utils.create_vocab_table(hparams.vocab_path)
        input_file_placeholder= tf.placeholder(shape=(),dtype=tf.string)
        input_dataset = tf.data.TextLineDataset(input_file_placeholder)

        iterator = iterator_utils.get_iterator_hierarchical_infer(input_dataset, input_vocab_table,
                                                     batch_size=hparams.predict_batch_size,
                                                     pad=hparams.pad, sess_max_len=hparams.sess_max_len)
        model = model_creator(hparams, mode, iterator, input_vocab_table=input_vocab_table)
        return InferModel(graph, model, input_file_placeholder, iterator)


def get_initializer(init_op, seed=None, init_weight=None):
    """Create an initializer. init_weight is only for uniform."""
    if init_op == "uniform":
        assert init_weight
        return tf.random_uniform_initializer( -init_weight, init_weight, seed=seed)
    elif init_op == "glorot_normal":
        return tf.keras.initializers.glorot_normal(seed=seed)
    elif init_op == "glorot_uniform":
        return tf.keras.initializers.glorot_uniform(seed=seed)
    else:
        raise ValueError("Unknown init_op %s" % init_op)



def create_embeddings(vocab_size,emb_size,emb_trainable,emb_pretrain,dtype=tf.float32):
    with tf.variable_scope("embeddings", dtype=dtype) as scope:
        emb_placeholder=None
        emb_init=None
        embedding = tf.get_variable("embedding", [vocab_size, emb_size], dtype, trainable=emb_trainable)
        if emb_pretrain:
            emb_placeholder = tf.placeholder(tf.float32, [vocab_size, emb_size])
            emb_init = embedding.assign(emb_placeholder)
        return embedding, emb_init, emb_placeholder


def _single_cell(unit_type, num_units, forget_bias, in_to_hidden_dropout):

    # Cell Type
    if unit_type == "lstm":
        single_cell = tf.contrib.rnn.BasicLSTMCell(num_units,forget_bias=forget_bias)
    elif unit_type == "gru":
        single_cell = tf.contrib.rnn.GRUCell(num_units)
    elif unit_type == "layer_norm_lstm":
        single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units,forget_bias=forget_bias,layer_norm=True)
    elif unit_type == "rnn":
      single_cell = tf.contrib.rnn.BasicRNNCell(
          num_units)
    else:
        raise ValueError("Unknown unit type %s!" % unit_type)
    # Dropout (= 1 - keep_prob)
    if in_to_hidden_dropout > 0.0:
        single_cell = tf.contrib.rnn.DropoutWrapper(
            cell=single_cell, input_keep_prob=(1.0 - in_to_hidden_dropout))
    return single_cell


def _cell_list(unit_type, num_units, num_layers, forget_bias, in_to_hidden_dropout, mode):
  """Create a list of RNN cells."""
  cell_list = []
  for i in range(num_layers):
    in_to_hidden_drop = in_to_hidden_dropout[i] if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0
    single_cell = _single_cell(
        unit_type=unit_type,
        num_units=num_units[i],
        forget_bias=forget_bias,
        in_to_hidden_dropout=in_to_hidden_drop)
    cell_list.append(single_cell)
  return cell_list


def create_rnn_cell(unit_type, num_units, num_layers, forget_bias, in_to_hidden_dropout, mode):
  cell_list = _cell_list(unit_type=unit_type,
                         num_units=num_units,
                         num_layers=num_layers,
                         forget_bias=forget_bias,
                         in_to_hidden_dropout=in_to_hidden_dropout,
                         mode=mode)

  if len(cell_list) == 1:  # Single layer.
    return cell_list[0]
  else:  # Multi layers
    return tf.contrib.rnn.MultiRNNCell(cell_list)


def pool_rnn_output(pooling_method, rnn_outputs, rnn_last_state):
    if pooling_method=='last':
        if isinstance(rnn_last_state,tf.contrib.rnn.LSTMStateTuple): return rnn_last_state.h
        else: return rnn_last_state
    elif pooling_method=='mean':
        output_creator = pooling.MeanPooling(rnn_outputs)
    elif pooling_method == 'attention':
        output_creator = pooling.AttentionPooling(rnn_outputs)
    else:
        raise ValueError("Unknown Pooling method.")
    rnn_output = output_creator()
    return rnn_output


def gradient_clip(gradients, max_gradient_norm):
    # if the global_norm, i.e., the sum of the norms of all gradients, exceeds max_gradient_norm
    # then clip all gradients by the ratio of global_norm. Otherwise, all gradients remain as they are.
    # gradient_norm is the global_norm=sqrt(sum([l2norm(t)**2 for t in gradients]))
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
    gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
    # compute and log the global_norm of the new clipped gradients
    gradient_norm_summary.append(tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))
    return clipped_gradients, gradient_norm_summary, gradient_norm


def run_batch_evaluation(model, session):
    batch_count=0.0
    loss=0.0
    accuracy = 0.0
    while True:
        try:
            batch_loss,batch_accuracy,batch_size=model.eval(session)
            loss+=batch_loss
            accuracy+=batch_accuracy
            batch_count+=1
        except tf.errors.OutOfRangeError:
            break

    loss /= batch_count
    accuracy /= batch_count
    return loss, accuracy


def run_batch_prediction(model, session):
    concat_predictions = None
    batch_count = 0
    while True:
        try:
            batch_count += 1
            predictions = model.predict(session)
            if concat_predictions is None:
                concat_predictions = predictions
            else:
                concat_predictions = np.append(concat_predictions, predictions, axis=0)

        except tf.errors.OutOfRangeError:
            break
    return concat_predictions

def load_model(model, session, name, ckpt):
    start_time=time.time()
    #initialize all read-only tables of the graph, e.g., vocabulary tables or embedding tables.
    session.run(tf.local_variables_initializer())
    session.run(tf.tables_initializer())
    model.saver.restore(session, ckpt)
    print("loaded %s model parameters from %s, time %.2fs" % (name, ckpt, time.time()-start_time))
    return model


def create_or_load_model(model, session, name, model_dir, input_emb_weights=None):
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
        model = load_model(model, session, name, latest_ckpt)
    else:
        start_time = time.time()
        #initialize all global and local variables in the graph, e.g., the model's weights.
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())
        # initialize all read-only tables of the graph, e.g., vocabulary tables or embedding tables.
        session.run(tf.tables_initializer())
        if input_emb_weights is not None:
            session.run(model.input_emb_init, feed_dict={model.input_emb_placeholder: input_emb_weights})
            print ("created model %s with new parameters, time %.2fs" %(name,time.time()-start_time))
    return model


def add_summary(summary_writer, tag, value):
    """Add a new summary to the current summary_writer."""
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    # global_step value to record with the summary (optional).
    summary_writer.add_summary(summary, global_step=None)
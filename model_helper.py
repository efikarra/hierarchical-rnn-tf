import collections
import tensorflow as tf
import time
import iterator_utils
import vocab_utils
import numpy as np
import pooling
import model

class TrainModel(collections.namedtuple("TrainModel",("graph", "model", "iterator"))):
    pass


class EvalModel(collections.namedtuple("EvalModel",("graph", "model", "input_file_placeholder", "output_file_placeholder", "iterator"))):
    pass


class InferModel(collections.namedtuple("EvalModel",("graph", "model", "input_file_placeholder", "iterator"))):
    pass


def get_dataset_iterator(hparams, input_path, target_path,shuffle=True):
    if is_ffn(hparams.model_architecture):
        if is_hierarchical(hparams.model_architecture):
            dataset = tf.data.TFRecordDataset(input_path)
            iterator = iterator_utils.get_iterator_hierarchical_bow(dataset, batch_size=hparams.batch_size,
                                                                feature_size=hparams.feature_size,
                                                                random_seed=hparams.random_seed,shuffle=shuffle)
        else:
            dataset = tf.data.TFRecordDataset(input_path)
            iterator = iterator_utils.get_iterator_flat_bow(dataset, batch_size=hparams.batch_size,
                                                            feature_size=hparams.feature_size,
                                                            random_seed=hparams.random_seed, shuffle=shuffle)
    else:
        input_vocab_table = vocab_utils.create_vocab_table(hparams.vocab_path)
        input_dataset = tf.data.TextLineDataset(input_path)
        output_dataset = tf.data.TextLineDataset(target_path)
        if is_hierarchical(hparams.model_architecture):
            iterator = iterator_utils.get_iterator_hierarchical(input_dataset, output_dataset, input_vocab_table,
                                                            batch_size=hparams.batch_size,
                                                            random_seed=hparams.random_seed,
                                                            pad=hparams.pad,shuffle=shuffle)
        else:
            iterator = iterator_utils.get_iterator_flat(input_dataset, output_dataset, input_vocab_table,
                                                                batch_size=hparams.batch_size,
                                                                random_seed=hparams.random_seed,
                                                                pad=hparams.pad,shuffle=shuffle)
    return iterator


def is_hierarchical(model_architecture):
    if model_architecture=="h-rnn-ffn" or model_architecture=="h-rnn-rnn" or model_architecture=="h-rnn-cnn":
        return True
    elif model_architecture=="ffn" or model_architecture=="rnn" or model_architecture=="cnn": return False
    else: raise ValueError("Unknown model architecture %s."%model_architecture)

def is_ffn(model_architecture):
    if model_architecture=="h-rnn-ffn" or model_architecture=="ffn":
        return True
    else: return False


def create_train_model(model_creator, hparams, input_path, target_path, mode):
    graph = tf.Graph()
    with graph.as_default() , tf.container("train"):
        # quick and dirty. pick a common format for the input data.
        iterator = get_dataset_iterator(hparams, input_path, target_path)
        model = model_creator(hparams, mode, iterator)
        return TrainModel(graph, model, iterator)


def create_eval_model(model_creator, hparams, mode, shuffle=True):
    graph = tf.Graph()
    with graph.as_default(), tf.container("eval"):
        input_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        output_file_placeholder=None
        if not hparams.model_architecture=="ffn" and not hparams.model_architecture=="h-rnn-ffn":
            output_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        iterator = get_dataset_iterator(hparams, input_file_placeholder, output_file_placeholder,shuffle)
        model = model_creator(hparams, mode, iterator)
        return EvalModel(graph, model, input_file_placeholder, output_file_placeholder, iterator)


def create_infer_model(model_creator, hparams, mode):
    graph = tf.Graph()
    with graph.as_default(), tf.container("predict"):
        input_vocab_table = vocab_utils.create_vocab_table(hparams.vocab_path)
        input_file_placeholder= tf.placeholder(shape=(),dtype=tf.string)
        input_dataset = tf.data.TextLineDataset(input_file_placeholder)

        iterator = iterator_utils.get_iterator_hierarchical_infer(input_dataset, input_vocab_table,
                                                     batch_size=hparams.predict_batch_size,
                                                     pad=hparams.pad)
        model = model_creator(hparams, mode, iterator, input_vocab_table=input_vocab_table)
        return InferModel(graph, model, input_file_placeholder, iterator)


def rnn_network(inputs, dtype, rnn_type,
                unit_type, num_units, num_layers, hid_to_out_dropout, sequence_length, forget_bias, time_major,
                activations, mode):
    if rnn_type == 'uni':
        rnn_outputs, last_hidden_sate = rnn(inputs, dtype, unit_type, num_units, num_layers, hid_to_out_dropout,
                                                 sequence_length, forget_bias, time_major, activations, mode)
    elif rnn_type == 'bi':
        rnn_outputs, last_hidden_sate = bidirectional_rnn(inputs, dtype, unit_type, num_units, num_layers,
                                                          hid_to_out_dropout, sequence_length, forget_bias,
                                                          time_major, activations, mode)
        last_hidden_sate = tf.concat(last_hidden_sate, -1)
    else:
        raise ValueError("Unknown encoder_type %s" % rnn_type)
    return rnn_outputs, last_hidden_sate


def rnn(inputs, dtype, unit_type, num_units, num_layers, hid_to_out_dropout, sequence_length, forget_bias,
        time_major, activations, mode):
    cell = create_rnn_cell(unit_type, num_units, num_layers,
                           forget_bias, hid_to_out_dropout, mode, activations)
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


def ffn(inputs, layers, units_list, bias, hid_to_out_dropouts, activations, mode):
    layer_input = inputs
    for l in range(layers):
        hid_to_out_dropout = hid_to_out_dropouts[l] if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0
        layer_output = tf.layers.Dense(units_list[l], activation=get_activation_func(activations[l]), use_bias=bias)(layer_input)
        layer_output = tf.nn.dropout(layer_output, keep_prob=(1-hid_to_out_dropout))
        layer_input = layer_output
    return layer_input


def cnn(inputs, seq_lens, filter_sizes, num_filters, strides, activation, dropout, mode, padding="valid"):
    outputs=[]
    for i,filter_size in enumerate(filter_sizes):
        conv = tf.layers.conv2d(inputs, filters=num_filters, kernel_size=filter_size, strides=strides,
                                activation=get_activation_func(activation), padding=padding)
        # masks = []
        # mask = tf.expand_dims(tf.sequence_mask(seq_lens, maxlen=tf.shape(conv)[1], dtype=tf.float32), -1)
        # for n in range(num_filters):
        #     masks.append(mask)
        # mask = tf.stack(masks, axis=3)
        # masked_conv = conv * mask
        masked_conv = conv
        pooled=tf.reduce_max(masked_conv, axis=1, keep_dims=True)
        #pooled=tf.layers.max_pooling2d(conv, pool_size=pool_size, strides=strides, padding=padding)
        outputs.append(pooled)
    num_filters_total = num_filters * len(filter_sizes)
    cnn_output=tf.concat(outputs, axis=3)
    cnn_output=tf.reshape(cnn_output,[-1,num_filters_total])
    dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0
    cnn_output = tf.nn.dropout(cnn_output, (1-dropout))
    return cnn_output


def bidirectional_rnn(inputs, dtype, unit_type, num_units, num_bi_layers, hid_to_out_dropout,
                      sequence_length, forget_bias, time_major, activations, mode):
    # Construct forward and backward cells.
    #each one has num_bi_layers layers. Each layer has num_units.
    fw_cell = create_rnn_cell(unit_type, num_units, num_bi_layers,
                                           forget_bias, hid_to_out_dropout, mode, activations)
    bw_cell = create_rnn_cell(unit_type, num_units, num_bi_layers,
                                           forget_bias, hid_to_out_dropout, mode, activations)

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



def pool_rnn_output(pooling_method, rnn_outputs, rnn_last_state, sequence_length, attention_size=32):
    mask = tf.sequence_mask(sequence_length, maxlen=rnn_outputs.shape[1].value,
                                      dtype=rnn_outputs.dtype)
    attn_alphas = None
    if pooling_method=='last':
        if isinstance(rnn_last_state,tf.contrib.rnn.LSTMStateTuple): return rnn_last_state.h,attn_alphas
        else: return rnn_last_state,attn_alphas
    elif pooling_method=='mean':
        output_creator = pooling.MeanPooling(rnn_outputs,mask)
    elif pooling_method == 'attn':
        output_creator = pooling.AttentionPooling(inputs=rnn_outputs,mask=mask)
    elif pooling_method == 'attn_context':
        output_creator = pooling.AttentionWithContextPooling(inputs=rnn_outputs, attention_size=attention_size,
                                                            mask=mask)
    else:
        raise ValueError("Unknown Pooling method.")
    rnn_output = output_creator()
    if pooling_method == 'attn_context' or pooling_method =='attn':
        attn_alphas=output_creator.attn_alphas
    return rnn_output, attn_alphas


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

def get_activation_func(act_name):
    if act_name=="relu":
        return tf.nn.relu
    elif act_name=="leaky_relu":
        return tf.nn.leaky_relu
    elif act_name=="sigmoid":
        return tf.nn.sigmoid
    elif act_name=="softmax":
        return tf.nn.softmax
    elif act_name=="tanh":
        return tf.nn.tanh
    else:
        raise ValueError("Unknown value for activation function %s"%act_name)


def create_embeddings(vocab_size,emb_size,emb_trainable,emb_pretrain,dtype=tf.float32):
    with tf.variable_scope("embeddings", dtype=dtype) as scope:
        emb_placeholder=None
        emb_init=None
        embedding = tf.get_variable("embedding", [vocab_size, emb_size], dtype, trainable=emb_trainable)
        if emb_pretrain:
            emb_placeholder = tf.placeholder(tf.float32, [vocab_size, emb_size])
            emb_init = embedding.assign(emb_placeholder)
        return embedding, emb_init, emb_placeholder


def _single_cell(unit_type, num_units, forget_bias, hid_to_out_dropout, activation):

    # Cell Type
    if unit_type == "lstm":
        single_cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=forget_bias, activation=activation)
    elif unit_type == "gru":
        single_cell = tf.contrib.rnn.GRUCell(num_units, activation=activation)
    elif unit_type == "layer_norm_lstm":
        single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units, forget_bias=forget_bias, layer_norm=True, activation=activation)
    elif unit_type == "rnn":
      single_cell = tf.contrib.rnn.BasicRNNCell(
          num_units,activation=activation)
    else:
        raise ValueError("Unknown unit type %s!" % unit_type)
    # Dropout (= 1 - keep_prob)
    if hid_to_out_dropout > 0.0:
        single_cell = tf.contrib.rnn.DropoutWrapper(
            cell=single_cell, input_keep_prob=(1.0 - hid_to_out_dropout))
    return single_cell


def _cell_list(unit_type, num_units, num_layers, forget_bias, hid_to_out_dropout, mode, activations):
  """Create a list of RNN cells."""
  cell_list = []
  for i in range(num_layers):
    hid_to_out_dropout = hid_to_out_dropout[i] if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0
    single_cell = _single_cell(
        unit_type=unit_type,
        num_units=num_units[i],
        forget_bias=forget_bias,
        hid_to_out_dropout=hid_to_out_dropout,
        activation = activations[i])
    cell_list.append(single_cell)
  return cell_list


def create_rnn_cell(unit_type, num_units, num_layers, forget_bias, hid_to_out_dropout, mode, activations):
  cell_list = _cell_list(unit_type=unit_type,
                         num_units=num_units,
                         num_layers=num_layers,
                         forget_bias=forget_bias,
                         hid_to_out_dropout=hid_to_out_dropout,
                         mode=mode,
                         activations=[get_activation_func(act_name) for act_name in activations])

  if len(cell_list) == 1:  # Single layer.
    return cell_list[0]
  else:  # Multi layers
    return tf.contrib.rnn.MultiRNNCell(cell_list)



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
            batch_loss,batch_accuracy,batch_size,_=model.eval(session)
            loss+=batch_loss
            accuracy+=batch_accuracy
            batch_count+=1
        except tf.errors.OutOfRangeError:
            break

    loss /= batch_count
    accuracy /= batch_count
    return loss, accuracy


def run_batch_evaluation_and_prediction(model, session):
    batch_count=0.0
    loss=0.0
    accuracy = 0.0
    concat_predictions = None
    while True:
        try:
            batch_loss,batch_accuracy,batch_size,predictions=model.eval(session)
            loss+=batch_loss
            accuracy+=batch_accuracy
            batch_count+=1
            if concat_predictions is None:
                concat_predictions = predictions
            else:
                concat_predictions = np.append(concat_predictions, predictions, axis=0)
        except tf.errors.OutOfRangeError:
            break

    loss /= batch_count
    accuracy /= batch_count
    return loss, accuracy, concat_predictions


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

def get_model_creator(model_architecture):
    import hierarchical_model
    if model_architecture == "h-rnn-ffn": model_creator = hierarchical_model.H_RNN_FFN
    elif model_architecture == "h-rnn-cnn":
        model_creator = hierarchical_model.H_RNN_CNN
    elif model_architecture == "h-rnn-rnn":
        model_creator = hierarchical_model.H_RNN_RNN
    elif model_architecture == "rnn":
        model_creator = model.RNN
    elif model_architecture == "ffn":
        model_creator = model.FFN
    elif model_architecture == "cnn":
        model_creator = model.CNN
    else: raise ValueError("Unknown model architecture. Only simple_rnn is supported so far.")
    return model_creator


def get_tensor_dim(tensor,time_axis):
    return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]
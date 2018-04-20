import collections
import tensorflow as tf
import tfrecords_creator

class BatchedInput(collections.namedtuple("BatchedInput",
                                           ("initializer",
                                            "input",
                                            "target",
                                            "input_sess_length",
                                            "input_uttr_length",
                                            "batch_size"))):
    pass



def get_iterator_hierarchical_infer(input_dataset, input_vocab_table, batch_size, pad,
                            uttr_delimiter="#", word_delimiter=" "):
    pad_id = tf.cast(input_vocab_table.lookup(tf.constant(pad)),tf.int32)

    input_dataset = input_dataset.map(lambda inp: tf.sparse_tensor_to_dense(tf.string_split(tf.string_split([inp],
                                                                                                            delimiter=uttr_delimiter).values
                                                                   , delimiter=word_delimiter), default_value=pad))
    # remove input sequences of zero length
    input_dataset = input_dataset.filter(lambda inp: tf.size(inp) > 0)
    # Map words to ids
    input_dataset = input_dataset.map(lambda inp: tf.cast(input_vocab_table.lookup(inp), tf.int32))
    # get actual length of input sequence
    input_output_dataset = input_dataset.map(lambda inp: (inp, tf.shape(inp)[0],
                                                                      tf.cast(tf.count_nonzero(inp, axis=1), tf.int32)))

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=(tf.TensorShape([None, None]),
                           tf.TensorShape([])),
            padding_values=(pad_id, 0)
        )

    batched_dataset = batching_func(input_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    (inputs, sess_lens, uttr_lens) = (batched_iter.get_next())
    return BatchedInput(initializer=batched_iter.initializer,
                        input=inputs, target=None,
                        input_sess_length=sess_lens,
                        input_uttr_length=uttr_lens,
                        batch_size=tf.size(sess_lens))


def get_iterator_hierarchical(input_dataset, output_dataset, input_vocab_table, batch_size, random_seed, pad,
                              output_buffer_size=None, uttr_delimiter="#", word_delimiter=" ",
                              label_delimiter=" "):
    if not output_buffer_size: output_buffer_size = batch_size * 1000

    pad_id = tf.cast(input_vocab_table.lookup(tf.constant(pad)),tf.int32)
    input_output_dataset = tf.data.Dataset.zip((input_dataset, output_dataset))
    input_output_dataset = input_output_dataset.shuffle(output_buffer_size, random_seed)

    input_output_dataset=input_output_dataset.map(
        lambda inp,out: (tf.sparse_tensor_to_dense(tf.string_split(tf.string_split([inp], delimiter=uttr_delimiter).values
                                                                   , delimiter=word_delimiter), default_value=pad),
                         tf.string_to_number(tf.string_split([out], delimiter=label_delimiter).values, tf.int32)),)

    #remove input sequences of zero length
    input_output_dataset = input_output_dataset.filter(lambda inp,out: tf.size(inp)>0)
    # Map words to ids
    input_output_dataset = input_output_dataset.map(lambda inp,out:
                                                    (tf.cast(input_vocab_table.lookup(inp),tf.int32),out), )
    # get actual length of input sequence
    def count1d(t):
        return tf.cast(tf.size(tf.where(tf.not_equal(t, pad_id))), tf.int32)
    input_output_dataset = input_output_dataset.map(lambda inp,out:(inp,out,tf.shape(inp)[0],
                                                                    tf.map_fn(count1d, inp, dtype=tf.int32)))

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=(tf.TensorShape([None,None]),
                           tf.TensorShape([None]),
                           tf.TensorShape([]),
                           tf.TensorShape([None])),
            padding_values=(pad_id,0,0,0)
        )
    batched_dataset = batching_func(input_output_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    # inputs.shape = [batch_size, max_sess_len, max_uttr_len]
    # inputs.shape = [batch_size, max_sess_len]
    (inputs,outputs,sess_lens,uttr_lens)=(batched_iter.get_next())
    return BatchedInput(initializer=batched_iter.initializer,
                        input=inputs, target=outputs,
                        input_sess_length=sess_lens,
                        input_uttr_length=uttr_lens,
                        batch_size=tf.size(sess_lens))


def get_iterator_hierarchical_bow(dataset, batch_size, random_seed, feature_size, output_buffer_size=None):
    if not output_buffer_size: output_buffer_size = batch_size * 1000
    dataset = dataset.map(lambda example: tfrecords_creator.parse_sequence_tfrecord(example, feature_size), num_parallel_calls=5)
    dataset = dataset.shuffle(output_buffer_size, random_seed)

    def batching_func(x):
        return  x.padded_batch(batch_size, padded_shapes={

            "labels": tf.TensorShape([None]),
            "length": tf.TensorShape([]),  # Likewise for the length of the sequence
            "features": tf.TensorShape([None, feature_size])  # but the seqeunce is variable length, we pass that information to TF
        })

    batched_dataset = batching_func(dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    # inputs.shape = [batch_size, max_sess_len, max_uttr_len]
    # inputs.shape = [batch_size, max_sess_len]
    inputs = batched_iter.get_next()
    return BatchedInput(initializer=batched_iter.initializer,
                        input=inputs["features"], target=inputs["labels"],
                        input_sess_length=inputs["length"],
                        input_uttr_length=None,
                        batch_size=tf.size(inputs["length"]))

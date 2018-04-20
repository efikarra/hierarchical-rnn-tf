import tensorflow as tf
import numpy as np
import preprocess

def sequence_to_tf_example(sequence, labels):
    example = tf.train.SequenceExample()
    # A non-sequential feature
    sequence_length = sequence.shape[0]
    example.context.feature["length"].int64_list.value.append(sequence_length)
    # Feature lists for the two sequential features of our example
    fl_tokens = example.feature_lists.feature_list["tokens"]
    fl_labels = example.feature_lists.feature_list["labels"]
    for i in range(sequence_length):
        fl_tokens.feature.add().float_list.value.extend(sequence[i])
        fl_labels.feature.add().int64_list.value.append(labels[i])
    return example

def vector_to_tf_example(array, labels):
    pass


def parse_tfrecord(example):
    feature = {'label': tf.FixedLenFeature([], tf.int64),
               'features': tf.FixedLenFeature([12624], tf.float32)}
    parsed_example = tf.parse_single_example(serialized=example, features=feature)
    return {"features": parsed_example["features"], "label": parsed_example["label"]}


def parse_sequence_tfrecord(example, feature_size):
    context_features = {
        "length": tf.FixedLenFeature([], dtype=tf.int64),
    }
    sequence_features = {
        "tokens": tf.FixedLenSequenceFeature([feature_size], dtype=tf.int64),
        "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    }

    # Parse the example (returns a dictionary of tensors)
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=example,
        context_features=context_features,
        sequence_features=sequence_features
    )
    return {"features": sequence_parsed["tokens"], "length": context_parsed["length"],
            "labels": sequence_parsed["labels"]}


def convert_utterances_to_tfrecords(savepath, utterances, labels):
    with open(savepath, 'w') as f:
        writer = tf.python_io.TFRecordWriter(f.name)
        for i in range(utterances.shape[0]):
            print i
            record = vector_to_tf_example(utterances[i],labels[i])
            writer.write(record.SerializeToString())


def convert_sessions_to_tfrecords(savepath, sessions, labels):
    with open(savepath, 'w') as f:
        writer = tf.python_io.TFRecordWriter(f.name)
        for i in range(len(sessions)):
            record = sequence_to_tf_example(sessions[i], labels[i])
            writer.write(record.SerializeToString())


def convert_sessions_bow_to_tfrecords(data_folder, out_folder):
    bow_tr,bow_dev,bow_te,labs_tr,labs_dev,labs_te = preprocess.load_bow(data_folder)
    for i,sess in enumerate(bow_tr):
        bow_tr[i] = np.squeeze(np.asarray(sess.todense()))
    for i,sess in enumerate(bow_dev):
        bow_dev[i] = np.squeeze(np.asarray(sess.todense()))
    for i,sess in enumerate(bow_te):
        bow_te[i] = np.squeeze(np.asarray(sess.todense()))
    convert_sessions_to_tfrecords(out_folder+"train_bow_sess.tfrecord", bow_tr, labs_tr)
    convert_sessions_to_tfrecords(out_folder + "dev_bow_sess.tfrecord", bow_dev, labs_dev)
    convert_sessions_to_tfrecords(out_folder + "test_bow_sess.tfrecord", bow_te, labs_te)


def convert_utterances_bow_to_tfrecords(data_folder, out_folder):
    bow_tr, bow_dev, bow_te, labs_tr, labs_dev, labs_te = preprocess.load_bow(data_folder)
    for i,sess in enumerate(bow_tr):
        bow_tr[i] = np.squeeze(np.asarray(sess.todense()))
    for i,sess in enumerate(bow_dev):
        bow_dev[i] = np.squeeze(np.asarray(sess.todense()))
    for i,sess in enumerate(bow_te):
        bow_te[i] = np.squeeze(np.asarray(sess.todense()))
    bow_tr = np.concatenate(bow_tr,axis=0)
    bow_dev = np.concatenate(bow_dev,axis=0)
    bow_te = np.concatenate(bow_te,axis=0)
    labs_tr = [item for sublist in labs_tr for item in sublist]
    labs_dev = [item for sublist in labs_dev for item in sublist]
    labs_te = [item for sublist in labs_te for item in sublist]
    convert_utterances_to_tfrecords(out_folder+"train_bow_uttr.tfrecord", bow_tr, labs_tr)
    convert_utterances_to_tfrecords(out_folder+"val_bow_uttr.tfrecord", bow_dev, labs_dev)
    convert_utterances_to_tfrecords(out_folder+"test_bow_uttr.tfrecord", bow_te, labs_te)


def test_dataset():
    dataset = tf.contrib.data.TFRecordDataset("experiments/data/val_bow_uttr.tfrecord")
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=5)
    # get actual length of session sequence
    batch_size = 4
    batched_dataset = dataset.batch(batch_size)
    batched_iter = batched_dataset.make_initializable_iterator()
    # inputs.shape = [batch_size, max_sess_len, max_uttr_len]
    # inputs.shape = [batch_size, max_sess_len]

    input = batched_iter.get_next()

    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        sess.run(batched_iter.initializer)
        next_element = sess.run(input)
        print next_element["label"].shape
        print next_element["features"].shape
        print next_element["features"]
        print next_element["label"]

if __name__=="__main__":
    pass
    # convert_sessions_bow_to_tfrecords("experiments/mhddata_pickle/", "experiments/data/")
    # convert_utterances_bow_to_tfrecords("experiments/mhddata_pickle/", "experiments/data/")
import tensorflow as tf
import tfrecords_creator
import iterator_utils
import vocab_utils


def test_tfrecords(filepath):
    dataset = tf.contrib.data.TFRecordDataset(filepath)
    parse = lambda inp: tfrecords_creator.parse_tfrecord(inp,12624)
    dataset = dataset.map(parse, num_parallel_calls=5)
    # get actual length of session sequence
    batch_size = 128
    batched_dataset = dataset.batch(batch_size)
    batched_iter = batched_dataset.make_initializable_iterator()
    # inputs.shape = [batch_size, max_sess_len, max_uttr_len]
    # inputs.shape = [batch_size, max_sess_len]

    input = batched_iter.get_next()

    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        sess.run(batched_iter.initializer)
        data_size = 0
        while True:
            try:
                next_element = sess.run(input)
                # print next_element["label"].shape
                # print next_element["features"].shape
                # print next_element["features"]
                # print next_element["label"]
                data_size+=next_element["features"].shape[0]
            except tf.errors.OutOfRangeError:
                print("end of dataset")
                break
        print "Total data: %d"%data_size


def test_text_data(inputpath,outputpath,vocab_path,out_dir):
    input_dataset = tf.contrib.data.TextLineDataset(inputpath)
    output_dataset = tf.contrib.data.TextLineDataset(outputpath)
    vocab_size, vocab_path = vocab_utils.check_vocab(vocab_path, out_dir,
                                                     unk="<unk>", pad="<pad>")
    input_vocab_table = vocab_utils.create_vocab_table(vocab_path)
    reverse_input_vocab_table = vocab_utils.create_inverse_vocab_table(vocab_path)
    iterator = iterator_utils.get_iterator_flat(input_dataset, output_dataset, input_vocab_table, batch_size=32, random_seed=None, pad="<pad>",
                              output_buffer_size=None, word_delimiter=" ")
    input_words = reverse_input_vocab_table.lookup(tf.to_int64(iterator.input))
    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        sess.run(iterator.initializer)
        input,target,input_uttr_length,input_words=sess.run([iterator.input,iterator.target,iterator.input_uttr_length,input_words])
        while True:
            try:
                print input.shape
                for uttr in input_words:
                    print " ".join(uttr)
                print target
                print input_uttr_length
            except tf.errors.OutOfRangeError:
                print("end of dataset")
                break


if __name__=="__main__":
    # convert_sessions_bow_to_tfrecords("experiments/mhddata_pickle/", "experiments/data/")
    # convert_utterances_bow_to_tfrecords("experiments/mhddata_pickle/", "experiments/data/tfrecords/")
    # test_tfrecords("experiments/data/tfrecords/test_bow_uttr.tfrecord")
    test_text_data("experiments/data/val_input_uttr.txt", "experiments/data/val_target_uttr.txt", "experiments/data/0.00.0vocab_uttr.txt", "experiments/out_model/")
""" Methods to preprocesses the mhddata to transform them into the input format that the NN models need. """

import nltk
import itertools
import numpy as np
import os
import utils
import cPickle

def load_pickle_train_val_test_labels(data_folder, train_label_file, val_label_file, test_label_file):
    """ Load pickle mhddata labels from data_folder."""
    with open(os.path.join(data_folder, train_label_file), "rb") as f:
        labs_tr = cPickle.load(f)
    print("% s loaded." % train_label_file)
    with open(os.path.join(data_folder, val_label_file), "rb") as f:
        labs_val = cPickle.load(f)
    print("% s loaded." % val_label_file)
    with open(os.path.join(data_folder, test_label_file), "rb") as f:
        labs_te = cPickle.load(f)
    print("% s loaded." % test_label_file)
    return labs_tr, labs_val, labs_te


def load_pickle_train_val_test_input(data_folder, train_input_file, val_input_file, test_input_file):
    """ Load pickle mhddata input data from data_folder."""
    with open(os.path.join(data_folder, train_input_file), "rb") as f:
        input_tr = cPickle.load(f)
    print("% s loaded." % train_input_file)
    with open(os.path.join(data_folder, val_input_file), "rb") as f:
        input_val = cPickle.load(f)
    print("% s loaded." % val_input_file)
    with open(os.path.join(data_folder, test_input_file), "rb") as f:
        input_te = cPickle.load(f)
    print("% s loaded." % test_input_file)
    return input_tr, input_val, input_te


def load_pickle_train_val_test(data_folder, train_input_file, train_label_file,
                               val_input_file, val_label_file, test_input_file, test_label_file):
    """ Load pickle mhddata input data and labels from data_folder."""
    labs_tr, labs_val, labs_te = load_pickle_train_val_test_input(data_folder, train_label_file, val_label_file,
                                                                  test_label_file)
    input_tr, input_val, input_te = load_pickle_train_val_test_labels(data_folder, train_input_file, val_input_file,
                                                                      test_input_file)
    return input_tr, labs_tr, input_val, labs_val, input_te, labs_te


def save_pickle_train_val_test(train_input, train_labels, val_input, val_labels, test_input, test_labels, out_folder,
                               suffix):
    with open(os.path.join(out_folder, "train_" + suffix + ".pickle"), "wb") as f:
        cPickle.dump(train_input, f)
    with open(os.path.join(out_folder, "val_" + suffix + ".pickle"), "wb") as f:
        cPickle.dump(val_input, f)
    with open(os.path.join(out_folder, "test_" + suffix + ".pickle"), "wb") as f:
        cPickle.dump(test_input, f)
    with open(os.path.join(out_folder, "train_target_" + suffix + ".pickle"), "wb") as f:
        cPickle.dump(train_labels, f)
    with open(os.path.join(out_folder, "val_target_" + suffix + ".pickle"), "wb") as f:
        cPickle.dump(val_labels, f)
    with open(os.path.join(out_folder, "test_target_" + suffix + ".pickle"), "wb") as f:
        cPickle.dump(test_labels, f)
    print("Done saving")


def regroup_text_sessions(sessions, labels, session_size):
    """ Split original sessions (both input data and labels) into subsessions of size session_size."""
    new_sessions = []
    new_labels = []
    for i, sess in enumerate(sessions):
        count = 1
        new_sess = []
        new_lab = []
        for j, uttr in enumerate(sess):
            new_sess.append(uttr)
            new_lab.append(labels[i][j])
            if count % session_size == 0 or count == len(sess):
                new_sessions.append(new_sess)
                new_labels.append(new_lab)
                new_sess = []
                new_lab = []
            count += 1
    return new_sessions, new_labels


def regroup_text_labels(labels, session_size):
    """ Split original sessions (only labels) into subsessions of size session_size."""
    new_labels = []
    for i, sess in enumerate(labels):
        count = 1
        new_lab = []
        for j, uttr in enumerate(sess):
            new_lab.append(labels[i][j])
            if count % session_size == 0 or count == len(sess):
                new_labels.append(new_lab)
                new_lab = []
            count += 1
    return new_labels


def flatten_text_sessions(words_tr, labs_tr, words_val, labs_val, words_te, labs_te):
    """ Flatten train/val/test sessions to get lists of utterances. """
    new_words_tr = [words for uttr in words_tr for words in uttr]
    new_words_val = [words for uttr in words_val for words in uttr]
    new_words_te = [words for uttr in words_te for words in uttr]
    new_labs_tr = [lab for uttr in labs_tr for lab in uttr]
    new_labs_val = [lab for uttr in labs_val for lab in uttr]
    new_labs_te = [lab for uttr in labs_te for lab in uttr]
    return new_words_tr, new_labs_tr, new_words_val, new_labs_val, new_words_te, new_labs_te


def split_text_sessions(words_tr, labs_tr, words_val, labs_val, words_te, labs_te, session_size, size_limit=None):
    """ Split original train/val/test sessions (both input data and labels) into subsessions of size session_size.
        IF session_size if None, don't split at all, keep original sessions.
        If size_limit is not None, split in half, sessions with length > size_limit.
    """
    if session_size == None:
        new_words_tr = words_tr
        new_labs_tr = labs_tr
        new_words_dev = words_val
        new_labs_dev = labs_val
        new_words_te = words_te
        new_labs_te = labs_te
    else:
        new_words_tr, new_labs_tr = regroup_text_sessions(words_tr, labs_tr, session_size)
        new_words_dev, new_labs_dev = regroup_text_sessions(words_val, labs_val, session_size)
        new_words_te, new_labs_te = regroup_text_sessions(words_te, labs_te, session_size)
    if size_limit is not None:
        new_words_tr, new_labs_tr = split_long_sessions(new_words_tr, new_labs_tr, size_limit)
        new_words_dev, new_labs_dev = split_long_sessions(new_words_dev, new_labs_dev, size_limit)
        new_words_te, new_labs_te = split_long_sessions(new_words_te, new_labs_te, size_limit)
    print("New train sessions: %d" % len(new_words_tr))
    print("New val sessions: %d" % len(new_words_dev))
    print("New test sessions: %d" % len(new_words_te))
    assert count_utterences(words_tr) == count_utterences(new_words_tr)
    assert count_utterences(words_val) == count_utterences(new_words_dev)
    assert count_utterences(words_te) == count_utterences(new_words_te)
    return new_words_tr, new_labs_tr, new_words_dev, new_labs_dev, new_words_te, new_labs_te


def split_text_labels(labs_tr, labs_val, labs_te, session_size, size_limit=None):
    """ Split original train/val/test sessions (only labels) into subsessions of size session_size.
            IF session_size if None, don't split at all, keep original sessions.
            If size_limit is not None, split in half, sessions with length > size_limit.
        """
    if session_size == None:
        new_labs_tr = labs_tr
        new_labs_dev = labs_val
        new_labs_te = labs_te
    else:
        new_labs_tr = regroup_text_labels(labs_tr, session_size)
        new_labs_dev = regroup_text_labels(labs_val, session_size)
        new_labs_te = regroup_text_labels(labs_te, session_size)
    if size_limit is not None:
        new_labs_tr = split_long_sessions_labels(new_labs_tr, size_limit)
        new_labs_dev = split_long_sessions_labels(new_labs_dev, size_limit)
        new_labs_te = split_long_sessions_labels(new_labs_te, size_limit)
    print("New train sessions: %d" % len(new_labs_tr))
    print("New val sessions: %d" % len(new_labs_dev))
    print("New test sessions: %d" % len(new_labs_te))
    assert count_utterences(labs_tr) == count_utterences(new_labs_tr)
    assert count_utterences(labs_val) == count_utterences(new_labs_dev)
    assert count_utterences(labs_te) == count_utterences(new_labs_te)
    return new_labs_tr, new_labs_dev, new_labs_te


def split_long_sessions(sessions, labs, size_limit):
    """ For sessions with size>size_limit, split them (both input data and labels) in half."""
    new_words = []
    new_labs = []
    for i, sess in enumerate(sessions):
        if len(sess) > size_limit:
            new_words.append(sess[0:len(sess) / 2])
            new_labs.append(labs[i][0:len(sess) / 2])
            new_words.append(sess[len(sess) / 2:])
            new_labs.append(labs[i][len(sess) / 2:])
        else:
            new_words.append(sess)
            new_labs.append(labs[i])
    return new_words, new_labs


def split_long_sessions_labels(labs, size_limit):
    """ For sessions with size>size_limit, split them (only labels) in half."""
    new_labs = []
    for i, sess in enumerate(words):
        if len(sess) > size_limit:
            new_labs.append(labs[i][0:len(sess) / 2])
            new_labs.append(labs[i][len(sess) / 2:])
        else:
            new_labs.append(labs[i])
    return new_labs


def split_session(sess, session_size, to_sparse=True):
    import scipy as sp
    indices = [i for i in range(0, len(sess), session_size)]
    splits = []
    for j in range(len(indices) - 1):
        if to_sparse:
            splits.append(sp.sparse.csr_matrix(sess[indices[j]:indices[j + 1]]))
        else:
            splits.append(sess[indices[j]:indices[j + 1]])
    splits.append(sess[indices[len(indices) - 1]:])
    return splits


def split_bow_sessions(data_folder, train_input_file, train_label_file,
                       val_input_file, val_label_file, test_input_file, test_label_file, session_size, to_sparse=True):
    bow_tr, labs_tr, bow_dev, labs_dev, bow_te, labs_te = load_pickle_train_val_test(data_folder, train_input_file,
                                                                                     train_label_file, val_input_file,
                                                                                     val_label_file, test_input_file,
                                                                                     test_label_file)
    if session_size == None:
        new_bow_tr = bow_tr
        new_labs_tr = labs_tr
        new_bow_dev = bow_dev
        new_labs_dev = labs_dev
        new_bow_te = bow_te
        new_labs_te = labs_te
    else:
        new_bow_tr = []
        new_labs_tr = []
        for i, sess in enumerate(bow_tr):
            sess = np.squeeze(np.asarray(sess.todense()))
            new_bow_tr += split_session(sess, session_size, to_sparse)
            new_labs_tr += split_session(labs_tr[i], session_size, to_sparse=False)
        new_bow_dev = []
        new_labs_dev = []
        for i, sess in enumerate(bow_dev):
            sess = np.squeeze(np.asarray(sess.todense()))
            new_bow_dev += split_session(sess, session_size, to_sparse)
            new_labs_dev += split_session(labs_dev[i], session_size, to_sparse=False)
        new_bow_te = []
        new_labs_te = []
        for i, sess in enumerate(bow_te):
            sess = np.squeeze(np.asarray(sess.todense()))
            new_bow_te += split_session(sess, session_size, to_sparse)
            new_labs_te += split_session(labs_te[i], session_size, to_sparse=False)

    assert count_utterences(new_bow_tr) == count_utterences(bow_tr)
    assert count_utterences(new_bow_dev) == count_utterences(bow_dev)
    assert count_utterences(new_bow_te) == count_utterences(bow_te)

    return new_bow_tr, new_labs_tr, new_bow_dev, new_labs_dev, new_bow_te, new_labs_te


def fit_labels(sess_labs):
    lab_dict = {}
    for sess in sess_labs:
        for lab in sess:
            if lab not in lab_dict:
                lab_dict[lab] = len(lab_dict)
    return lab_dict


def transform_labels(sess_labs, lab_dict):
    new_sess_labs = []
    for sess in sess_labs:
        new_labs = []
        for lab in sess: new_labs.append(lab_dict[lab])
        new_sess_labs.append(new_labs)
    return new_sess_labs


def create_bin_sgmt_labels(labs_tr, labs_val, labs_te):
    new_labs_tr = convert_to_bin_sgmt(labs_tr)
    new_labs_val = convert_to_bin_sgmt(labs_val)
    new_labs_te = convert_to_bin_sgmt(labs_te)

    lab_dict = {'B': 1, 'I': 0}
    new_labs_tr = transform_labels(new_labs_tr, lab_dict)
    new_labs_val = transform_labels(new_labs_val, lab_dict)
    new_labs_te = transform_labels(new_labs_te, lab_dict)

    assert count_utterences(labs_tr) == count_utterences(new_labs_tr)
    assert count_utterences(labs_val) == count_utterences(new_labs_val)
    assert count_utterences(labs_te) == count_utterences(new_labs_te)

    return new_labs_tr, new_labs_val, new_labs_te, lab_dict


def create_topic_sgmt_labels(labs_tr, labs_val, labs_te):
    new_labs_tr = convert_to_topic_sgmt(labs_tr)
    new_labs_val = convert_to_topic_sgmt(labs_val)
    new_labs_te = convert_to_topic_sgmt(labs_te)

    lab_dict = fit_labels(new_labs_tr)
    new_labs_tr = transform_labels(new_labs_tr, lab_dict)
    new_labs_val = transform_labels(new_labs_val, lab_dict)
    new_labs_te = transform_labels(new_labs_te, lab_dict)

    assert count_utterences(labs_tr) == count_utterences(new_labs_tr)
    assert count_utterences(labs_val) == count_utterences(new_labs_val)
    assert count_utterences(labs_te) == count_utterences(new_labs_te)

    return new_labs_tr, new_labs_val, new_labs_te, lab_dict


def create_broader_topic_labels(labs_tr, labs_val, labs_te, lab2ltr, lid2lab, lt2ltid):
    new_labs_tr = utils.convert_to_broader_topic(labs_tr, lab2ltr, lid2lab, lt2ltid)
    new_labs_val = utils.convert_to_broader_topic(labs_val, lab2ltr, lid2lab, lt2ltid)
    new_labs_te = utils.convert_to_broader_topic(labs_te, lab2ltr, lid2lab, lt2ltid)

    assert count_utterences(labs_tr) == count_utterences(new_labs_tr)
    assert count_utterences(labs_val) == count_utterences(new_labs_val)
    assert count_utterences(labs_te) == count_utterences(new_labs_te)

    return new_labs_tr, new_labs_val, new_labs_te


def convert_to_bin_sgmt(input_labels, beg_symbol="B", in_symbol="I"):
    bin_sgmt_labels = []
    for sess_labs in input_labels:
        bin_sess_labs = []
        for j, uttr_lab in enumerate(sess_labs):
            if j == 0 or uttr_lab != sess_labs[j - 1]:
                bin_sess_labs.append(beg_symbol)
            else:
                bin_sess_labs.append(in_symbol)
        bin_sgmt_labels.append(bin_sess_labs)
    return bin_sgmt_labels


def convert_to_topic_sgmt(input_labels, beg_suffix="B", in_suffix="I"):
    topic_sgmt_labels = []
    for sess_labs in input_labels:
        topic_sess_labs = []
        for j, uttr_lab in enumerate(sess_labs):
            if j == 0 or uttr_lab != sess_labs[j - 1]:
                topic_sess_labs.append(beg_suffix + "_" + str(uttr_lab))
            else:
                topic_sess_labs.append(in_suffix + "_" + str(uttr_lab))
        topic_sgmt_labels.append(topic_sess_labs)
    return topic_sgmt_labels


def save_sessions_to_text(words_tr, labs_tr, words_val, labs_val, words_te, labs_te, out_folder, suffix):
    save_input_sessions_to_text(words_tr, words_val, words_te, out_folder, suffix)
    save_labels_sessions_to_text(labs_tr, labs_val, labs_te, out_folder, suffix)


def save_labels_sessions_to_text(labs_tr, labs_val, labs_te, out_folder, suffix):
    for i, labs in enumerate(labs_tr):
        labs_tr[i] = [str(l) for l in labs]
    for i, labs in enumerate(labs_te):
        labs_te[i] = [str(l) for l in labs]
    for i, labs in enumerate(labs_val):
        labs_val[i] = [str(l) for l in labs]
    utils.save_sess_labels_to_file(os.path.join(out_folder, "train_target_" + suffix + ".txt"), labs_tr,
                                   uttr_delimiter=" ")
    utils.save_sess_labels_to_file(os.path.join(out_folder, "val_target_" + suffix + ".txt"), labs_val,
                                   uttr_delimiter=" ")
    utils.save_sess_labels_to_file(os.path.join(out_folder, "test_target_" + suffix + ".txt"), labs_te,
                                   uttr_delimiter=" ")


def save_input_sessions_to_text(words_tr, words_val, words_te, out_folder, suffix):
    utils.save_sess_uttrs_to_file(os.path.join(out_folder, "train_input_" + suffix + ".txt"), [[" ".join(uttr) for uttr in sess] for sess in words_tr])
    utils.save_sess_uttrs_to_file(os.path.join(out_folder, "val_input_" + suffix + ".txt"), [[" ".join(uttr) for uttr in sess] for sess in words_val])
    utils.save_sess_uttrs_to_file(os.path.join(out_folder, "test_input_" + suffix + ".txt"), [[" ".join(uttr) for uttr in sess] for sess in words_te])


def save_utterances_to_text(words_tr, labs_tr, words_val, labs_val, words_te, labs_te, out_folder, suffix):
    save_input_utterances_to_text(words_tr, words_val, words_te, out_folder, suffix)
    save_labels_utterances_to_text(labs_tr, labs_val, labs_te, out_folder, suffix)


def save_labels_utterances_to_text(labs_tr, labs_val, labs_te, out_folder, suffix):
    utils.save_to_file(os.path.join(out_folder, "train_target_" + suffix + ".txt"), [str(l) for l in labs_tr])
    utils.save_to_file(os.path.join(out_folder, "val_target_" + suffix + ".txt"), [str(l) for l in labs_val])
    utils.save_to_file(os.path.join(out_folder, "test_target_" + suffix + ".txt"), [str(l) for l in labs_te])


def save_input_utterances_to_text(words_tr, words_val, words_te, out_folder, suffix):
    utils.save_to_file(os.path.join(out_folder, "train_input_" + suffix + ".txt"), [" ".join(uttr) for uttr in words_tr])
    utils.save_to_file(os.path.join(out_folder, "val_input_" + suffix + ".txt"), [" ".join(uttr) for uttr in words_val])
    utils.save_to_file(os.path.join(out_folder, "test_input_" + suffix + ".txt"), [" ".join(uttr) for uttr in words_te])


def get_subset(input, target, size=10):
    indices = np.random.choice(len(input), size, replace=False)
    input_sub = []
    target_sub = []
    for i in indices:
        input_sub.append(input[i])
        target_sub.append(target[i])
    return input_sub, target_sub


def create_subsets(data_folder, out_folder, train_input_file, train_target_file, val_input_file,
                   val_target_file, test_input_file, test_target_file, train_size=10, val_size=4, test_size=4):
    """ Load train/val/test data after preprocessing them into NN's format and create subsets."""
    train_input = utils.load_file(os.path.join(data_folder, train_input_file))
    train_target = utils.load_file(os.path.join(data_folder, train_target_file))
    val_input = utils.load_file(os.path.join(data_folder, val_input_file))
    val_target = utils.load_file(os.path.join(data_folder, val_target_file))
    test_input = utils.load_file(os.path.join(data_folder, test_input_file))
    test_target = utils.load_file(os.path.join(data_folder, test_target_file))
    print("Train sessions: %d " % len(train_input))
    print("Train labels: %d " % len(train_target))
    print("Val sessions: %d " % len(val_input))
    print("Val labels: %d " % len(val_target))
    print("Test sessions: %d " % len(test_input))
    print("Test labels: %d " % len(test_target))

    print("Creating subsets of size (train, val, test) = (%d,%d,%d)" % (train_size, val_size, test_size))
    train_input_sub, train_target_sub = get_subset(train_input, train_target, size=train_size)
    val_input_sub, val_target_sub = get_subset(val_input, val_target, size=val_size)
    test_input_sub, test_target_sub = get_subset(test_input, test_target, size=test_size)
    print os.path.split(train_input_file)
    utils.save_file(os.path.join(out_folder, train_input_file.split(".")[0] + "_sub.txt"), train_input_sub)
    utils.save_file(os.path.join(out_folder, train_target_file.split(".")[0] + "_sub.txt"), train_target_sub)
    utils.save_file(os.path.join(out_folder, val_input_file.split(".")[0] + "_sub.txt"), val_input_sub)
    utils.save_file(os.path.join(out_folder, val_target_file.split(".")[0] + "_sub.txt"), val_target_sub)
    utils.save_file(os.path.join(out_folder, test_input_file.split(".")[0] + "_sub.txt"), test_input_sub)
    utils.save_file(os.path.join(out_folder, test_target_file.split(".")[0] + "_sub.txt"), test_target_sub)
    return train_input_sub, train_target_sub, val_input_sub, val_target_sub, test_input_sub, test_target_sub


def constract_vocabulary(data_folder, input_file, vocab_name, max_freq, min_freq, sessions=True):
    """ Build vocabulary.
        data_folder: folder to read data.
        input_file: input file to read data.
        vocab_name: filename to save vocabulary.
    """
    train_input = utils.load_file(os.path.join(data_folder, input_file))
    print("Train data: %d " % len(train_input))

    # tokenize input data.
    if sessions:
        train_input = tokenize_sessions(train_input)
    else:
        train_input = tokenize_utterances(train_input)

    # build vocabulary from train utterances
    train_input_f = train_input
    if sessions:
        train_input_f = [uttr for sess in train_input for uttr in sess]
    vocab, oov_words = vocabulary(train_input_f, max_freq=max_freq, min_freq=min_freq)
    print("Vocab size: %d " % len(vocab))
    print("Words removed from vocab: %d " % len(oov_words))

    utils.save_file(os.path.join(data_folder, str(max_freq) + str(min_freq) + vocab_name + ".txt"), vocab)
    utils.save_file(os.path.join(data_folder, str(max_freq) + str(min_freq) + vocab_name + "_oov.txt"), oov_words)
    save_vocab_dict(os.path.join(data_folder, str(max_freq) + str(min_freq) + vocab_name + ".txt"),
                    os.path.join(data_folder, str(max_freq) + str(min_freq) + vocab_name + ".pickle"))


def vocabulary(tokenized_seqs, max_freq=0.0, min_freq=0.0):
    # if max_freq is float, remove the max_freq% most frequent words.
    # if max_freq is integer, remove the words with frequency > max_freq.
    # if min_freq is float, remove the min_freq% less frequent words.
    # if min_freq is integer, remove the words with frequency < min_freq.
    # compute word frequencies
    vocab = set()
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_seqs))
    sorted_freqs = sorted(freq_dist.items(), key=lambda x: x[1])
    max_idx = len(vocab) - 1
    min_idx = 0
    if isinstance(max_freq, float):
        max_idx = len(sorted_freqs) - int(np.ceil(max_freq * len(sorted_freqs)))
    else:
        for i, (word, freq) in enumerate(reversed(sorted_freqs)):
            if freq > max_freq:
                max_idx = len(sorted_freqs) - i
                break
    if isinstance(min_freq, float):
        min_idx = int(np.ceil(min_freq * len(sorted_freqs)))
    else:
        for i, (word, freq) in enumerate(sorted_freqs):
            if freq > min_freq:
                min_idx = i
                break
    vocab = [word for (word, freq) in sorted_freqs[min_idx:max_idx]]
    oov_words = [word for (word, freq) in sorted_freqs[0:min_idx] + sorted_freqs[max_idx:]]
    assert len(vocab) + len(oov_words) == len(sorted_freqs)
    return vocab, oov_words


def save_vocab_dict(vocab_file, out_file):
    vocab = utils.load_file(vocab_file)
    vocab = {i: v for i, v in enumerate(vocab)}
    print len(vocab)
    utils.save_object(out_file, vocab)


def avg_sess_uttr_length(sessions):
    sum_sess_len = 0.0
    sum_uttr_len = 0.0
    max_sess = 0.0
    min_sess = 10000000
    max_uttr = 0.0
    min_uttr = 10000000
    for sess in sessions:
        sum_sess_len += len(sess)
        if len(sess) > max_sess: max_sess = len(sess)
        if len(sess) < min_sess: min_sess = len(sess)
        for uttr in sess:
            sum_uttr_len += len(uttr)
            if len(uttr) > max_uttr: max_uttr = len(uttr)
            if len(uttr) < min_uttr: min_uttr = len(uttr)
    return sum_sess_len / len(sessions), max_sess, min_sess, sum_uttr_len / sum_sess_len, max_uttr, min_uttr


def count_utterences(sessions):
    import scipy.sparse
    count = 0
    for sess in sessions:
        if type(sess) is np.ndarray or scipy.sparse.issparse(sess):
            cc = sess.shape[0]
        else:
            cc = len(sess)
        count += cc
    return count


def tokenize_sentences(data, delimiter=" "):
    data_tok = []
    for d in data:
        data_tok.append(d.split(delimiter))
    return data_tok


def tokenize_sessions(sessions, word_delimiter=" ", uttr_delimiter="#"):
    sessions_tok = []
    for sess in sessions:
        sess_tok = []
        for uttr in sess.split(uttr_delimiter):
            sess_tok.append(uttr.split(word_delimiter))
        sessions_tok.append(sess_tok)
    return sessions_tok


def tokenize_utterances(utterances, word_delimiter=" "):
    utterances_tok = []
    for uttr in utterances:
        utterances_tok.append(uttr.split(word_delimiter))
    return utterances_tok


def split_sessions_into_utterances(sessions, delimiter):
    splitted_sessions = []
    for i, sess in enumerate(sessions):
        splitted_sessions[i] = sess.split(delimiter)
    return splitted_sessions


def print_stats(data_folder, train_input_file, val_input_file, test_input_file, is_list_of_sessions):
    """ Print statistics for train/test/van data.
        train_input_file/val_input_file/test_input_file are the mhddata after preprocessing. """

    train_input = utils.load_file(os.path.join(data_folder, train_input_file))
    val_input = utils.load_file(os.path.join(data_folder, val_input_file))
    test_input = utils.load_file(os.path.join(data_folder, test_input_file))
    print("Train data: %d " % len(train_input))
    print("Val data: %d " % len(val_input))
    print("Test data: %d " % len(test_input))

    # tokenize utterances within each session on space
    if is_list_of_sessions:
        train_input = tokenize_sessions(train_input)
        val_input = tokenize_sessions(val_input)
        test_input = tokenize_sessions(test_input)
    else:
        train_input = tokenize_utterances(train_input)
        val_input = tokenize_utterances(val_input)
        test_input = tokenize_utterances(test_input)
    avg_sess_len, max_sess, min_sess, avg_uttr_len, max_uttr, min_uttr = avg_sess_uttr_length(train_input)
    print(
                "Train: avg_sess_len = %.3f, max_sess = %d, min_sess = %d, avg_uttr_len = %.3f, max_uttr = %d, min_uttr = %d " % (
            avg_sess_len, max_sess, min_sess, avg_uttr_len, max_uttr, min_uttr))
    avg_sess_len, max_sess, min_sess, avg_uttr_len, max_uttr, min_uttr = avg_sess_uttr_length(val_input)
    print(
                "Dev: avg_sess_len = %.3f, max_sess = %d, min_sess = %d, avg_uttr_len = %.3f, max_uttr = %d, min_uttr = %d " % (
            avg_sess_len, max_sess, min_sess, avg_uttr_len, max_uttr, min_uttr))
    avg_sess_len, max_sess, min_sess, avg_uttr_len, max_uttr, min_uttr = avg_sess_uttr_length(test_input)
    print(
                "Test: avg_sess_len = %.3f, max_sess = %d, min_sess = %d, avg_uttr_len = %.3f, max_uttr = %d, min_uttr = %d " % (
            avg_sess_len, max_sess, min_sess, avg_uttr_len, max_uttr, min_uttr))

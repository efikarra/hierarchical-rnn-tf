import nltk
import itertools
import numpy as np
import os
import utils


def load_pickle_data(data_folder, train_input_file, train_label_file,
                     val_input_file, val_label_file, test_input_file, test_label_file):
    import cPickle
    with open(os.path.join(data_folder,train_input_file), "rb") as f:
        tr = cPickle.load(f)
    print("% s loaded."%train_input_file)
    with open(os.path.join(data_folder,val_input_file), "rb") as f:
        val = cPickle.load(f)
    print("% s loaded." % val_input_file)
    with open(os.path.join(data_folder,test_input_file), "rb") as f:
        te = cPickle.load(f)
    print("% s loaded." % test_input_file)
    with open(os.path.join(data_folder,train_label_file), "rb") as f:
        labs_tr = cPickle.load(f)
    print("% s loaded." % train_label_file)
    with open(os.path.join(data_folder,val_label_file), "rb") as f:
        labs_val = cPickle.load(f)
    print("% s loaded." % val_label_file)
    with open(os.path.join(data_folder,test_label_file), "rb") as f:
        labs_te = cPickle.load(f)
    print("% s loaded." % test_label_file)
    return tr,labs_tr,val,labs_val,te,labs_te


def save_pickle_data(train_input, train_labels, val_input, val_labels, test_input, test_labels, out_folder, suffix):
    import cPickle
    with open(os.path.join(out_folder,"train_"+suffix+".pickle"), "wb") as f:
        cPickle.dump(train_input, f)
    with open(os.path.join(out_folder,"val_"+suffix+".pickle"), "wb") as f:
        cPickle.dump(val_input, f)
    with open(os.path.join(out_folder,"test_"+suffix+".pickle"), "wb") as f:
        cPickle.dump(test_input, f)
    with open(os.path.join(out_folder,"train_lab_"+suffix+".pickle"), "wb") as f:
        cPickle.dump(train_labels, f)
    with open(os.path.join(out_folder,"val_lab_"+suffix+".pickle"), "wb") as f:
        cPickle.dump(val_labels, f)
    with open(os.path.join(out_folder,"test_lab_"+suffix+".pickle"), "wb") as f:
        cPickle.dump(test_labels, f)
    print("Done saving")


def regroup_text_sessions(sessions,labels,session_size):
    new_sessions = []
    new_labels = []
    for i,sess in enumerate(sessions):
        count = 1
        new_sess = []
        new_lab = []
        for j,uttr in enumerate(sess):
            new_sess.append(uttr)
            new_lab.append(labels[i][j])
            if count % session_size == 0 or count == len(sess):
                new_sessions.append(new_sess)
                new_labels.append(new_lab)
                new_sess = []
                new_lab = []
            count += 1
    return new_sessions,new_labels


def flatten_text_sessions(data_folder, out_folder, suffix):
    words_tr, labs_tr, words_val, labs_val, words_te, labs_te = load_pickle_data(data_folder, "splits_words_tr.pickle",
                                                                                 "splits_labs_tr.pickle",
                                                                                 "splits_words_dev.pickle",
                                                                                 "splits_labs_dev.pickle",
                                                                                 "splits_words_te.pickle",
                                                                                 "splits_labs_te.pickle")
    words_tr = [" ".join(word) for uttr in words_tr for word in uttr]
    words_val = [" ".join(word) for uttr in words_val for word in uttr]
    words_te = [" ".join(word) for uttr in words_te for word in uttr]
    labs_tr = [str(lab) for uttr in labs_tr for lab in uttr]
    labs_val = [str(lab) for uttr in labs_val for lab in uttr]
    labs_te = [str(lab) for uttr in labs_te for lab in uttr]
    utils.save_to_file(os.path.join(out_folder,"train_input_"+suffix+".txt"), words_tr)
    utils.save_to_file(os.path.join(out_folder, "train_target_" + suffix + ".txt"), labs_tr)
    utils.save_to_file(os.path.join(out_folder,"val_input_"+suffix+".txt"), words_val)
    utils.save_to_file(os.path.join(out_folder,"val_target_"+suffix+".txt"), labs_val)
    utils.save_to_file(os.path.join(out_folder,"test_input_"+suffix+".txt"), words_te)
    utils.save_to_file(os.path.join(out_folder,"test_target_"+suffix+".txt"), labs_te)



def split_text_sessions(data_folder,session_size):
    words_tr, labs_tr, words_dev, labs_dev, words_te, labs_te = load_pickle_data(data_folder,"splits_words_tr.pickle",
                                                                                 "splits_labs_tr.pickle",
                                                                                 "splits_words_dev.pickle",
                                                                                 "splits_labs_dev.pickle",
                                                                                "splits_words_te.pickle",
                                                                                 "splits_labs_te.pickle")
    new_words_tr, new_labs_tr = regroup_text_sessions(words_tr, labs_tr, session_size)
    assert count_utterences(words_tr)==count_utterences(new_words_tr)
    new_words_dev, new_labs_dev =regroup_text_sessions(words_dev, labs_dev, session_size)
    assert count_utterences(words_dev) == count_utterences(new_words_dev)
    new_words_te, new_labs_te =regroup_text_sessions(words_te, labs_te, session_size)
    assert count_utterences(words_te) == count_utterences(new_words_te)
    return new_words_tr,new_labs_tr,new_words_dev,new_labs_dev,new_words_te, new_labs_te


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


def split_bow_sessions(data_folder,session_size,to_sparse=True):
    bow_tr, labs_tr, bow_dev, labs_dev, bow_te, labs_te = load_pickle_data(data_folder,
                                                                                      "splits_bow_tr.pickle",
                                                                                      "splits_labs_tr.pickle",
                                                                                      "splits_bow_dev.pickle",
                                                                                      "splits_labs_dev.pickle",
                                                                                      "splits_bow_te.pickle",
                                                                                      "splits_labs_te.pickle")
    new_bow_tr=[]
    new_labs_tr=[]
    for i,sess in enumerate(bow_tr):
        sess = np.squeeze(np.asarray(sess.todense()))
        new_bow_tr+=split_session(sess, session_size, to_sparse)
        new_labs_tr+=split_session(labs_tr[i], session_size, to_sparse=False)
    new_bow_dev = []
    new_labs_dev = []
    for i,sess in enumerate(bow_dev):
        sess = np.squeeze(np.asarray(sess.todense()))
        new_bow_dev+=split_session(sess, session_size,to_sparse)
        new_labs_dev+=split_session(labs_dev[i], session_size, to_sparse=False)
    new_bow_te = []
    new_labs_te = []
    for i,sess in enumerate(bow_te):
        sess = np.squeeze(np.asarray(sess.todense()))
        new_bow_te+=split_session(sess, session_size,to_sparse)
        new_labs_te+=split_session(labs_te[i], session_size, to_sparse=False)

    assert count_utterences(new_bow_tr) == count_utterences(bow_tr)
    assert count_utterences(new_bow_dev) == count_utterences(bow_dev)
    assert count_utterences(new_bow_te) == count_utterences(bow_te)

    return new_bow_tr,new_labs_tr,new_bow_dev,new_labs_dev,new_bow_te, new_labs_te



def convert_mhddata_to_text(words_tr, labs_tr, words_dev, labs_dev, words_te, labs_te, out_folder, suffix):

    for i, labs in enumerate(labs_tr):
        labs_tr[i] = [str(l) for l in labs]
    for i, labs in enumerate(labs_te):
        labs_te[i] = [str(l) for l in labs]
    for i, labs in enumerate(labs_dev):
        labs_dev[i] = [str(l) for l in labs]

    utils.save_sess_words_to_file(os.path.join(out_folder,"train_input_"+suffix+".txt"), words_tr)
    utils.save_sess_words_to_file(os.path.join(out_folder,"val_input_"+suffix+".txt"), words_te)
    utils.save_sess_words_to_file(os.path.join(out_folder,"test_input_"+suffix+".txt"), words_dev)
    utils.save_sess_labels_to_file(os.path.join(out_folder,"train_target_"+suffix+".txt"), labs_tr, uttr_delimiter=" ")
    utils.save_sess_labels_to_file(os.path.join(out_folder,"val_target_"+suffix+".txt"), labs_te, uttr_delimiter=" ")
    utils.save_sess_labels_to_file(os.path.join(out_folder,"test_target_"+suffix+".txt"), labs_dev, uttr_delimiter=" ")


def get_subset(input, target, size=10):
    indices = np.random.choice(len(input),size, replace=False)
    input_sub = []
    target_sub = []
    for i in indices:
        input_sub.append(input[i])
        target_sub.append(target[i])
    return input_sub,target_sub


def create_subsets(data_folder, out_folder, train_input_file, train_target_file, val_input_file,
                   val_target_file, test_input_file, test_target_file, train_size=10,val_size=4,test_size=4):
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

    print("Creating subsets of size (train, val, test) = (%d,%d,%d)"%(train_size,val_size,test_size))
    train_input_sub, train_target_sub = get_subset(train_input, train_target, size=train_size)
    val_input_sub, val_target_sub = get_subset(val_input, val_target, size=val_size)
    test_input_sub, test_target_sub = get_subset(test_input, test_target, size=test_size)
    print os.path.split(train_input_file)
    utils.save_file(os.path.join(out_folder, train_input_file.split(".")[0] + "_sub.txt"), train_input_sub)
    utils.save_file(os.path.join(out_folder, train_target_file.split(".")[0] + "_sub.txt"), train_input_sub)
    utils.save_file(os.path.join(out_folder, val_input_file.split(".")[0] + "_sub.txt"), val_input_sub)
    utils.save_file(os.path.join(out_folder, val_target_file.split(".")[0] + "_sub.txt"), val_target_sub)
    utils.save_file(os.path.join(out_folder, test_input_file.split(".")[0] + "_sub.txt"), test_input_sub)
    utils.save_file(os.path.join(out_folder, test_target_file.split(".")[0] + "_sub.txt"), test_target_sub)
    return train_input_sub, train_target_sub, val_input_sub, val_target_sub, test_input_sub, test_target_sub


def create_ovr_data(data_folder, train_target_file, val_target_file, test_target_file,classes):
    train_target = utils.load_file(os.path.join(data_folder, train_target_file))
    val_target = utils.load_file(os.path.join(data_folder, val_target_file))
    test_target = utils.load_file(os.path.join(data_folder, test_target_file))
    for c in classes:
        train_target_ovr = convert_to_ovr(class_one=c, input_labels=train_target)
        val_target_ovr = convert_to_ovr(class_one=c, input_labels=val_target)
        test_target_ovr = convert_to_ovr(class_one=c, input_labels=test_target)
        utils.save_file(os.path.join("experiments/ovr_targets", str(c) + "_" + train_target_file), train_target_ovr)
        utils.save_file(os.path.join("experiments/ovr_targets", str(c) + "_" + val_target_file), val_target_ovr)
        utils.save_file(os.path.join("experiments/ovr_targets", str(c) + "_" + test_target_file), test_target_ovr)

def preprocess_text_data(data_folder, train_input_file, train_target_file, val_input_file,
                         val_target_file, test_input_file, test_target_file, vocab_file, max_freq, min_freq,sessions=True):

    # load a list of sessions where each session is a list of utterances
    train_input = utils.load_file(os.path.join(data_folder, train_input_file))
    train_target = utils.load_file(os.path.join(data_folder, train_target_file))
    val_input = utils.load_file(os.path.join(data_folder, val_input_file))
    val_target = utils.load_file(os.path.join(data_folder, val_target_file))
    test_input = utils.load_file(os.path.join(data_folder, test_input_file))
    test_target = utils.load_file(os.path.join(data_folder, test_target_file))
    print("Train data: %d " % len(train_input))
    print("Train labels: %d " % len(train_target))
    print("Val data: %d " % len(val_input))
    print("Val labels: %d " % len(val_target))
    print("Test data: %d " % len(test_input))
    print("Test labels: %d " % len(test_target))

    # tokenize utterances within each session on space
    if sessions:
        train_input = tokenize_sessions(train_input)
        val_input = tokenize_sessions(val_input)
        test_input = tokenize_sessions(test_input)
    else:
        train_input = tokenize_utterances(train_input)
        val_input = tokenize_utterances(val_input)
        test_input = tokenize_utterances(test_input)

    # build vocabulary from train utterances
    train_input_f = train_input
    if sessions:
        train_input_f = [uttr for sess in train_input for uttr in sess]
    vocab = build_vocabulary(train_input_f, max_freq=max_freq, min_freq=min_freq)
    print("Vocab size: %d " % len(vocab))

    utils.save_file(os.path.join(data_folder, str(max_freq) + str(min_freq) + vocab_file), vocab)
    print_stats(train_input, val_input, test_input)


def build_vocabulary(tokenized_seqs, max_freq=0.0, min_freq=0.0):
    # compute word frequencies
    vocab=set()
    freq_dist=nltk.FreqDist(itertools.chain(*tokenized_seqs))
    sorted_words = dict(sorted(freq_dist.items(), key=lambda x: x[1])).keys()
    max_idx=len(vocab)-1
    min_idx=0
    if isinstance(max_freq, float):
        max_idx=len(sorted_words)-int(np.ceil(max_freq * len(sorted_words)))
    if isinstance(min_freq, float):
        min_idx=int(np.ceil(min_freq * len(sorted_words)))
    vocab=sorted_words[min_idx:max_idx]
    return vocab


def avg_sess_uttr_length(sessions):
    sum_sess_len=0.0
    sum_uttr_len = 0.0
    max_sess=0.0
    min_sess=10000000
    max_uttr = 0.0
    min_uttr = 10000000
    for sess in sessions:
        sum_sess_len+=len(sess)
        if len(sess) > max_sess: max_sess = len(sess)
        if len(sess) < min_sess: min_sess = len(sess)
        for uttr in sess:
            sum_uttr_len+=len(uttr)
            if len(uttr)>max_uttr:max_uttr=len(uttr)
            if len(uttr) < min_uttr: min_uttr = len(uttr)
    return sum_sess_len/len(sessions),max_sess,min_sess,sum_uttr_len/sum_sess_len,max_uttr,min_uttr

def count_utterences(sessions):
    import scipy.sparse
    count=0
    for sess in sessions:
        if type(sess) is np.ndarray or scipy.sparse.issparse(sess) : cc=sess.shape[0]
        else: cc=len(sess)
        count += cc
    return count


def convert_to_ovr(class_one, input_labels):
    ovr_labels = []
    for l in input_labels:
        ovr_labels.append('1' if int(l)==class_one else '0')
    return ovr_labels


def tokenize_sentences(data, delimiter=" "):
    data_tok=[]
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
    splitted_sessions=[]
    for i,sess in enumerate(sessions):
        splitted_sessions[i] = sess.split(delimiter)
    return splitted_sessions


def print_stats(train_input,val_input,test_input):
    avg_sess_len, max_sess, min_sess, avg_uttr_len, max_uttr, min_uttr = avg_sess_uttr_length(train_input)
    print("Train: avg_sess_len = %.3f, max_sess = %d, min_sess = %d, avg_uttr_len = %.3f, max_uttr = %d, min_uttr = %d " % (
        avg_sess_len, max_sess, min_sess, avg_uttr_len, max_uttr, min_uttr))
    avg_sess_len, max_sess, min_sess, avg_uttr_len, max_uttr, min_uttr = avg_sess_uttr_length(val_input)
    print("Dev: avg_sess_len = %.3f, max_sess = %d, min_sess = %d, avg_uttr_len = %.3f, max_uttr = %d, min_uttr = %d " % (
        avg_sess_len, max_sess, min_sess, avg_uttr_len, max_uttr, min_uttr))
    avg_sess_len, max_sess, min_sess, avg_uttr_len, max_uttr, min_uttr = avg_sess_uttr_length(test_input)
    print("Test: avg_sess_len = %.3f, max_sess = %d, min_sess = %d, avg_uttr_len = %.3f, max_uttr = %d, min_uttr = %d " % (
        avg_sess_len, max_sess, min_sess,avg_uttr_len, max_uttr, min_uttr))
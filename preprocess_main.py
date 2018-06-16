""" Use this to preprocess mhddata and create data in the input format that NNs need."""
import preprocess

def preprocess_text_sess_data(session_size, size_limit=None):
    """ Load mhddata data and create text files for hierarchical models.
        session_size: size of subsessions to split the original sessions.
        size_limit: if not None, split
    """
    words_tr, labs_tr, words_dev, labs_dev, words_te, labs_te = preprocess.load_pickle_train_val_test(
        "experiments/data/mhddata/",
        "splits_words_tr.pickle", "splits_labs_tr.pickle",
        "splits_words_dev.pickle", "splits_labs_dev.pickle", "splits_words_te.pickle",
        "splits_labs_te.pickle")
    new_words_tr, new_labs_tr, new_words_val, new_labs_val, new_words_te, new_labs_te \
        = preprocess.split_text_sessions(words_tr, labs_tr, words_dev, labs_dev, words_te, labs_te
                                         , session_size=session_size, size_limit=size_limit)

    preprocess.save_sessions_to_text(new_words_tr, new_labs_tr, new_words_val, new_labs_val, new_words_te,
                                     new_labs_te,
                                     "experiments/data",
                                     "sess_" + (str(session_size) if session_size is not None else "full"))


def preprocess_text_uttr_data():
    """ Load mhddata data and create text files for non-hierarchical models."""
    words_tr, labs_tr, words_val, labs_val, words_te, labs_te = preprocess.load_pickle_train_val_test(
        "experiments/data/mhddata/", "splits_words_tr.pickle",
        "splits_labs_tr.pickle", "splits_words_dev.pickle", "splits_labs_dev.pickle",
        "splits_words_te.pickle", "splits_labs_te.pickle")
    new_words_tr, new_labs_tr, new_words_val, new_labs_val, new_words_te, new_labs_te = \
        preprocess.flatten_text_sessions(words_tr, labs_tr, words_val, labs_val, words_te, labs_te)
    preprocess.save_utterances_to_text(new_words_tr, new_labs_tr, new_words_val, new_labs_val, new_words_te,
                                       new_labs_te,
                                       out_folder="experiments/data", suffix="uttr")


def preprocess_bow_sess_data(session_size):
    new_bow_tr, new_labs_tr, new_bow_dev, new_labs_dev, new_bow_te, new_labs_te = \
        preprocess.split_bow_sessions("experiments/data/mhddata/", "splits_bow_tr.pickle", "splits_labs_tr.pickle",
                                      "splits_bow_dev.pickle", "splits_labs_dev.pickle", "splits_bow_te.pickle",
                                      "splits_labs_te.pickle", session_size=session_size)
    preprocess.save_pickle_train_val_test(new_bow_tr, new_labs_tr, new_bow_dev, new_labs_dev, new_bow_te,
                                          new_labs_te, "experiments/data/", suffix="sess_" + (str(session_size)
                                                                                              if session_size is not None else "full"))


def preprocess_labels_segmentation(session_size, joint_labels=True):
    """ load mhddata labels and create labels for utterance segmentation classification.
        joint_labels: False if binary labels 1/0 for begin/inside segment.
        joint_labels: True if doubling the number of labels by creating two new labels
        for each existing label C: Begin_C/Inside_C. """

    labs_tr, labs_val, labs_te = preprocess.load_pickle_train_val_test_labels("experiments/data/mhddata/",
                                                                              "splits_labs_tr.pickle",
                                                                              "splits_labs_dev.pickle",
                                                                              "splits_labs_te.pickle")
    if joint_labels:
        new_labs_tr, new_labs_val, new_labs_te, lab_dict = preprocess.create_topic_sgmt_labels(labs_tr, labs_val,
                                                                                               labs_te)
    else:
        new_labs_tr, new_labs_val, new_labs_te, lab_dict = preprocess.create_bin_sgmt_labels(labs_tr, labs_val, labs_te)
    new_labs_tr, new_labs_val, new_labs_te = \
        preprocess.split_text_labels(new_labs_tr, new_labs_val, new_labs_te, session_size, size_limit=None)

    name_suffix = "sess_" + (str(session_size) if session_size is not None else "full") + (
        "_sgmt_mult" if joint_labels else "_sgmt_bin")
    preprocess.save_labels_sessions_to_text(new_labs_tr, new_labs_val, new_labs_te, "experiments/data/", name_suffix)
    import cPickle
    with open("experiments/data/lid2lab_" + ("sgmt_mult.pickle" if joint_labels else "sgmt_bin.pickle"), "wb") as f:
        cPickle.dump(lab_dict, f)


def preprocess_labels_broader_topics(session_size=None):
    """ preprocess mhddata labels for utterance classification into the broader topic categories."""
    import cPickle
    labs_tr, labs_val, labs_te = preprocess.load_pickle_train_val_test_labels("experiments/data/mhddata/",
                                                                              "splits_labs_tr.pickle",
                                                                              "splits_labs_dev.pickle",
                                                                              "splits_labs_te.pickle")
    lab2ltr = cPickle.load(open("experiments/data/mhddata/mhddata_lab2ltr.pickle", "rb"))
    lid2lab = cPickle.load(open("experiments/data/mhddata/mhddata_lid2lab.pickle", "rb"))
    lt2ltid = cPickle.load(open("experiments/data/mhddata/mhddata_lt2ltid.pickle", "rb"))
    new_labs_tr, new_labs_val, new_labs_te = preprocess. \
        create_broader_topic_labels(labs_tr, labs_val, labs_te, lab2ltr, lid2lab, lt2ltid)
    name_suffix = "sess_" + (str(session_size) if session_size is not None else "full") + "_lt"
    new_labs_tr, new_labs_val, new_labs_te = \
        preprocess.split_text_labels(new_labs_tr, new_labs_val, new_labs_te,
                                     session_size, size_limit=None)
    preprocess.save_labels_sessions_to_text(new_labs_tr, new_labs_val, new_labs_te, "experiments/data/", name_suffix)


if __name__ == '__main__':
    # preprocess mhddata for non-hierarchical utterance classification models.
    preprocess_text_uttr_data()
    # preprocess mhddata for hierarchical utterance classification models.
    preprocess_text_sess_data(session_size=400, size_limit=800)
    # build vocabulary based on train set.
    preprocess.constract_vocabulary('experiments/data/', "train_input_uttr.txt", "vocab", max_freq=0.0,
                                    min_freq=0.1, sessions=False)

    # preprocess mhddata labels for utterance segmentation classification.
    preprocess_labels_segmentation(session_size=400, joint_labels=False)

    # preprocess mhddata for utterance classification into broader topic categories.
    # Not used for now. So far we haven't trained on the broader topics. We just trained on the original topics and we mapped the predictions into
    # the broader topics for evaluation.
    # preprocess_labels_broader_topics()

    # vocab_file = "vocab"
    # preprocess.create_subsets('experiments/data', 'experiments/data/subsets', "train_input_sess_100.txt", "train_target_sess_100.txt",
    #                           "val_input_sess_100.txt", "val_target_sess_100.txt", "test_input_sess_100.txt", "test_target_sess_100.txt",
    #                           train_size=10, val_size=4,
    #                           test_size=4)
    # preprocess.constract_vocabulary('experiments/data/subsets', "train_input_sess_100_sub.txt","vocab_sub", max_freq=0.0,
    #                                 min_freq=0.1, sessions=True)

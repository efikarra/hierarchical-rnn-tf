""" Use this to preprocess mhddata and create data in the input format that NNs need."""
import preprocess


def preprocess_text_sess_data(session_size, data_folder="experiments/data/mhddata/", out_folder="experiments/data/",
                              tr_input_fname="splits_words_tr.pickle", tr_labels_fname="splits_labs_tr.pickle",
                              val_input_fname="splits_words_dev.pickle", val_labels_fname="splits_labs_dev.pickle",
                              te_input_fname="splits_words_te.pickle", te_labels_fname="splits_labs_te.pickle",
                              size_limit=None):
    """ Load mhddata data and create text files for hierarchical models.
        session_size: size of subsessions to split the original sessions.
        size_limit: if not None, split
    """
    words_tr, labs_tr, words_dev, labs_dev, words_te, labs_te = preprocess.load_pickle_train_val_test(
        data_folder, tr_input_fname, tr_labels_fname, val_input_fname, val_labels_fname, te_input_fname,
        te_labels_fname)
    new_words_tr, new_labs_tr, new_words_val, new_labs_val, new_words_te, new_labs_te \
        = preprocess.split_text_sessions(words_tr, labs_tr, words_dev, labs_dev, words_te, labs_te
                                             , session_size=session_size, size_limit=size_limit)

    preprocess.save_sessions_to_text(new_words_tr, new_labs_tr, new_words_val, new_labs_val, new_words_te,
                                         new_labs_te,
                                         out_folder,
                                     "sess_" + (str(session_size) if session_size is not None else "full"))


def preprocess_text_uttr_data(data_folder="experiments/data/mhddata/", out_folder="experiments/data/",
                              tr_input_fname="splits_words_tr.pickle", tr_labels_fname="splits_labs_tr.pickle",
                              val_input_fname="splits_words_dev.pickle", val_labels_fname="splits_labs_dev.pickle",
                              te_input_fname="splits_words_te.pickle", te_labels_fname="splits_labs_te.pickle"):
    """ Load mhddata data and create text files for non-hierarchical models."""
    words_tr, labs_tr, words_dev, labs_dev, words_te, labs_te = preprocess.load_pickle_train_val_test(
        data_folder, tr_input_fname, tr_labels_fname, val_input_fname, val_labels_fname, te_input_fname,
        te_labels_fname)
    new_words_tr, new_labs_tr, new_words_val, new_labs_val, new_words_te, new_labs_te = \
        preprocess.flatten_text_sessions(words_tr, labs_tr, words_dev, labs_dev, words_te, labs_te)
    preprocess.save_utterances_to_text(new_words_tr, new_labs_tr, new_words_val, new_labs_val, new_words_te,
                                           new_labs_te,
                                           out_folder=out_folder, suffix="uttr")


def preprocess_bow_sess_data(session_size, data_folder="experiments/data/mhddata/", out_folder="experiments/data/",
                             tr_input_fname="splits_bow_tr.pickle", tr_labels_fname="splits_labs_tr.pickle",
                             val_input_fname="splits_bow_dev.pickle", val_labels_fname="splits_labs_dev.pickle",
                             te_input_fname="splits_bow_te.pickle", te_labels_fname="splits_labs_te.pickle"):
    new_bow_tr, new_labs_tr, new_bow_dev, new_labs_dev, new_bow_te, new_labs_te = \
        preprocess.split_bow_sessions(data_folder, tr_input_fname, tr_labels_fname,
                                          val_input_fname, val_labels_fname, te_input_fname,
                                          te_labels_fname, session_size=session_size)
    preprocess.save_pickle_train_val_test(new_bow_tr, new_labs_tr, new_bow_dev, new_labs_dev, new_bow_te,
                                              new_labs_te, out_folder, suffix="sess_" + (str(session_size)
                                                                                     if session_size is not None else "full"))


def preprocess_labels_segmentation(session_size, data_folder="experiments/data/mhddata/",
                                   out_folder="experiments/data/",
                                   tr_labels_fname="splits_labs_tr.pickle", val_labels_fname="splits_labs_dev.pickle",
                                   te_labels_fname="splits_labs_te.pickle", joint_labels=True):
    """ load mhddata labels and create labels for utterance segmentation classification.
        joint_labels: False if binary labels 1/0 for begin/inside segment.
        joint_labels: True for BIO style labels. For each existing label, two new labels are created:
         Begin_C/Inside_C. """

    labs_tr, labs_val, labs_te = preprocess.load_pickle_train_val_test_labels(data_folder,
                                                                                  tr_labels_fname,
                                                                                  val_labels_fname,
                                                                                  te_labels_fname)
    if joint_labels:
        new_labs_tr, new_labs_val, new_labs_te, lab_dict = preprocess.create_topic_sgmt_labels(labs_tr, labs_val,
                                                                                                   labs_te)
    else:
        new_labs_tr, new_labs_val, new_labs_te, lab_dict = preprocess.create_bin_sgmt_labels(labs_tr, labs_val, labs_te)
    new_labs_tr, new_labs_val, new_labs_te = \
        preprocess.split_text_labels(new_labs_tr, new_labs_val, new_labs_te, session_size, size_limit=None)

    name_suffix = "sess_" + (str(session_size) if session_size is not None else "full") + (
        "_sgmt_mult" if joint_labels else "_sgmt_bin")
    preprocess.save_labels_sessions_to_text(new_labs_tr, new_labs_val, new_labs_te, out_folder, name_suffix)
    import cPickle
    with open("experiments/data/lid2lab_" + ("sgmt_mult.pickle" if joint_labels else "sgmt_bin.pickle"), "wb") as f:
        cPickle.dump(lab_dict, f)


def preprocess_labels_broader_topics(data_folder="experiments/data/mhddata/", out_folder="experiments/data/",
                                     tr_labels_fname="splits_labs_tr.pickle", val_labels_fname="splits_labs_dev.pickle",
                                     te_labels_fname="splits_labs_te.pickle", session_size=None):
    """ preprocess mhddata labels for utterance classification into the broader topic categories."""
    import cPickle
    labs_tr, labs_val, labs_te = preprocess.load_pickle_train_val_test_labels(data_folder,
                                                                                  tr_labels_fname,
                                                                                  val_labels_fname,
                                                                                  te_labels_fname)
    lab2ltr = cPickle.load(open("experiments/data/mhddata/mhddata_lab2ltr.pickle", "rb"))
    lid2lab = cPickle.load(open("experiments/data/mhddata/mhddata_lid2lab.pickle", "rb"))
    lt2ltid = cPickle.load(open("experiments/data/mhddata/mhddata_lt2ltid.pickle", "rb"))
    new_labs_tr, new_labs_val, new_labs_te = preprocess. \
        create_broader_topic_labels(labs_tr, labs_val, labs_te, lab2ltr, lid2lab, lt2ltid)
    name_suffix = "sess_" + (str(session_size) if session_size is not None else "full") + "_lt"
    new_labs_tr, new_labs_val, new_labs_te = \
        preprocess.split_text_labels(new_labs_tr, new_labs_val, new_labs_te,
                                         session_size, size_limit=None)
    preprocess.save_labels_sessions_to_text(new_labs_tr, new_labs_val, new_labs_te, out_folder, name_suffix)


if __name__ == '__main__':
    # folder in which mhddata are saved.
    mhd_data_folder = "experiments/data/mhddata/"
    # folder to save the preprocessed data. These data will be the input to the NN models.
    out_folder = "experiments/data/"
    # preprocess mhddata for non-hierarchical utterance classification models (simple RNN and CNN).
    # Data will be saved into text files with one utterance per line.
    preprocess_text_uttr_data(mhd_data_folder, out_folder)
    # preprocess mhddata for hierarchical utterance classification models. If session_size is not None, original sessions
    # will be splitted into subsessions of session_size.
    # Data will be saved into text files with one session per line. Utterances are separated by '#' symbol.
    preprocess_text_sess_data(session_size=400, data_folder=mhd_data_folder, out_folder=out_folder, size_limit=800)

    # preprocess mhddata for hierarchical utterance classification models that take as input bag of words features.
    # If session_size is not None, original sessions (containing bag of words) will be splitted into subsessions of session_size.
    # Data will be saved into numpy arrays.
    preprocess_bow_sess_data(session_size=400, data_folder=mhd_data_folder, out_folder=out_folder)

    # preprocess mhddata labels for utterance segmentation classification. set joint_labels=True for BIO style labels
    # or set joint_labels=False for binary labels 1/0 for begin/inside of segment.
    preprocess_labels_segmentation(session_size=400, joint_labels=False, data_folder=mhd_data_folder,
                                   out_folder=out_folder)

    # build and save vocabulary in the out_folder based on train set.
    preprocess.constract_vocabulary(out_folder=out_folder, input_path="experiments/data/train_input_uttr.txt",
                                        vocab_name="vocab", max_freq=0.0,
                                        min_freq=0.1, sessions=False)

    # preprocess mhddata labels for utterance classification into the 7 broader topic categories.
    # Not used for now. So far we haven't trained on the broader topics. We just trained on the original topics
    # and we mapped the predictions into the broader topics for evaluation.
    # preprocess_labels_broader_topics(data_folder=mhd_data_folder, out_folder=out_folder)

    # create subsets of the preprocessed data for quick tests of the models.
    subsets_out_folder = 'experiments/data/subsets'
    preprocess.create_subsets(out_folder, subsets_out_folder, "train_input_sess_400.txt", "train_target_sess_400.txt",
                              "val_input_sess_400.txt", "val_target_sess_400.txt", "test_input_sess_400.txt",
                              "test_target_sess_400.txt",
                                  train_size=10, val_size=4,
                                  test_size=4)
    # create vocabulary for subsets.
    preprocess.constract_vocabulary(out_folder=subsets_out_folder,
                                        input_path=subsets_out_folder + "/train_input_sess_400_sub.txt",
                                        vocab_name="vocab_sub", max_freq=0.0, min_freq=0.1, sessions=True)

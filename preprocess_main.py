import preprocess
import argparse
import os

import utils

def preprocess_text_sess_data(session_size):
    new_words_tr, new_labs_tr, new_words_dev, new_labs_dev, new_words_te, new_labs_te\
        =preprocess.split_text_sessions("experiments/mhddata_pickle/", session_size=session_size)

    preprocess.convert_mhddata_to_text(new_words_tr, new_labs_tr, new_words_dev, new_labs_dev, new_words_te,
                                       new_labs_te,
                                    "experiments/data", "sess_"+str(session_size))
    data_folder = 'experiments/data'
    train_input_file="train_input_sess_"+str(session_size)+".txt"
    train_target_file="train_target_sess_"+str(session_size)+".txt"
    val_input_file="val_input_sess_"+str(session_size)+".txt"
    val_target_file="val_target_sess_"+str(session_size)+".txt"
    test_input_file="test_input_sess_"+str(session_size)+".txt"
    test_target_file="test_target_sess_"+str(session_size)+".txt"
    vocab_file="vocab_sess_"+str(session_size)+".txt"
    preprocess.preprocess_text_data(data_folder, train_input_file, train_target_file, val_input_file,
                         val_target_file, test_input_file, test_target_file, vocab_file, max_freq=0.0, min_freq=0.0,sessions=True)



def preprocess_text_uttr_data():
    preprocess.flatten_text_sessions("experiments/mhddata_pickle/", "experiments/data", suffix="uttr")
    data_folder = 'experiments/data'
    train_input_file = 'train_input_uttr.txt'
    train_target_file = 'train_target_uttr.txt'
    val_input_file = 'val_input_uttr.txt'
    val_target_file = 'val_target_uttr.txt'
    test_input_file = 'test_input_uttr.txt'
    test_target_file = 'test_target_uttr.txt'
    vocab_file = 'vocab_uttr.txt'
    preprocess.preprocess_text_data(data_folder, train_input_file, train_target_file, val_input_file,
                                    val_target_file, test_input_file, test_target_file, vocab_file, max_freq=0.0,
                                    min_freq=0.0, sessions=False)


def preprocess_bow_sess_data(session_size):
    new_bow_tr, new_labs_tr, new_bow_dev, new_labs_dev, new_bow_te, new_labs_te = \
        preprocess.split_bow_sessions("experiments/mhddata_pickle/", session_size=session_size)
    preprocess.save_pickle_data(new_bow_tr, new_labs_tr, new_bow_dev, new_labs_dev, new_bow_te,
                                new_labs_te, "experiments/mhddata_pickle/", suffix="bow_"+str(session_size))


def preprocess_bow_uttr_data(session_size):
    pass


if __name__ == '__main__':
    # preprocess_text_uttr_data()
    # preprocess_text_sess_data(session_size=50)
    preprocess_bow_sess_data(session_size=100)
    # #
    # #
    # # # preprocess.create_subsets(data_folder, 'experiments/data/subsets', train_input_file, train_target_file, val_input_file,
    # # #                val_target_file, test_input_file, test_target_file, train_size=500, val_size=100, test_size=100)
#                    val_target_file, test_input_file, test_target_file, vocab_file, max_freq=0.0, min_freq=0.0,sessions=True)

import preprocess
import argparse
import os

import utils

if __name__ == '__main__':

    # new_words_tr, new_labs_tr, new_words_dev, new_labs_dev, new_words_te, new_labs_te\
    #     =preprocess.split_text_sessions("experiments/mhddata_pickle/", session_size=20)
    # preprocess.convert_mhddata_to_text(new_words_tr, new_labs_tr, new_words_dev, new_labs_dev, new_words_te, new_labs_te,
    #                                    "experiments/data","sess_new")

    preprocess.flatten_sessions("experiments/mhddata_pickle/","experiments/data",suffix="uttr")

    data_folder = 'experiments/data'
    train_input_file='train_input_sess_20.txt'
    train_target_file='train_target_sess_20.txt'
    val_input_file='val_input_sess_20.txt'
    val_target_file='val_target_sess_20.txt'
    test_input_file='test_input_sess_20.txt'
    test_target_file='test_target_sess_20.txt'
    vocab_file='vocab_sess.txt'

    train_input_file = 'train_input_uttr.txt'
    train_target_file = 'train_target_uttr.txt'
    val_input_file = 'val_input_uttr.txt'
    val_target_file = 'val_target_uttr.txt'
    test_input_file = 'test_input_uttr.txt'
    test_target_file = 'test_target_uttr.txt'
    vocab_file = 'vocab_uttr.txt'


    # preprocess.create_subsets(data_folder, 'experiments/data/subsets', train_input_file, train_target_file, val_input_file,
    #                val_target_file, test_input_file, test_target_file, train_size=500, val_size=100, test_size=100)
    preprocess.preprocess_text_data(data_folder, train_input_file, train_target_file, val_input_file,
                         val_target_file, test_input_file, test_target_file, vocab_file, max_freq=0.0, min_freq=0.0,sessions=False)

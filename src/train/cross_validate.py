import src.preprocess
from src.preprocess import preprocess_main
import gc
import time
from sklearn.model_selection import ParameterGrid
from multiprocessing.pool import ThreadPool
import src.train.cross_validate_helper
import traceback
import numpy as np
import sys

def preprocess_for_cross_validation(data_folder, out_folder, session_size, n_splits):
    # preprocess mhddata for hierarchical utterance classification models. If session_size is not None, original sessions
    # will be splitted into subsessions of session_size.
    # Data will be saved into text files with one session per line. Utterances are separated by '#' symbol.
    preprocess_main.preprocess_text_sess_data(session_size=400, data_folder=data_folder, out_folder=out_folder,
                                              tr_input_fname="splits_words_tr.pickle",
                                              tr_labels_fname="splits_labs_tr.pickle",
                                              val_input_fname="splits_words_dev.pickle",
                                              val_labels_fname="splits_labs_dev.pickle",
                                              te_input_fname="splits_words_te.pickle",
                                              te_labels_fname="splits_labs_te.pickle",
                                              size_limit=800)
    # build and save vocabulary in the out_folder based on train set.
    src.preprocess.constract_vocabulary(out_folder=out_folder, input_path="experiments/data/train_input_uttr.txt",
                                        vocab_name="vocab", max_freq=0.0,
                                        min_freq=0.1, sessions=False)

def cross_validation(params, n_processes):
    gc.disable()
    params.model_architecture = "h-rnn-rnn"
    params.predictions_filename = 'predictions.txt'
    params.eval_batch_size = 2
    params.predict_batch_size = 2
    params.save_trans_params = True
    params.ckpt = None
    # Other
    params.gpu = None
    params.random_seed = None
    params.log_device_placement = False
    params.timeline = False
    # optimizer
    params.learning_rate = 0.01
    params.optimizer = 'adam'
    params.colocate_gradients_with_ops = True
    params.start_decay_step = 0
    params.decay_steps = 10000
    params.decay_factor = 0.98
    params.max_gradient_norm = 5.0
    # training
    params.batch_size = 2
    params.num_epochs = 10
    params.num_ckpt_epochs = 1
    # network
    params.init_op = 'uniform'
    params.init_weight = 0.1
    params.uttr_time_major = False
    params.sess_time_major = False
    params.input_emb_trainable = True
    params.out_bias = True
    params.forget_bias = 1.0
    params.connect_inp_to_out = False
    params.uttr_activation = "relu"
    params.sess_activation = "relu"
    # cnn
    params.filter_sizes = '3,4'
    params.num_filters = 10
    params.pool_size = 1
    params.padding = 'valid'
    params.stride = 1
    #network
    params.uttr_layers = 1
    params.sess_layers = 1
    params.uttr_rnn_type = 'uni'
    params.sess_rnn_type = 'uni'
    params.uttr_unit_type = 'gru'
    params.sess_unit_type = 'gru'
    params.uttr_pooling = 'last'
    params.uttr_attention_size = 32
    params.input_emb_size = 300
    params.out_dir = 'experiments/out_model/splits'
    params.n_classes = 27
    params.hparams_path = None
    # What symbols to use for unk and pad.
    params.unk = '<unk>'
    params.pad = '<pad>'
    params.feature_size = 12624
    params.data_folder = 'experiments/data/splits'
    params.n_jobs = 6

    nn_params = {
        "uttr_units": [20, 50],
        "sess_units": [None],
        "uttr_hid_to_out_dropout": [2],
        "sess_hid_to_out_dropout": [None, 10, 20]
    }
    param_combs = list(ParameterGrid(nn_params))
    print("\n")
    print("Run Cross validation for model %s and %d param combinations." % (
    params.model_architecture, len(param_combs)))
    print("\n")

    loss_cv, acc_cv, f1_cv, pr_cv, rc_cv = [], [], [], [], []

    def cross_validate_comb(params, tr_val_labels, comb, i):
        print("Run cross validation for params: %s" % comb)
        params.uttr_units = comb["uttr_units"]
        params.sess_units = comb["sess_units"]
        params.uttr_hid_to_out_dropout = comb["uttr_hid_to_out_dropout"]
        params.sess_hid_to_out_dropout = comb["sess_hid_to_out_dropout"]
        avg_loss, avg_acc, avg_f1, avg_pr, avg_rc = src.train.train.cross_validate_helper.run_cross_validate(params, tr_val_labels)
        results = {}
        results["avg_loss"] = avg_loss
        results["avg_acc"] = avg_acc
        results["avg_f1"] = avg_f1
        results["avg_pr"] = avg_pr
        results["avg_rc"] = avg_rc
        return results, i

    def save_async_result_to_list(result, i, result_list):
        result_list[i] = result

    def callback_error(result):
        print('error', result)

    pool = ThreadPool(processes=n_processes)
    results = [{}] * len(param_combs)
    start_time_cv = time.time()
    for i, comb in enumerate(param_combs):
        try:
            pool.apply_async(cross_validate_comb, args=(params, tr_val_labels, comb, i)
                             , callback=lambda result: save_async_result_to_list(result[0], result[1], results),
                             error_callback=callback_error)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            traceback.print_tb(exc_tb)
    pool.close()
    pool.join()
    print("Cross validation finished in %f secs" % (time.time() - start_time_cv))
    for i, res in enumerate(results):
        print("Loss for comb %d: %.3f" % (i, res["avg_loss"]))
        print("Accuracy score for comb %d: %.3f" % (i, res["avg_acc"]))
        print("F1 score for comb %d: %.3f" % (i, res["avg_f1"]))
        print("Precision for comb %d: %.3f" % (i, res["avg_pr"]))
        print("Recall for comb %d: %.3f" % (i, res["avg_rc"]))
        loss_cv.append(res["avg_loss"])
        acc_cv.append(res["avg_acc"])
        f1_cv.append(res["avg_f1"])
        pr_cv.append(res["avg_pr"])
        rc_cv.append(res["avg_rc"])
    loss_min_idx = np.argmax(loss_cv)
    acc_max_idx = np.argmax(acc_cv)
    f1_max_idx = np.argmax(f1_cv)
    pr_max_idx = np.argmax(pr_cv)
    rc_max_idx = np.argmax(rc_cv)

    print("Min Loss score: %.3f for params %s" % (loss_cv[loss_min_idx], param_combs[loss_min_idx]))
    print("Max Accuracy score: %.3f for params %s" % (acc_cv[acc_max_idx], param_combs[acc_max_idx]))
    print("Max F1 score: %.3f for params %s" % (f1_cv[f1_max_idx], param_combs[f1_max_idx]))
    print("Max Precision: %.3f for params %s" % (pr_cv[pr_max_idx], param_combs[pr_max_idx]))
    print("Max Recall: %.3f for params %s" % (rc_cv[rc_max_idx], param_combs[rc_max_idx]))
    gc.enable()

if __name__=="__main__":
    import argparse
    params = argparse.Namespace()
    params.data_folder = "../data/data_parsed/wiki_727_final"
    params.splits_folder = "../data/wiki_727_out/splits_data/split_idxs"
    params.out_folder = "../data/wiki_727_out/splits_data"
    params.n_splits = 3
    documents, labels, _ = datasets.load_data_and_labels(params.data_folder)
    labels = preprocessing.transform_labels(labels, {'B': 1, 'NB': 0})
    # extract a fixed test set before cross validation.
    tr_val_documents, tr_val_labels, te_documents, te_labels = preprocessing.split_train_test_data_labels(documents,
                                                                                                          labels,
                                                                                                          test_size=0.2,
                                                                                                          random_seed=1234,
                                                                                                          out_folder=params.splits_folder)
    b_sizes = [2, 4, 6]
    ranks_win_sizes = [2, 4, 6]
    # create and save features for different splits.
    create_features_for_cross_validation(tr_val_documents, tr_val_labels, params, b_sizes, ranks_win_sizes)
    # run cross validation in parallel for different combinations of Random forest parameters.
    cross_validation(tr_val_labels, params, b_sizes, ranks_win_sizes, n_processes=4)
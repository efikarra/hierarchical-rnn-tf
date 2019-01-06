""" After training models and get predictions on test data, use this module to run evaluation methods on the predictions."""
import numpy as np
import cPickle
import src.utils
from src.utils import sess_probs_to_labs
from src.evaluation import evaluate_methods


def print_format_metrics(results):
    bin_metrics = ["precision", "recall", "auc", "rprecision", "f1score"]
    orders = ["accuracy"]  + [met + "_w" for met in bin_metrics] + bin_metrics
    line=""
    line_separator=""
    for met in orders:
        line+=line_separator+str(results[met])
        line_separator="|"
    return line


def regroup_sessions(pred_sessions, tgt_sessions, orig_tgt_sessions):
    # trancate predictions of pred_sessions to remove those predictions that correspond to padding.
    for i, sess in enumerate(tgt_sessions):
        pred_sessions[i] = pred_sessions[i][0:len(sess)]
    # flatten pred_sessions in order to regroup them based on orig_tgt_sessions lengths.
    sessions_f = [item for sublist in pred_sessions for item in sublist]
    regrouped_sessions = []
    idx=0
    for i, sess in enumerate(orig_tgt_sessions):
        regrouped_sessions.append(np.array(sessions_f[idx:idx+len(sess)]))
        idx+=len(sess)
    for i, sess in enumerate(orig_tgt_sessions):
        assert len(sess)==regrouped_sessions[i].shape[0]
    return regrouped_sessions


def session_flat_predictions(probs, labs):
    format_preds = []
    idx=0
    for i, sess in enumerate(labs):
        format_preds.append(np.array(probs[idx:idx + len(sess)]))
        idx+=len(sess)
    for i, sess in enumerate(labs):
        assert len(sess)==format_preds[i].shape[0]
    return format_preds


def get_result_in_diff_metrics(pred_sessions, tgt_sessions, tr_doc_label_mat):
    # flatten predictions and target sessions in order to compute all metrics.
    pred_y_f = [item for sublist in pred_sessions for item in sublist]
    true_y_f = [item for sublist in tgt_sessions for item in sublist]
    results = evaluate_methods.get_overall_scores_in_diff_metrics(true_y_f, pred_y_f, tr_doc_label_mat)
    return results


def print_per_utter_result(models_names, labs_predictions, labs_test, sids_te, docs_te, lid2name, outpath, print_lines=False, print_uttrs=False):
    assert(len(models_names)==len(labs_predictions))
    models_line = "Truelab|"+"|".join(models_names)+"|Utterance"+"\n"
    accs_line = "100|"+"|".join(["%.4f"]*len(models_names))+"|\n"
    overall_accs = tuple([evaluate_methods.get_accuracy_from_sessions(labs_test, labs_predictions[i]) for i in range(len(models_names))])
    with open(outpath, 'w') as f:
        if print_lines:
            print(models_line)
            print(accs_line % overall_accs)
        f.write(models_line)
        f.write(accs_line % overall_accs)

        for sidx in range(len(labs_test)):
            truel = labs_test[sidx]
            sess_accs = tuple([evaluate_methods.get_accuracy(true_y=truel, y_hat=labs_predictions[j][sidx]) for j in range(len(models_names))])

            sess_line = "\nSessionIndex: %d  SessionID: %d \n" % (sidx, sids_te[sidx])
            if print_lines:
                print(sess_line)
                print models_line
                print accs_line % sess_accs

            f.write(sess_line)
            f.write(models_line)
            f.write(accs_line % sess_accs)

            for i in range(len(labs_test[sidx])):
                uttr_labels = tuple([lid2name[truel[i]]])+tuple([lid2name[labs_predictions[j][sidx][i]] for j in range(len(models_names))])
                uttr_line = "%s|%s|"+ "|".join(["%s"]*len(models_names))+"\n"
                uttr_line = uttr_line % (uttr_labels + tuple([docs_te[sidx][i]]))
                if print_lines and print_uttrs:
                    print uttr_line
                f.write(uttr_line)


def format_rnn_predictions(predictions, labs, out_filename):
    """Format hierarchical models predictions into the required common format: list of sessions where each session
            is an array of shape = (sess_length, n_classes) in case of probabilities or shape = (sess_length, 1) in case of labels."""
    if type(predictions) is dict:
        sess_predictions = session_flat_predictions(predictions["probabilities"], labs)
        sess_lab_preds = predictions["labels"]
    else:
        sess_predictions = session_flat_predictions(predictions, labs)
        sess_lab_preds = sess_probs_to_labs(sess_predictions)
    if out_filename is not None:
        cPickle.dump(sess_predictions, open("experiments/formatted_preds/" + out_filename + "_prob.pkl", "wb"))
        cPickle.dump(sess_lab_preds, open("experiments/formatted_preds/" + out_filename + "_pred.pkl", "wb"))
    return sess_predictions, sess_lab_preds


def format_hierarchical_predictions(predictions, tgt_session_labs, orig_labs, out_filename):
    """Format hierarchical models predictions into the required common format: list of sessions where each session
        is an array of shape = (sess_length, n_classes) in case of probabilities or shape = (sess_length, 1) in case of labels."""
    format_probs = None
    if predictions["probabilities"] is not None and len(predictions["probabilities"])>0:
        format_probs = regroup_sessions(predictions["probabilities"], tgt_session_labs, orig_labs)
        cPickle.dump(format_probs, open("experiments/formatted_preds/"+out_filename+"_prob.pkl", "wb"))
    sess_labels = regroup_sessions(predictions["labels"], tgt_session_labs, orig_labs)
    cPickle.dump(sess_labels, open( "experiments/formatted_preds/"+out_filename+"_pred.pkl", "wb"))
    return format_probs, sess_labels


def results_pipeline():
    """ Compute evaluation results for RNN, H_RNN_RNN, H_RNN_RNN_CRF, H_RNN_RNN_ConnInp, H_RNN_RNN_CRF_ConnInp."""
    # load mappings from ids to topic names etc.
    # lab2ltr: map from topic code to broader topic letter.
    lab2ltr = cPickle.load(open("experiments/data/mhddata/mhddata_lab2ltr.pickle", "rb"))
    # lid2lab: map from topic code to topic id.
    lid2lab = cPickle.load(open("experiments/data/mhddata/mhddata_lid2lab.pickle", "rb"))
    # lt2ltid: map from broader topic letter to broader topic id.
    lt2ltid = cPickle.load(open("experiments/data/mhddata/mhddata_lt2ltid.pickle", "rb"))
    # ltid2name: map from broader topic id to broader topic short name.
    ltid2name = cPickle.load(open("experiments/data/mhddata/mhddata_ltid2name.pickle", "rb"))
    # lid2name: map from topic id to topic short name.
    lid2name = cPickle.load(open("experiments/data/mhddata/splits_lid2name.pickle", "rb"))

    # labs_te: target test labels as a list of sessions where each session is a list of utterances.
    labs_te = cPickle.load(open("experiments/data/mhddata/splits_labs_te.pickle", "rb"))
    # sids_te: test sessions ids.
    sids_te = cPickle.load(open("experiments/data/mhddata/splits_sids_te.pickle", "rb"))
    # docs_te: test sessions origina; utterances text.
    docs_te = cPickle.load(open("experiments/data/mhddata/splits_docs_te.pickle", "rb"))
    # labs_sess: target test labels in the format used by the trained models.
    # So for example in experiments/data/test_target_sess_400.txt labels file, original sessions are splitted into
    # subsessions of length 400.
    labs_sess = src.utils.load_labs_from_text_sess("experiments/data/test_target_sess_400.txt")

    # each predictions.pickle contains a dict with entries "probabilities" and "labels". These are the outputs of each model for the test set.
    # First we transform (and save) predictions as produced by the models into a common format in order to run the evaluation methods.
    # The required common format is: list of sessions where each session
    # is an array of shape = (sess_length, n_classes) in case of probabilities or shape = (sess_length, 1) in case of labels.
    rnn_preds = cPickle.load(open("experiments/eval_output_old/rnn/predictions.pickle", "rb"))
    out_filename = "RNN"
    _, rnn_lab_preds = format_rnn_predictions (rnn_preds, labs_te, out_filename)

    hrnn_preds = cPickle.load(open("experiments/eval_output_old/h_rnn_rnn/predictions.pickle", "rb"))
    out_file = "H_RNN_RNN"
    _, hrnn_lab_preds = format_hierarchical_predictions(hrnn_preds, labs_sess, labs_te, out_file)

    hrnn_crf_preds = cPickle.load(open("experiments/eval_output_old/h_rnn_rnn_crf/predictions.pickle", "rb"))
    out_filename = "H_RNN_RNN_CRF"
    _, hrnn_crf_lab_preds = format_hierarchical_predictions(hrnn_crf_preds, labs_sess, labs_te, out_filename)

    hrnn_conn_preds = cPickle.load(open("experiments/eval_output_old/results_400_conn/h_rnn_rnn/predictions.pickle", "rb"))
    out_file = "H_RNN_RNN_ConnInp"
    _, hrnn_conn_lab_preds = format_hierarchical_predictions(hrnn_conn_preds, labs_sess, labs_te, out_file)

    hrnn_crf_conn_preds = cPickle.load(open("experiments/eval_output_old/results_400_conn/h_rnn_rnn_crf/predictions.pickle", "rb"))
    out_filename = "H_RNN_RNN_CRF_ConnInp"
    _, hrnn_crf_conn_lab_preds = format_hierarchical_predictions(hrnn_crf_conn_preds, labs_sess, labs_te, out_filename)


    # compute all metrics for the formatted model predictions
    print("Metrics for original predictions:")
    h_models_names = ["H_RNN_RNN", "H_RNN_RNN_CRF", "H_RNN_RNN_ConnInp", "H_RNN_RNN_CRF_ConnInp"]
    flat_model_names=["RNN"]
    h_labs_pred_list = [hrnn_lab_preds, hrnn_crf_lab_preds, hrnn_conn_lab_preds, hrnn_crf_conn_lab_preds]
    flat_labs_pred_list = [rnn_lab_preds]
    splits_labarr_tr_f = cPickle.load(open("experiments/data/mhddata/splits_labarr_tr_f.pickle"))
    metrics_pipeline(labs_te, sids_te, docs_te, lid2name, h_labs_pred_list, flat_labs_pred_list, h_models_names,
                     flat_model_names, splits_labarr_tr_f, outpath="experiments/results/results.txt")

    # compute all metrics after eliminating the 4-count rule in the formatted models predictions.
    print("\n")
    print("Metrics for predictions without the 4-count rule:")
    mental_health_topics = [10, 11]
    hrnn_lab_preds_4r = src.utils.fix_uttr_count_rule(hrnn_lab_preds, set(mental_health_topics))
    hrnn_crf_lab_preds_4r = src.utils.fix_uttr_count_rule(hrnn_crf_lab_preds, set(mental_health_topics))
    hrnn_conn_lab_preds_4r = src.utils.fix_uttr_count_rule(hrnn_conn_lab_preds, set(mental_health_topics))
    hrnn_crf_conn_lab_preds_4r = src.utils.fix_uttr_count_rule(hrnn_crf_conn_lab_preds, set(mental_health_topics))
    h_models_names_4r = ["H_RNN_RNN_4r", "H_RNN_RNN_CRF_4r", "H_RNN_RNN_ConnInp_4r", "H_RNN_RNN_CRF_ConnInp_4r"]
    h_labs_pred_list_4r = [hrnn_lab_preds_4r, hrnn_crf_lab_preds_4r, hrnn_conn_lab_preds_4r, hrnn_crf_conn_lab_preds_4r]
    metrics_pipeline(labs_te, sids_te, docs_te, lid2name, h_labs_pred_list_4r, flat_labs_pred_list, h_models_names_4r,
                     flat_model_names, splits_labarr_tr_f, outpath="experiments/results/results_4rule.txt")

    # compute all metrics after mapping the formatted model predictions into broader topics.
    print("\n")
    print("Metrics for predictions mapped into broader topics:")
    labs_te_lt = src.utils.convert_to_broader_topic(labs_te, lab2ltr, lid2lab, lt2ltid)
    rnn_lab_preds_lt = src.utils.convert_to_broader_topic(rnn_lab_preds, lab2ltr, lid2lab, lt2ltid)
    hrnn_lab_preds_lt = src.utils.convert_to_broader_topic(hrnn_lab_preds, lab2ltr, lid2lab, lt2ltid)
    hrnn_crf_lab_preds_lt = src.utils.convert_to_broader_topic(hrnn_crf_lab_preds, lab2ltr, lid2lab, lt2ltid)
    hrnn_conn_lab_preds_lt = src.utils.convert_to_broader_topic(hrnn_conn_lab_preds, lab2ltr, lid2lab, lt2ltid)
    hrnn_crf_conn_lab_preds_lt = src.utils.convert_to_broader_topic(hrnn_crf_conn_lab_preds, lab2ltr, lid2lab, lt2ltid)

    h_models_names = ["H_RNN_RNN_lt", "H_RNN_RNN_CRF_lt", "H_RNN_RNN_ConnInp_lt", "H_RNN_RNN_CRF_ConnInp_lt"]
    flat_model_names = ["RNN_lt"]
    h_labs_pred_list = [hrnn_lab_preds_lt, hrnn_crf_lab_preds_lt, hrnn_conn_lab_preds_lt, hrnn_crf_conn_lab_preds_lt]
    flat_labs_pred_list = [rnn_lab_preds_lt]
    labs_tr = cPickle.load(open("experiments/data/mhddata/splits_labs_tr.pickle", "rb"))
    labs_tr_lt = src.utils.convert_to_broader_topic(labs_tr, lab2ltr, lid2lab, lt2ltid)
    labarr_tr_lt_f = src.utils.flatten_nested_labels(labs_tr_lt)
    splits_labarr_tr_lt_f = src.utils.get_lab_arr(labarr_tr_lt_f)
    metrics_pipeline(labs_te_lt, sids_te, docs_te, ltid2name, h_labs_pred_list, flat_labs_pred_list, h_models_names,
                     flat_model_names, splits_labarr_tr_lt_f, outpath="experiments/results/results_lt.txt")


def metrics_pipeline(labs_te, sids_te, docs_te, lid2name, h_labs_pred_list, flat_labs_pred_list, h_models_names,
                     flat_model_names, splits_labarr_tr_f, first_uttr_acc=True, outpath ="experiments/results/results.txt"):
    if first_uttr_acc:
        for i,labs in enumerate(h_labs_pred_list):
            print "%s accuracy on 1st uttr: %.4f" % (h_models_names[i], evaluate_methods.accuracy_on_first_sgmt_uttr(labs_te, labs))

    labs_pred_list = h_labs_pred_list + flat_labs_pred_list
    model_names = h_models_names+flat_model_names

    for i, labs_pred in enumerate(labs_pred_list):
        results = get_result_in_diff_metrics(labs_pred, labs_te, splits_labarr_tr_f)
        print model_names[i]+": " + print_format_metrics(results)
    # print labels and utterances for various models.
    print_per_utter_result(model_names, labs_pred_list, labs_te, sids_te, docs_te, lid2name,
                           outpath, print_lines=False, print_uttrs=False)


if __name__ == '__main__':
    results_pipeline()





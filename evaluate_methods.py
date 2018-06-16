import numpy as np
from utils import get_lab_arr, save_sq_mat_with_labels
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix

# most methods of this module were copied from Jihyun's github: https://github.com/jihyunp/PP_dialog_models
def R_precision(true_y, pred_y):
    R = np.sum(true_y)
    trueidxs = np.where(true_y)[0]
    retrieved_R_docs = np.argsort(pred_y)[::-1][:int(R)]

    n_true = len(set(retrieved_R_docs).intersection(set(trueidxs)))
    return n_true / float(R)


def get_accuracy(true_y, y_hat):
    assert len(true_y) == len(y_hat)
    numcorr = lambda a, b: np.where(np.array(a) == np.array(b))[0].shape[0]
    return numcorr(true_y, y_hat) / float(len(true_y)) * 100.0


def get_accuracy_per_lab(true_y, y_hat, n_labels):
    true_y_arr = get_lab_arr(true_y, n_labels)
    yhat_arr = get_lab_arr(y_hat, n_labels)

    accs = []
    n_utter = np.sum(true_y_arr, axis=0)
    for tidx in range(n_labels):
        accs.append(get_accuracy(true_y_arr[:, tidx], yhat_arr[:, tidx]))
    return accs, n_utter


def get_binary_classification_scores(true_y, pred_y, n_labels):
    true_y_arr = get_lab_arr(true_y, n_labels)
    yhat_arr = get_lab_arr(pred_y, n_labels)

    precisions = []
    recalls = []
    aucs = []
    rprecisions = []
    fscores = []

    for tidx in range(n_labels):
        precisions.append(precision_score(true_y_arr[:, tidx], yhat_arr[:, tidx]))
        recalls.append(recall_score(true_y_arr[:, tidx], yhat_arr[:, tidx]))
        fscores.append(f1_score(true_y_arr[:, tidx], yhat_arr[:, tidx]))
        aucs.append(roc_auc_score(true_y_arr[:, tidx], yhat_arr[:, tidx]))
        rprecisions.append(R_precision(true_y_arr[:, tidx], yhat_arr[:, tidx]))

    return {"precision":precisions, "recall":recalls, "auc":aucs, "rprecision":rprecisions, "f1score":fscores}


def get_overall_scores_in_diff_metrics(true_y, pred_y, tr_doc_label_mat):
    n_states = tr_doc_label_mat.shape[1]
    marginals = get_marginals(tr_doc_label_mat)
    results = {}

    acc = get_accuracy(true_y, pred_y)
    results["accuracy"] = acc

    scores = get_binary_classification_scores(true_y, pred_y, n_states)
    for sc in sorted(scores.keys()):
        weighted = get_weighted_avg(scores[sc], marginals)
        notweighted = np.mean(scores[sc])
        results[sc+"_w"] = weighted
        results[sc] = notweighted

    return results


def print_row_of_diff_metrics(model_name, result_numbers, headers=None, filename="./overall_result.csv"):

    if headers is None:
        bin_metrics = ["precision", "recall", "auc", "rprecision", "f1score"]
        headers = ["model", "accuracy"] + [met+"_w" for met in bin_metrics]  + bin_metrics
        print(",".join(headers))

    with open(filename, 'a') as f:
        f.write(model_name)
        print(model_name),
        for met in headers[1:]:
            f.write(",%.4f" % result_numbers[met])
            print(",%.4f" % result_numbers[met]),
        f.write("\n")


def get_weighted_avg(score_list, weights):
    return np.dot(score_list, weights)


def get_weighted_avg_from_ymat(score_list, doc_label_mat):
    weights = get_marginals(doc_label_mat)
    weighted_avg = np.dot(score_list, weights)
    return weighted_avg


def get_marginals(doc_label_mat):
    n_docs_in_labels = np.sum(doc_label_mat, axis=0)
    totalsum = np.sum(n_docs_in_labels)
    weights = n_docs_in_labels / float(totalsum)
    return weights


def save_confusion_matrix(true_f, pred_f, lid2shortname, filename):
    conf = confusion_matrix(true_f, pred_f)
    save_sq_mat_with_labels(conf, lid2shortname, filename)
    return conf


def verify_uttr_count_rule(labs, excl_labs):
    """ Verify whether labs keep the 4-count rule,
        i.e., within a segment there are not smaller segments (with <=3 utterances) belonging to a different topic.
        labs: labels as a list of sessions where each session is a list of utterances."""
    for i,sess_labs in enumerate(labs):
        j=0
        while j < len(sess_labs)-1:
            topic1 = sess_labs[j]
            while j<len(sess_labs)-1 and sess_labs[j]==sess_labs[j+1]:
                j+=1
            if j < len(sess_labs)-1:
                topic2 = sess_labs[j+1]
                topic2_c = 1
                j += 1
            while j<len(sess_labs)-1 and sess_labs[j]==sess_labs[j+1]:
                topic2_c+=1
                j+=1
            if (j<len(sess_labs)-1) and (sess_labs[j+1]==topic1) and (topic2_c<=3) and (topic2 not in excl_labs):
                print "Session: %d, Uttr: %d Topic 1: %s, Topic 2: %s, Topic2_count %d"%(i, j, topic1, topic2, topic2_c)


def get_accuracy_from_sessions(true_y, y_hat):
    """true_y: true labels as a list of sessions.
        y_hat: predicted labels as a list of sessions"""
    assert len(true_y) == len(y_hat)
    # flatten sessions list
    lab_predictions_f = [item for sublist in y_hat for item in sublist]
    targets_f = [item for sublist in true_y for item in sublist]
    accuracy = get_accuracy(true_y=targets_f , y_hat=lab_predictions_f)
    return accuracy



def accuracy_on_first_sgmt_uttr(y_true, y_hat):
    """ Computes accuracy only on the first utterance of each segment. """
    sum = 0.0
    count = 0.0
    for i, sess in enumerate(y_true):
        for j, lab in enumerate(sess):
            if j == 0 or sess[j - 1] != sess[j]:
                count+=1
                if y_hat[i][j - 1]!=y_hat[i][j] and y_hat[i][j]==sess[j]: sum+=1
    accuracy = sum/count
    return accuracy
import pickle
import os
import numpy as np
import src.train
from src.main import process_or_load_hparams
import src.evaluation

def run_cross_validate(params, tr_val_labels):
    pks, wds, cnfms, f1_scores, accuracies = [], [], [], [], []
    for fold in range(params.n_splits):
        print("Run model %s for fold %d" % (params.model_architecture, fold))
        params.train_input_path = os.path.join(os.path.join(params.data_folder, 'splits_%d' % fold),params.train_input_path)
        params.train_target_path = os.path.join(os.path.join(params.data_folder, 'splits_%d' % fold),params.train_target_path)
        # Validation data files.
        params.val_input_path = os.path.join(os.path.join(params.data_folder, 'splits_%d' % fold),params.val_input_path)
        params.val_target_path = os.path.join(os.path.join(params.data_folder, 'splits_%d' % fold),params.val_target_path)

        params.out_dir = os.path.join(params.out_dir,  "out_" + str(fold))
        if not os.path.exists(params.out_dir): os.makedirs(params.out_dir)

        # restrict tensoflow to run only in the specified gpu. This has no effect if run on a machine with no gpus.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params.gpu)
        hparams = process_or_load_hparams(params.out_dir, params, params.hparams_path)
        losses = src.train.train.train(params)

        if not os.path.exists(params.eval_output_folder): os.makedirs(hparams.eval_output_folder)
        ckpt = hparams.ckpt
        pk, wd, cnfm, f1_sc, acc = src.evaluation.evaluation.evaluate(params)
        pks.append(pk)
        wds.append(wd)
        cnfms.append(cnfm)
        f1_scores.append(f1_sc)
        accuracies.append(acc)
    avg_f1 = np.mean(f1_scores)
    avg_wd = np.mean(wds)
    avg_pk = np.mean(pks)
    print("Avg. F1 score %f" % avg_f1)
    print("Avg. Wd %f" % avg_wd)
    print("Avg. Pk %f" % avg_pk)
    return avg_f1, avg_wd, avg_pk
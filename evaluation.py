import tensorflow as tf
import model_helper
import os
import cPickle
import numpy as np
import utils


def eval(model, sess, iterator, iterator_feed_dict):
    # initialize the iterator with the data on which we will evaluate the model.
    sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
    loss, accuracy = model_helper.run_batch_evaluation(model, sess)
    return loss, accuracy


def eval_and_precit(model, sess, iterator, iterator_feed_dict):
    # initialize the iterator with the data on which we will evaluate the model.
    sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
    loss, accuracy, predictions = model_helper.run_batch_evaluation_and_prediction(model, sess)
    return loss, accuracy, predictions


def evaluate(hparams, ckpt):
    # get Model class.
    model_creator = model_helper.get_model_creator(hparams.model_architecture)
    print("Starting evaluation and predictions:")
    # create eval graph.
    eval_model = model_helper.create_eval_model(model_creator, hparams, tf.contrib.learn.ModeKeys.EVAL, shuffle=False)
    eval_sess = tf.Session(config=utils.get_config_proto(), graph=eval_model.graph)
    with eval_model.graph.as_default():
        # load pretrained model.
        loaded_eval_model = model_helper.load_model(eval_model.model, eval_sess, "evaluation", ckpt)
    if hparams.val_target_path:
        iterator_feed_dict = {
            eval_model.input_file_placeholder: hparams.eval_input_path,
            eval_model.output_file_placeholder: hparams.eval_target_path
        }
    else:
        iterator_feed_dict = {
            eval_model.input_file_placeholder: hparams.eval_input_path,
        }
    eval_loss, eval_accuracy, predictions = eval_and_precit(loaded_eval_model, eval_sess, eval_model.iterator,
                                                            iterator_feed_dict)
    print("Eval loss: %.3f, Eval accuracy: %.3f" % (eval_loss, eval_accuracy))
    # only models with CRF include trans. params.
    if hparams.save_trans_params:
        transition_params = eval_sess.run(loaded_eval_model.transition_params)
        if transition_params is not None:
            print("Saving transition parameters:")
            np.savetxt(os.path.join(hparams.eval_output_folder, "transition_params.txt"), transition_params)

    print("Saving predictions:")
    cPickle.dump(predictions,
                 open(os.path.join(hparams.eval_output_folder, hparams.predictions_filename.split(".")[0] + ".pickle"),
                      "wb"))

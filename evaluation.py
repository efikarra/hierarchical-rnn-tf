import tensorflow as tf
import model_helper
import os
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


def predict(model, sess, iterator, iterator_feed_dict):
    sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
    predictions = model_helper.run_batch_prediction(model, sess)
    return predictions


def evaluate(hparams, ckpt):
    model_creator = model_helper.get_model_creator(hparams.model_architecture)

    #dirty! change this! pisk a common data format for both models.
    if hparams.val_target_path or hparams.model_architecture=="h-ffn-rnn":
        print("Starting evaluation and predictions:")
        eval_model = model_helper.create_eval_model(model_creator, hparams, tf.contrib.learn.ModeKeys.EVAL)
        eval_sess = tf.Session(config=utils.get_config_proto(), graph=eval_model.graph)
        with eval_model.graph.as_default():
            loaded_eval_model = model_helper.load_model(eval_model.model, eval_sess, "evaluation", ckpt)
        iterator_feed_dict={
            eval_model.input_file_placeholder: hparams.eval_input_path,
            eval_model.output_file_placeholder: hparams.eval_target_path
        }
        eval_loss, eval_accuracy, predictions = eval_and_precit(loaded_eval_model, eval_sess, eval_model.iterator, iterator_feed_dict)
        print("Eval loss: %.3f, Eval accuracy: %.3f"%(eval_loss,eval_accuracy))
    else:
        print("Starting predictions:")
        prediction_model = model_helper.create_infer_model(model_creator, hparams, tf.contrib.learn.ModeKeys.INFER)
        prediction_sess = tf.Session(config=utils.get_config_proto(), graph=prediction_model.graph)
        with prediction_model.graph.as_default():
            loaded_prediction_model = model_helper.load_model(prediction_model.model, prediction_sess, "prediction", ckpt)
            iterator_feed_dict = {
                prediction_model.input_file_placeholder: hparams.eval_input_path,
            }
        predictions=predict(loaded_prediction_model, prediction_sess, prediction_model.iterator, iterator_feed_dict)
    print("Saving predictions:")
    np.savetxt(os.path.join(hparams.eval_output_folder, hparams.predictions_filename), predictions)
    # save_labels(predictions["classes"], os.path.join(hparams.eval_output_folder, "classes"))
    # save_probabilities(predictions["probabilities"], os.path.join(hparams.eval_output_folder, "probabilities"))



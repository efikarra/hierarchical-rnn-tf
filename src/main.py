""" run this module from command line to run the different models."""
import argparse
import tensorflow as tf
import os
import numpy as np
import train
import utils
import evaluation
from src.utils import vocab_utils


def add_arguments(parser):
    """ Supported command line arguments"""
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # Data
    parser.add_argument("--train_input_path", type=str, default=None,
                        help="Train input file path.")
    parser.add_argument("--train_target_path", type=str, default=None,
                        help="Train target file path.")
    parser.add_argument("--val_input_path", type=str, default=None,
                        help="Validation input file path.")
    parser.add_argument("--val_target_path", type=str, default=None,
                        help="Validation target file path.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Directory to save log/checkpoint files.")
    parser.add_argument("--hparams_path", type=str, default=None,
                        help=("Path to standard hparams json file that overrides"
                              "hparams values from FLAGS."))
    parser.add_argument("--input_emb_file", type=str, default=None, help="Input embedding external file.")
    parser.add_argument("--n_classes", type=int, default=None, help="Number of output classes.")

    # Vocab
    parser.add_argument("--vocab_path", type=str, default=None, help="Vocabulary file path.")
    parser.add_argument("--unk", type=str, default="<unk>",
                        help="Unknown symbol")
    parser.add_argument("--pad", type=str, default="<pad>",
                        help="Padding symbol")

    # network
    parser.add_argument("--model_architecture", type=str, default="rnn",
                        help="h-rnn-rnn | h-rnn-ffn | h-rnn-cnn | h-rnn-rnn-crf | rnn | ffn | cnn. Model architecture.")
    parser.add_argument("--input_emb_size", type=int, default=32,
                        help="Input embedding size. If input_emb_file is given, the input_emb_size is inferred from "
                             "the loaded embeddings.")
    parser.add_argument("--input_emb_trainable", type=bool, default=True, help="Whether to train embedding layer.")
    parser.add_argument("--forget_bias", type=float, default=1.0,
                        help="Forget bias for BasicLSTMCell.")
    # cnn
    parser.add_argument("--filter_sizes", type=str, default='3',
                        help="Only for CNN models: List of filter sizes separated by comma.")
    parser.add_argument("--num_filters", type=int, default=100,
                        help="Only for CNN models: Number of filters for each filter size.")
    parser.add_argument("--pool_size", type=int, default=None,
                        help="Only for CNN models: Size of the max pooling layer. if None, no max pooling is applied.")
    parser.add_argument("--padding", type=str, default='3',
                        help="valid | same. Only for CNN models: Valid means that we slide the filters over an "
                             "utterance without padding the edges.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Only for CNN models: An integer specifying the stride "
                             "of the convolution along the height and width.")

    # initializer
    parser.add_argument("--init_op", type=str, default="uniform",
                        help="uniform | glorot_normal | glorot_uniform. Initializer for all model weights.")
    parser.add_argument("--init_weight", type=float, default=0.1,
                        help=("for uniform init_op: uniform distr: [-init_weight, init_weight]."))

    # hierarchical rnn
    parser.add_argument("--out_bias", type=bool, default=True, help="Whether to use bias in the output layer.")
    parser.add_argument("--uttr_units", type=str, default='32', help="Hidden units of utterance model. "
                                                                     "A list of units separated by comma should be given"
                                                                     " for multiple layers.")
    # !!! I NEVER REALLY TESTED MORE THAN 1 LAYER SO MOST LIKELY THERE ARE BUGS FOR >1 LAYER. !!!
    parser.add_argument("--uttr_layers", type=int, default=1, help="Number of utterance model layers.")
    parser.add_argument("--sess_units", type=str, default='32', help="Hidden units of session model. "
                                                                     "A list of units separated by comma should be given"
                                                                     " for multiple layers.")
    parser.add_argument("--sess_layers", type=int, default=1, help="Number of session model layers.")
    parser.add_argument("--uttr_hid_to_out_dropout", type=str, default='0.0',
                        help="List of hidden to output layer dropouts for utterance model layers.")
    parser.add_argument("--sess_hid_to_out_dropout", type=str, default='0.0',
                        help="List of hidden to output layer dropouts for session model layers.")
    parser.add_argument("--uttr_rnn_type", type=str, default="uni",
                        help="uni | bi .")
    parser.add_argument("--sess_rnn_type", type=str, default="uni",
                        help="uni | bi.")
    parser.add_argument('--uttr_unit_type', type=str, default="rnn",
                        help="rnn | lstm | gru.")
    parser.add_argument('--sess_unit_type', type=str, default="rnn",
                        help="rnn | lstm | gru.")
    parser.add_argument('--uttr_pooling', type=str, default="last",
                        help="last | mean | attn | attn_context . Pooling scheme to transform "
                             "utterance model hidden states into an utterance representation.")
    parser.add_argument("--uttr_attention_size", type=int, default=32,
                        help="Attention size if attention with context is used in the utterance level.")
    parser.add_argument('--connect_inp_to_out', type=bool, default=False,
                        help="Whether to directly connect the utterance representation to the output layer.")
    # ffn
    parser.add_argument("--feature_size", type=int, default=32, help="ONLY FOR FFN: Number of features.")
    parser.add_argument("--uttr_activation", type=str, default='relu',
                        help="List of activation functions for each layer of the utterance model.")
    parser.add_argument("--sess_activation", type=str, default='relu',
                        help="List of activation functions for each layer of the session model.")
    # training
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train the model.")
    parser.add_argument("--num_ckpt_epochs", type=int, default=2,
                        help="Number of epochs until the next checkpoint saving.")

    # optimizer
    parser.add_argument("--optimizer", type=str, default="sgd", help="sgd | adam")
    parser.add_argument("--learning_rate", type=float, default=0.1,
                        help="Learning rate.")
    parser.add_argument("--start_decay_step", type=int, default=0,
                        help="When we start to decay.")
    parser.add_argument("--decay_steps", type=int, default=10000,
                        help="How frequent we decay.")
    parser.add_argument("--decay_factor", type=float, default=0.98,
                        help="How much we decay.")
    parser.add_argument("--colocate_gradients_with_ops", type="bool", nargs="?",
                        const=True,
                        default=True,
                        help=("Whether try colocating gradients with "
                              "corresponding op"))
    parser.add_argument("--max_gradient_norm", type=float, default=5.0,
                        help="Clip gradients to this norm.")

    # Other
    parser.add_argument("--gpu", type=int, default=0,
                        help="Gpu machine to run the code (if gpus available)")
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed (>0, set a specific seed).")
    parser.add_argument("--log_device_placement", type="bool", nargs="?",
                        const=True, default=False, help="Debug GPU allocation.")
    parser.add_argument("--timeline", type="bool", nargs="?",
                        const=True, default=False, help="Whether to log timeline information for use in Tensorboard.")
    parser.add_argument("--save_trans_params", type=bool, default=True,
                        help="Only for CRF: Whether to save the transition parameters.")

    # Evaluation/Prediction
    parser.add_argument("--eval_output_folder", type=str, default=None,
                        help="Output folder to save evaluation data. If None, a new model is trained. If a value is given,"
                             " an existing model is loaded from out_dir and evaluated.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint file to evaluate an existing model.")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="Batch size for evaluation.")
    parser.add_argument("--eval_input_path", type=str, default=None,
                        help="Input file path to perform evaluation.")
    parser.add_argument("--eval_target_path", type=str, default=None,
                        help="Output file path to perform evaluation.")
    parser.add_argument("--predictions_filename", type=str, default="predictions.txt",
                        help="Filename to save predictions.")


def create_hparams(flags):
    return tf.contrib.training.HParams(
        # data
        input_emb_file=flags.input_emb_file,
        out_dir=flags.out_dir,
        train_input_path=flags.train_input_path,
        train_target_path=flags.train_target_path,
        val_input_path=flags.val_input_path,
        val_target_path=flags.val_target_path,
        hparams_path=flags.hparams_path,
        # Vocab
        vocab_path=flags.vocab_path,
        unk=flags.unk,
        pad=flags.pad,
        # network
        model_architecture=flags.model_architecture,
        n_classes=flags.n_classes,
        forget_bias=flags.forget_bias,
        input_emb_size=flags.input_emb_size,
        input_emb_trainable=flags.input_emb_trainable,
        connect_inp_to_out=flags.connect_inp_to_out,
        # cnn
        filter_sizes=flags.filter_sizes,
        num_filters=flags.num_filters,
        pool_size=flags.pool_size,
        padding=flags.padding,
        stride=flags.stride,
        # initializer
        init_weight=flags.init_weight,
        init_op=flags.init_op,
        # hierarchical rnn
        uttr_units=flags.uttr_units,
        uttr_layers=flags.uttr_layers,
        sess_units=flags.sess_units,
        sess_layers=flags.sess_layers,
        uttr_hid_to_out_dropout=flags.uttr_hid_to_out_dropout,
        sess_hid_to_out_dropout=flags.sess_hid_to_out_dropout,
        uttr_rnn_type=flags.uttr_rnn_type,
        sess_rnn_type=flags.sess_rnn_type,
        uttr_unit_type=flags.uttr_unit_type,
        sess_unit_type=flags.sess_unit_type,
        uttr_pooling=flags.uttr_pooling,
        uttr_attention_size=flags.uttr_attention_size,
        out_bias=flags.out_bias,
        # ffn
        feature_size=flags.feature_size,
        uttr_activation=flags.uttr_activation,
        sess_activation=flags.sess_activation,
        # training
        batch_size=flags.batch_size,
        num_epochs=flags.num_epochs,
        num_ckpt_epochs=flags.num_ckpt_epochs,
        # optimizer
        colocate_gradients_with_ops=flags.colocate_gradients_with_ops,
        learning_rate=flags.learning_rate,
        optimizer=flags.optimizer,
        start_decay_step=flags.start_decay_step,
        decay_steps=flags.decay_steps,
        decay_factor=flags.decay_factor,
        max_gradient_norm=flags.max_gradient_norm,
        # evaluation/prediction
        eval_output_folder=flags.eval_output_folder,
        ckpt=flags.ckpt,
        eval_input_path=flags.eval_input_path,
        eval_target_path=flags.eval_target_path,
        eval_batch_size=flags.eval_batch_size,
        predictions_filename=flags.predictions_filename,
        # Other
        random_seed=flags.random_seed,
        log_device_placement=flags.log_device_placement,
        gpu=flags.gpu,
        timeline=flags.timeline,
        save_trans_params=flags.save_trans_params
    )


def process_hparams(hparams):
    # load pretrained embedding matrix if input_emb_file is given
    hparams.input_emb_weights = np.loadtxt(hparams.input_emb_file, delimiter=' ') if hparams.input_emb_file else None
    # if pretrained embeddings are given, infer the value of hparams.input_emb_size
    hparams.input_emb_size = hparams.input_emb_weights.shape[
        1] if hparams.input_emb_weights is not None else hparams.input_emb_size
    hparams.add_hparam("input_emb_pretrain", hparams.input_emb_file is not None)
    # extend vocabulary if necessary to add <pad> and <unk> symbols.
    vocab_size = None
    vocab_path = None
    if hparams.vocab_path is not None:
        vocab_size, vocab_path = vocab_utils.check_vocab(hparams.vocab_path, hparams.out_dir,
                                                         unk=hparams.unk, pad=hparams.pad)
    hparams.add_hparam("vocab_size", vocab_size)
    hparams.set_hparam("vocab_path", vocab_path)
    hparams.uttr_units = [int(units) for units in hparams.uttr_units.split(",")]
    hparams.sess_units = [int(units) for units in hparams.sess_units.split(",")]
    hparams.uttr_hid_to_out_dropout = [float(d) for d in hparams.uttr_hid_to_out_dropout.split(",")]
    hparams.sess_hid_to_out_dropout = [float(d) for d in hparams.sess_hid_to_out_dropout.split(",")]
    hparams.uttr_activation = [act_name for act_name in hparams.uttr_activation.split(",")]
    hparams.sess_activation = [act_name for act_name in hparams.sess_activation.split(",")]
    hparams.filter_sizes = [int(f_size) for f_size in hparams.filter_sizes.split(",")]
    return hparams


def check_hparams(hparams):
    """ Check for incompatibilities between the parameters."""
    if hparams.model_architecture == "h-rnn-rnn" or hparams.model_architecture == "h-rnn-cnn" or \
            hparams.model_architecture == "h-rnn-rnn-crf" or hparams.model_architecture == "rnn" or \
            hparams.model_architecture == "cnn":
        if hparams.vocab_path is None: raise ValueError("If RNN or CNN in the utterance level, input vocab "
                                                        "should not be None.")


def process_or_load_hparams(out_dir, default_hparams, hparams_path):
    hparams = default_hparams
    # if a Hparams path is given as argument, override the default_hparams.
    hparams = utils.maybe_parse_standard_hparams(hparams, hparams_path)
    # extend HParams to add some parameters necessary for the training.
    hparams = process_hparams(hparams)
    # check compatibility of HParams
    check_hparams(hparams)
    # Save HParams
    utils.save_hparams(out_dir, hparams)

    # Print HParams
    print("Print hyperparameters:")
    utils.print_hparams(hparams)
    return hparams


def run_main(default_hparams, train_fn, evaluation_fn):
    out_dir = default_hparams.out_dir
    if not tf.gfile.Exists(out_dir): tf.gfile.MakeDirs(out_dir)
    hparams = process_or_load_hparams(out_dir, default_hparams, default_hparams.hparams_path)
    # restrict tensoflow to run only in the specified gpu. This has no effect if run on a machine with no gpus.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(hparams.gpu)
    # if there hparams.eval_output_folder is not None. we proceed with evaluation based on an existing model.
    # Otherwise, we train a new model.
    if hparams.eval_output_folder:
        if not tf.gfile.Exists(hparams.eval_output_folder): tf.gfile.MakeDirs(hparams.eval_output_folder)
        ckpt = hparams.ckpt
        # if no checkpoint path is given as input, load the latest checkpoint from the output folder.
        if not ckpt:
            ckpt = tf.train.latest_checkpoint(out_dir)
        # Evaluation of an existing model.
        evaluation_fn(hparams, ckpt)
    else:
        # Train a new model.
        train_fn(hparams)


def main(flags):
    # create hparams from command line arguments. hparams objects holds all the parameters that are required to run the models.
    default_hparams = create_hparams(flags)
    train_fn = train.train
    evaluation_fn = evaluation.evaluate
    run_main(default_hparams, train_fn, evaluation_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # add the possible command line arguments to the parser.
    add_arguments(parser)
    # parse command line args into the flags object.
    flags, unparsed = parser.parse_known_args()
    main(flags)

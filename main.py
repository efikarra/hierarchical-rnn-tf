import argparse
import tensorflow as tf
import os
import model_helper
import train
import utils
import evaluation
import vocab_utils


def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # Data
    parser.add_argument("--train_input_path", type=str, default=None,
                        help="Train input file path.")
    parser.add_argument("--train_target_path", type=str, default=None,
                        help="Train target file path.")
    parser.add_argument("--val_input_path", type=str, default=None,
                        help="Validation input file path for validation dataset.")
    parser.add_argument("--val_target_path", type=str, default=None,
                        help="Validation target file path for validation dataset.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Store log/model files.")
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
    parser.add_argument("--model_architecture", type=str, default="simple-rnn",
                        help="h-rnn-rnn | h-rnn-ffn | h-rnn-cnn | rnn |ffn. Model architecture.")
    parser.add_argument("--emb_size", type=int, default=32, help="Input embedding size.")
    parser.add_argument("--input_emb_trainable", type=bool, default=True, help="Train embedding layer weights.")
    parser.add_argument("--forget_bias", type=float, default=1.0,
                        help="Forget bias for BasicLSTMCell.")
    parser.add_argument("--uttr_time_major", type="bool", nargs="?", const=True,
                        default=False,
                        help="Whether to use time-major mode for utterance dynamic RNN.")
    parser.add_argument("--sess_time_major", type="bool", nargs="?", const=True,
                        default=False,
                        help="Whether to use time-major mode for session dynamic RNN.")
    # initializer
    parser.add_argument("--init_op", type=str, default="uniform",
                        help="uniform | glorot_normal | glorot_uniform")
    parser.add_argument("--init_weight", type=float, default=0.1,
                        help=("for uniform init_op, initialize weights "
                              "between [-this, this]."))

    # hierarchical rnn
    parser.add_argument("--out_bias", type=bool, default=True, help="Whether to use bias in the output layer.")
    parser.add_argument("--uttr_units", type=str, default='32', help="Hidden units of utterance model. "
                                                                     "A list of units separated by comma should be given"
                                                                     " for multiple layers.")
    parser.add_argument("--uttr_layers", type=int, default=1, help="Number of utterance model layers.")
    parser.add_argument("--sess_units", type=str, default='32', help="Hidden units of session model. "
                                                                     "A list of units separated by comma should be given"
                                                                     " for multiple layers.")
    parser.add_argument("--sess_layers", type=int, default=1, help="Number of session model layers.")
    parser.add_argument("--uttr_in_to_hid_dropout", type=str, default='0.0', help="List of input to hidden layer dropouts for utterance model.")
    parser.add_argument("--sess_in_to_hid_dropout", type=str, default='0.0', help="List of input to hidden layer dropouts for session model.")
    parser.add_argument("--uttr_rnn_type", type=str, default="uni",
                        help="uni | bi . For bi, we build enc_layers*2 bi-directional layers.")
    parser.add_argument("--sess_rnn_type", type=str, default="uni",
                        help="uni | bi . For bi, we build enc_layers*2 bi-directional layers.")
    parser.add_argument('--uttr_unit_type', type=str, default="rnn",
                        help="rnn | lstm | gru.")
    parser.add_argument('--sess_unit_type', type=str, default="rnn",
                        help="rnn | lstm | gru.")
    parser.add_argument('--uttr_pooling', type=str, default="last",
                        help="last | mean | attn | attn_context . Pooling scheme for utterance hidden states to represent an utterance.")
    parser.add_argument("--uttr_attention_size", type=int, default=32, help="Attention size if attention with context is used in the utterance level.")
    #ffn
    parser.add_argument("--feature_size", type=int, default=32, help="Number of features if ffn as utterance encoder.")
    parser.add_argument("--uttr_activation", type=str, default='relu',
                        help="List of activation functions for each layer of the utterance model.")
    # training
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--num_ckpt_epochs", type=int, default=2,
                        help="Number of epochs until the next checkpoint saving.")

    # optimizer
    parser.add_argument("--optimizer", type=str, default="sgd", help="sgd | adam")
    parser.add_argument("--learning_rate", type=float, default=0.1,
                        help="Learning rate. Adam: 0.001 | 0.0001")
    parser.add_argument("--start_decay_step", type=int, default=0,
                        help="When we start to decay")
    parser.add_argument("--decay_steps", type=int, default=10000,
                        help="How frequent we decay")
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
    parser.add_argument("--random_seed",type=int,default=None,
                        help="Random seed (>0, set a specific seed).")
    parser.add_argument("--log_device_placement", type="bool", nargs="?",
                        const=True, default=False, help="Debug GPU allocation.")
    parser.add_argument("--timeline", type="bool", nargs="?",
                        const=True, default=False, help="Log timeline information.")

    # Evaluation/Prediction
    parser.add_argument("--eval_output_folder", type=str, default=None,
                        help="Output folder to save evaluation data.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint file.")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="Batch size for evaluation mode.")
    parser.add_argument("--predict_batch_size", type=int, default=32,
                        help="Batch size for prediction mode.")
    parser.add_argument("--eval_input_path", type=str, default=None,
                        help="Input file path to perform evaluation and/or prediction.")
    parser.add_argument("--eval_target_path", type=str, default=None,
                        help="Output file path to perform evaluation and prediction.")
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
        uttr_time_major=flags.uttr_time_major,
        sess_time_major=flags.sess_time_major,
        n_classes=flags.n_classes,
        forget_bias=flags.forget_bias,
        input_emb_size=flags.emb_size,
        input_emb_trainable=flags.input_emb_trainable,
        # initializer
        init_weight=flags.init_weight,
        init_op=flags.init_op,
        # hierarchical rnn
        uttr_units=flags.uttr_units,
        uttr_layers=flags.uttr_layers,
        sess_units=flags.sess_units,
        sess_layers=flags.sess_layers,
        uttr_in_to_hid_dropout=flags.uttr_in_to_hid_dropout,
        sess_in_to_hid_dropout=flags.sess_in_to_hid_dropout,
        uttr_rnn_type=flags.uttr_rnn_type,
        sess_rnn_type=flags.sess_rnn_type,
        uttr_unit_type=flags.uttr_unit_type,
        sess_unit_type=flags.sess_unit_type,
        uttr_pooling=flags.uttr_pooling,
        uttr_attention_size=flags.uttr_attention_size,
        out_bias=flags.out_bias,
        #ffn
        feature_size=flags.feature_size,
        uttr_activation=flags.uttr_activation,
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
        predict_batch_size=flags.predict_batch_size,
        predictions_filename = flags.predictions_filename,
        # Other
        random_seed=flags.random_seed,
        log_device_placement=flags.log_device_placement,
        gpu=flags.gpu,
        timeline=flags.timeline
    )


def extend_hparams(hparams):
    hparams.add_hparam("input_emb_pretrain", hparams.input_emb_file is not None)
    vocab_size = None
    vocab_path = None
    if hparams.vocab_path is not None:
        vocab_size, vocab_path = vocab_utils.check_vocab(hparams.vocab_path, hparams.out_dir,
                                                     unk=hparams.unk, pad=hparams.pad)
    hparams.add_hparam("vocab_size",vocab_size)
    hparams.set_hparam("vocab_path",vocab_path)
    hparams.uttr_units = [int(units) for units in hparams.uttr_units.split(",")]
    hparams.sess_units = [int(units) for units in hparams.sess_units.split(",")]
    hparams.uttr_in_to_hid_dropout = [float(d) for d in hparams.uttr_in_to_hid_dropout.split(",")]
    hparams.sess_in_to_hid_dropout = [float(d) for d in hparams.sess_in_to_hid_dropout.split(",")]
    hparams.uttr_activation = [act_name for act_name in hparams.uttr_activation.split(",")]
    return hparams


def check_hparams(hparams):
    if hparams.model_architecture == "h-rnn-rnn" or hparams.model_architecture == "h-ffn-rnn":
        if hparams.vocab_path is None: raise ValueError("If RNN or CNN in the utterance level, input vocab "
                                                        "should not be None.")
    pass


def create_or_load_hparams(out_dir, default_hparams, flags):
    # if the out_dir already contains hparams file, load these hparams.
    hparams = utils.load_hparams(out_dir)
    if not hparams:
        hparams = default_hparams
        hparams = utils.maybe_parse_standard_hparams(
            hparams, flags.hparams_path)
        hparams = extend_hparams(hparams)
        check_hparams(hparams)
    else:
        #ensure that the loaded hparams and the command line hparams are compatible. If not, the command line hparams are overwritten!
        hparams = utils.ensure_compatible_hparams(hparams, default_hparams, flags)

    # Save HParams
    if not tf.gfile.Exists(hparams.out_dir):
        tf.gfile.MakeDirs(hparams.out_dir)
    utils.save_hparams(out_dir, hparams)

    # Print HParams
    print("Print hyperparameters:")
    utils.print_hparams(hparams)
    return hparams


def run_main(flags, default_hparams, train_fn, evaluation_fn):
    out_dir = flags.out_dir
    if not tf.gfile.Exists(out_dir): tf.gfile.MakeDirs(out_dir)
    hparams = create_or_load_hparams(out_dir, default_hparams, flags)
    # restrict tensoflow to run only in the specified gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(hparams.gpu)
    # if there is an evaluation output folder in the command line arguments
    # we proceed with evaluation based on an existing model. Otherwise, we train a new model.
    if flags.eval_output_folder:
        # Evaluation
        ckpt = flags.ckpt
        if not ckpt:
            ckpt = tf.train.latest_checkpoint(out_dir)
        evaluation_fn(hparams, ckpt)
    else:
        train_fn(hparams)


def main(flags):
    # create hparams from command line arguments
    default_hparams = create_hparams(flags)
    train_fn = train.train
    evaluation_fn = evaluation.evaluate
    run_main(flags, default_hparams, train_fn, evaluation_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # add the possible command line arguments to the parser.
    add_arguments(parser)
    # parse command line args
    flags, unparsed = parser.parse_known_args()
    main(flags)

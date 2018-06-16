import main
import argparse


def test_flat_tfrecs(params):
    params.train_input_path = 'experiments/data/tfrecords/train_bow_uttr.tfrecord'
    params.train_target_path = None
    params.val_input_path = 'experiments/data/tfrecords/val_bow_uttr.tfrecord'
    params.val_target_path = None
    params.vocab_path = None

    # Test data files to run evaluation on.
    params.eval_input_path = "experiments/data/tfrecords/test_bow_uttr.tfrecord"
    params.eval_target_path = None


def test_flat_text(params):
    params.train_input_path = 'experiments/data/train_input_uttr.txt'
    params.train_target_path = 'experiments/data/train_target_uttr.txt'
    params.val_input_path = 'experiments/data/val_input_uttr.txt'
    params.val_target_path = 'experiments/data/val_target_uttr.txt'
    params.vocab_path = 'experiments/data/0.01vocab.txt'

    # Test data files to run evaluation on.
    params.eval_input_path = 'experiments/data/test_input_uttr.txt'
    params.eval_target_path = 'experiments/data/test_target_uttr.txt'


def test_hierarchical_text_sub(params):
    params.train_input_path = 'experiments/data/subsets/train_input_sess_100_sub.txt'
    params.train_target_path = 'experiments/data/subsets/train_target_sess_100_sub.txt'
    # Validation data files.
    params.val_input_path = 'experiments/data/subsets/val_input_sess_100_sub.txt'
    params.val_target_path = 'experiments/data/subsets/val_target_sess_100_sub.txt'
    params.vocab_path = 'experiments/data/subsets/0.00.1vocab_sub.txt'

    params.eval_input_path = 'experiments/data/subsets/val_input_sess_100_sub.txt'
    params.eval_target_path = 'experiments/data/subsets/val_target_sess_100_sub.txt'


def test_hierarchical_text(params):
    params.train_input_path = 'experiments/data/train_input_sess_100.txt'
    params.train_target_path = 'experiments/data/train_target_sess_100.txt'
    # Validation data files.
    params.val_input_path = 'experiments/data/val_input_sess_100.txt'
    params.val_target_path = 'experiments/data/val_target_sess_100.txt'
    params.vocab_path = 'experiments/data/0.00.1vocab.txt'

    params.eval_input_path = 'experiments/data/test_input_sess_100.txt'
    params.eval_target_path = 'experiments/data/test_target_sess_100.txt'


def test_hierarchical_tfrecs(params):
    params.train_input_path='experiments/data/tfrecords/train_bow_sess_100.tfrecord'
    params.train_target_path = None
    params.val_input_path='experiments/data/tfrecords/val_bow_sess_100.tfrecord'
    params.val_target_path = None
    params.vocab_path = None
    params.model_architecture = 'h-rnn-ffn'

    params.eval_input_path = 'experiments/data/tfrecords/val_bow_sess_100.tfrecord'
    params.eval_target_path = None



if __name__ == '__main__':
    params = argparse.Namespace()

    #Input data parameters.
    params.out_dir='experiments/out_model'
    params.n_classes = 27
    # params.input_emb_file = 'experiments/embeddings/glove.840B.300d_0.01vocab.txt'
    params.input_emb_file = None
    # params.hparams_path = params.out_dir + "/hparams"
    params.hparams_path = None
    # What symbols to use for unk and pad.
    params.unk='<unk>'
    params.pad='<pad>'
    params.feature_size=12624
    # Input sequence max length.

    # network
    # params.uttr_units = '32,32'
    params.uttr_units = '32'
    params.uttr_layers = 1
    params.sess_units = '32'
    params.sess_layers = 1
    # params.uttr_hid_to_out_dropout = '0.25,0.25'
    params.uttr_hid_to_out_dropout = '0.5'
    params.sess_hid_to_out_dropout = '0.1'
    params.uttr_rnn_type = 'uni'
    params.sess_rnn_type = 'uni'
    params.uttr_unit_type = 'gru'
    params.sess_unit_type = 'gru'
    params.uttr_pooling = 'attn_context'
    params.uttr_attention_size = 32
    params.input_emb_size = 300
    # cnn
    params.filter_sizes = '3,4'
    params.num_filters = 10
    params.pool_size = 1
    params.padding = 'valid'
    params.stride = 1

    # params.uttr_activation="sigmoid,sigmoid"
    params.uttr_activation = "relu"
    params.sess_activation="relu"

    # network
    params.init_op = 'uniform'
    params.init_weight = 0.1
    params.uttr_time_major = False
    params.sess_time_major = False
    params.input_emb_trainable = True
    params.out_bias=True
    params.forget_bias = 1.0
    params.connect_inp_to_out = True
    # training
    params.batch_size=2
    params.num_epochs=10
    params.num_ckpt_epochs=1
    # optimizer
    params.learning_rate=0.01
    params.optimizer='adam'
    params.colocate_gradients_with_ops = True
    params.start_decay_step = 0
    params.decay_steps = 10000
    params.decay_factor = 0.98
    params.max_gradient_norm = 5.0
    #Other
    # you don't care at all about the following 4 parameters.
    params.gpu = None
    params.random_seed = None
    params.log_device_placement = False
    params.timeline = False

    params.eval_output_folder=None
    # params.eval_output_folder = "experiments/eval_output"
    params.save_trans_params = True
    params.ckpt =None
    # params.ckpt='experiments/out_model/<CHECKPOINT_NAME>'

    params.eval_batch_size=2
    params.predict_batch_size=2
    #filename to save predictions on the test set. They will be saved in the eval_output_folder.
    params.predictions_filename='predictions.txt'

    test_hierarchical_text_sub(params)
    params.model_architecture="h-rnn-rnn"
    # params.model_architecture = "h-rnn-cnn"
    # params.model_architecture = "h-rnn-rnn-crf"

    # test_flat_text(params)
    # params.model_architecture = "rnn"
    # params.model_architecture = "cnn"

    main.main(params)

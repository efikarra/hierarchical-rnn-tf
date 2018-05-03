import json
import itertools

def print_ffn_params():
    with open("experiments/ffn_params.json","r") as f:
        params = json.load(f)

    param_combs=list(itertools.product(*params.values()))
    print params.items()
    params_names = {key:i for i,key in enumerate(params.keys())}

    gpu=2
    #ffn
    print("Params combinations: %d"%len(param_combs))
    for param in param_combs:
        print param
    for param in param_combs:
        model_architecture=param[params_names["model_architecture"]]
        learning_rate=param[params_names["learning_rate"]]
        uttr_units=param[params_names["uttr_units"]]
        uttr_layers=param[params_names["uttr_layers"]]
        uttr_in_to_hid_dropout=param[params_names["uttr_in_to_hid_dropout"]]
        batch_size=param[params_names["batch_size"]]
        num_epochs=param[params_names["num_epochs"]]
        uttr_activation=param[params_names["uttr_activation"]]
        name_suffix=model_architecture+"_lr_"+learning_rate+"_u_"+uttr_units\
                  +"_uttrl_"+uttr_layers+"_uttrdr_"+uttr_in_to_hid_dropout+"_b_"+batch_size+"_e_"+num_epochs+"_act_"+uttr_activation
        command = "stdbuf -oL python main.py --train_input_path=experiments/data/tfrecords/train_bow_uttr.tfrecord --val_input_path=experiments/data/tfrecords/val_bow_uttr.tfrecord " \
                "--out_dir=experiments/out_folders/ffn/out_model_"+name_suffix+" --n_classes=32 --num_ckpt_epochs=1 --optimizer=adam --model_architecture="\
                  + model_architecture+ " --learning_rate=" + learning_rate +\
                " --uttr_units="+ uttr_units+" --uttr_layers="+ uttr_layers+\
                " --uttr_in_to_hid_dropout="+ uttr_in_to_hid_dropout+ \
                " --feature_size=21012 --batch_size="+ batch_size+ " --num_epochs=" + num_epochs+ " --eval_batch_size="+batch_size+ \
                " --uttr_activation="+uttr_activation+" --gpu="+str(int(gpu%8))+" > logs_pcori/log_"+name_suffix+" 2>&1 &"
        if gpu%8==0:print("\n")
        gpu+=1
        print(command)


def print_rnn_params():
    with open("experiments/rnn_params.json", "r") as f:
        params = json.load(f)

    param_combs = list(itertools.product(*params.values()))
    print params.items()
    params_names = {key: i for i, key in enumerate(params.keys())}

    gpu = 5
    # ffn
    print("Params combinations: %d" % len(param_combs))
    for param in param_combs:
        print param
    for param in param_combs:
        model_architecture = param[params_names["model_architecture"]]
        learning_rate = param[params_names["learning_rate"]]
        uttr_units = param[params_names["uttr_units"]]
        uttr_layers = param[params_names["uttr_layers"]]
        uttr_in_to_hid_dropout = param[params_names["uttr_in_to_hid_dropout"]]
        batch_size = param[params_names["batch_size"]]
        num_epochs = param[params_names["num_epochs"]]
        uttr_rnn_type=param[params_names["uttr_rnn_type"]]
        uttr_pooling=param[params_names["uttr_pooling"]]
        uttr_unit_type=param[params_names["uttr_unit_type"]]
        emb_size=param[params_names["emb_size"]]
        init_op=param[params_names["init_op"]]
        name_suffix = model_architecture + "_lr_" + learning_rate + "_u_" + uttr_units \
                      + "_uttrl_" + uttr_layers + "_utype_"+ uttr_unit_type+"_uttrdr_" + \
                      uttr_in_to_hid_dropout + "_b_" + batch_size + "_e_" + num_epochs+"_uty_"+\
                      uttr_rnn_type+"_po_"+uttr_pooling+"_emb_"+emb_size+"_inop_"+init_op
        command = "stdbuf -oL python main.py --train_input_path=experiments/data/train_input_uttr.txt " \
                  "--train_target_path=experiments/data/train_target_uttr.txt --val_input_path=experiments/data/val_input_uttr.txt " \
                  "--val_target_path=experiments/data/val_target_uttr.txt --vocab_path=experiments/data/0.00.0vocab_uttr.txt "+ \
                  "--out_dir=experiments/out_folders/rnn/out_model_" + name_suffix + " --n_classes=32 --num_ckpt_epochs=1 --optimizer=adam --model_architecture=" \
                  + model_architecture + " --learning_rate=" + learning_rate + \
                  " --uttr_units=" + uttr_units + " --uttr_layers=" + uttr_layers + \
                  " --uttr_in_to_hid_dropout=" + uttr_in_to_hid_dropout + " --uttr_rnn_type="+uttr_rnn_type+ " --uttr_unit_type="+uttr_unit_type+\
                  " --uttr_pooling="+uttr_pooling+ " --emb_size="+emb_size+\
                  " --batch_size=" + batch_size + " --num_epochs=" + num_epochs + \
                  " --eval_batch_size=" + batch_size + " --init_op="+init_op+\
                  " --gpu=" + str(int(gpu % 8)) + " > logs_pcori/rnn/log_" + name_suffix + " 2>&1 &"
        if gpu % 8 == 0: print("\n")
        gpu += 1
        print(command)

def print_h_rnn_rnn_params():
    with open("experiments/h_rnn_rnn_params.json", "r") as f:
        params = json.load(f)

    param_combs = list(itertools.product(*params.values()))
    print params.items()
    params_names = {key: i for i, key in enumerate(params.keys())}

    gpu = 3
    # ffn
    print("Params combinations: %d" % len(param_combs))
    for param in param_combs:
        print param
    for param in param_combs:
        model_architecture = param[params_names["model_architecture"]]
        learning_rate = param[params_names["learning_rate"]]
        uttr_units = param[params_names["uttr_units"]]
        uttr_layers = param[params_names["uttr_layers"]]
        uttr_in_to_hid_dropout = param[params_names["uttr_in_to_hid_dropout"]]
        uttr_rnn_type=param[params_names["uttr_rnn_type"]]
        uttr_pooling=param[params_names["uttr_pooling"]]
        uttr_unit_type=param[params_names["uttr_unit_type"]]

        sess_units = param[params_names["sess_units"]]
        sess_layers = param[params_names["sess_layers"]]
        sess_in_to_hid_dropout = param[params_names["sess_in_to_hid_dropout"]]
        sess_rnn_type = param[params_names["sess_rnn_type"]]
        sess_unit_type = param[params_names["sess_unit_type"]]

        batch_size = param[params_names["batch_size"]]
        num_epochs = param[params_names["num_epochs"]]
        emb_size=param[params_names["emb_size"]]
        init_op=param[params_names["init_op"]]
        name_suffix = "h_rnn_rnn" + "_lr_" + learning_rate + "_u_" + uttr_units \
                      + "_uttrl_" + uttr_layers + "_utype_"+ uttr_unit_type+"_uttrdr_" + \
                      uttr_in_to_hid_dropout + "_b_" + batch_size +"_uty_"+\
                      uttr_rnn_type+"_su_"+sess_units+"_sl_"+sess_layers+"_sdr_"+sess_in_to_hid_dropout+"_sty_"+sess_rnn_type+"_su_"+sess_unit_type+"_po_"+uttr_pooling+"_emb_"+emb_size+"_inop_"+init_op
        command = "stdbuf -oL python main.py --train_input_path=experiments/data/train_input_sess_100.txt " \
                  "--train_target_path=experiments/data/train_target_sess_100.txt --val_input_path=experiments/data/val_input_sess_100.txt " \
                  "--val_target_path=experiments/data/val_target_sess_100.txt --vocab_path=experiments/data/0.00.0vocab_sess_100.txt "+ \
                  "--out_dir=experiments/out_folders/h_rnn_rnn/out_model_" + name_suffix + " --n_classes=32 --num_ckpt_epochs=1 --optimizer=adam --model_architecture=" \
                  + model_architecture + " --learning_rate=" + learning_rate + \
                  " --uttr_units=" + uttr_units + " --uttr_layers=" + uttr_layers + \
                  " --uttr_in_to_hid_dropout=" + uttr_in_to_hid_dropout + " --uttr_rnn_type="+uttr_rnn_type+ " --uttr_unit_type="+uttr_unit_type+\
                  " --uttr_pooling="+uttr_pooling+ " --sess_units=" + sess_units + " --sess_layers=" + sess_layers + \
                  " --sess_in_to_hid_dropout=" + sess_in_to_hid_dropout + " --sess_rnn_type="+sess_rnn_type+ " --sess_unit_type="+sess_unit_type+\
                  " --uttr_pooling="+uttr_pooling+\
                  " --emb_size="+emb_size+\
                  " --batch_size=" + batch_size + " --num_epochs=" + num_epochs + \
                  " --eval_batch_size=" + batch_size + " --init_op="+init_op+\
                  " --gpu=" + str(int(gpu % 8)) + " > logs_pcori/h_rnn_rnn/log_" + name_suffix + " 2>&1 &"
        if gpu % 8 == 0: print("\n")
        gpu += 1
        print(command)


def print_h_rnn_ffn_params():
    with open("experiments/h_rnn_ffn_params.json", "r") as f:
        params = json.load(f)

    param_combs = list(itertools.product(*params.values()))
    print params.items()
    params_names = {key: i for i, key in enumerate(params.keys())}

    gpu = 4
    # ffn
    print("Params combinations: %d" % len(param_combs))
    for param in param_combs:
        print param
    for param in param_combs:
        model_architecture = param[params_names["model_architecture"]]
        learning_rate = param[params_names["learning_rate"]]
        uttr_units = param[params_names["uttr_units"]]
        uttr_layers = param[params_names["uttr_layers"]]
        uttr_in_to_hid_dropout = param[params_names["uttr_in_to_hid_dropout"]]

        sess_units = param[params_names["sess_units"]]
        sess_layers = param[params_names["sess_layers"]]
        sess_in_to_hid_dropout = param[params_names["sess_in_to_hid_dropout"]]
        sess_rnn_type = param[params_names["sess_rnn_type"]]
        sess_unit_type = param[params_names["sess_unit_type"]]

        batch_size = param[params_names["batch_size"]]
        num_epochs = param[params_names["num_epochs"]]
        init_op=param[params_names["init_op"]]
        uttr_activation=param[params_names["uttr_activation"]]
        name_suffix = "h_rnn_ffn" + "_lr_" + learning_rate + "_u_" + uttr_units \
                      + "_uttrl_" + uttr_layers + "_uact_"+uttr_activation +"_utype_"+ \
                      uttr_in_to_hid_dropout + "_b_" + batch_size +"_su_"+sess_units+"_sl_"+sess_layers+\
                      "_sdr_"+sess_in_to_hid_dropout+"_sty_"+sess_rnn_type+"_su_"+sess_unit_type+"_inop_"+init_op
        command = "stdbuf -oL python main.py --train_input_path=experiments/data/tfrecords/train_bow_sess_100.tfrecord " \
                  "--val_input_path=experiments/data/tfrecords/val_bow_sess_100.tfrecord " \
                  "--out_dir=experiments/out_folders/h_rnn_ffn/out_model_" + name_suffix + " --n_classes=32 --feature_size=12624" \
                                                                                           " --num_ckpt_epochs=1 --optimizer=adam --model_architecture=" \
                  + model_architecture + " --max_gradient_norm=3.0 --learning_rate=" + learning_rate + \
                  " --uttr_units=" + uttr_units + " --uttr_layers=" + uttr_layers + " --uttr_activation="+uttr_activation+\
                  " --uttr_in_to_hid_dropout=" + uttr_in_to_hid_dropout + \
                  " --sess_units=" + sess_units + " --sess_layers=" + sess_layers + \
                  " --sess_in_to_hid_dropout=" + sess_in_to_hid_dropout + " --sess_rnn_type="+sess_rnn_type+ " --sess_unit_type="+sess_unit_type+\
                  " --batch_size=" + batch_size + " --num_epochs=" + num_epochs + \
                  " --eval_batch_size=" + batch_size + " --init_op="+init_op+\
                  " --gpu=" + str(int(gpu % 8)) + " > logs_pcori/h_rnn_ffn/log_" + name_suffix + " 2>&1 &"
        if gpu % 8 == 0: print("\n")
        gpu += 1
        print(command)

# print_h_rnn_ffn_params()
# print_h_rnn_rnn_params()
# print_rnn_params()
print_ffn_params()
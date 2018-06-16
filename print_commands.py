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
        uttr_hid_to_out_dropout=param[params_names["uttr_hid_to_out_dropout"]]
        batch_size=param[params_names["batch_size"]]
        num_epochs=param[params_names["num_epochs"]]
        uttr_activation=param[params_names["uttr_activation"]]
        name_suffix=model_architecture+"_lr_"+learning_rate+"_u_"+uttr_units\
                  +"_uttrl_"+uttr_layers+"_uttrdr_"+uttr_hid_to_out_dropout+"_b_"+batch_size+"_act_"+uttr_activation
        command = "stdbuf -oL python main.py --train_input_path=experiments/data/tfrecords/train_bow_uttr.tfrecord --val_input_path=experiments/data/tfrecords/val_bow_uttr.tfrecord " \
                "--out_dir=experiments/out_folders/ffn/out_model_"+name_suffix+" --n_classes=27 --num_ckpt_epochs=1 --optimizer=adam --model_architecture="\
                  + model_architecture+ " --learning_rate=" + learning_rate +\
                " --uttr_units="+ uttr_units+" --uttr_layers="+ uttr_layers+\
                " --uttr_hid_to_out_dropout="+ uttr_hid_to_out_dropout+ \
                " --feature_size=21012 --batch_size="+ batch_size+ " --num_epochs=" + num_epochs+ " --eval_batch_size="+batch_size+ \
                " --uttr_activation="+uttr_activation+" --gpu="+str(int(gpu%8))+" > logs_pcori/ffn/log_"+name_suffix+" 2>&1 &"
        if gpu%8==0:print("\n")
        gpu+=1
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
        uttr_hid_to_out_dropout = param[params_names["uttr_hid_to_out_dropout"]]

        sess_units = param[params_names["sess_units"]]
        sess_layers = param[params_names["sess_layers"]]
        sess_hid_to_out_dropout = param[params_names["sess_hid_to_out_dropout"]]
        sess_rnn_type = param[params_names["sess_rnn_type"]]
        sess_unit_type = param[params_names["sess_unit_type"]]

        batch_size = param[params_names["batch_size"]]
        num_epochs = param[params_names["num_epochs"]]
        init_op=param[params_names["init_op"]]
        uttr_activation=param[params_names["uttr_activation"]]
        name_suffix = "h_rnn_ffn" + "_lr_" + learning_rate + "_u_" + uttr_units \
                      + "_uttrl_" + uttr_layers + "_uact_"+uttr_activation +"_utype_"+ \
                      uttr_hid_to_out_dropout + "_b_" + batch_size +"_su_"+sess_units+"_sl_"+sess_layers+\
                      "_sdr_"+sess_hid_to_out_dropout+"_sty_"+sess_rnn_type+"_su_"+sess_unit_type+"_inop_"+init_op
        command = "stdbuf -oL python main.py --train_input_path=experiments/data/tfrecords/train_bow_sess_100.tfrecord " \
                  "--val_input_path=experiments/data/tfrecords/val_bow_sess_100.tfrecord " \
                  "--out_dir=experiments/out_folders/h_rnn_ffn/out_model_" + name_suffix + " --n_classes=27 --feature_size=12624" \
                                                                                           " --num_ckpt_epochs=1 --optimizer=adam --model_architecture=" \
                  + model_architecture + " --max_gradient_norm=3.0 --learning_rate=" + learning_rate + \
                  " --uttr_units=" + uttr_units + " --uttr_layers=" + uttr_layers + " --uttr_activation="+uttr_activation+\
                  " --uttr_hid_to_out_dropout=" + uttr_hid_to_out_dropout + \
                  " --sess_units=" + sess_units + " --sess_layers=" + sess_layers + \
                  " --sess_hid_to_out_dropout=" + sess_hid_to_out_dropout + " --sess_rnn_type="+sess_rnn_type+ " --sess_unit_type="+sess_unit_type+\
                  " --batch_size=" + batch_size + " --num_epochs=" + num_epochs + \
                  " --eval_batch_size=" + batch_size + " --init_op="+init_op+\
                  " --gpu=" + str(int(gpu % 8)) + " > logs_pcori/h_rnn_ffn/log_" + name_suffix + " 2>&1 &"
        if gpu % 8 == 0: print("\n")
        gpu += 1
        print(command)


def print_cnn_params(eval=False):
    with open("experiments/cnn_params.json","r") as f:
        params = json.load(f)

    param_combs=list(itertools.product(*params.values()))
    print params.items()
    params_names = {key:i for i,key in enumerate(params.keys())}

    gpu=0
    print("Params combinations: %d"%len(param_combs))
    for param in param_combs:
        print param
    for param in param_combs:
        model_architecture=param[params_names["model_architecture"]]
        learning_rate=param[params_names["learning_rate"]]
        filter_sizes=param[params_names["filter_sizes"]]
        num_filters=param[params_names["num_filters"]]
        padding=param[params_names["padding"]]
        stride=param[params_names["stride"]]
        uttr_hid_to_out_dropout=param[params_names["uttr_hid_to_out_dropout"]]
        batch_size=param[params_names["batch_size"]]
        num_epochs=param[params_names["num_epochs"]]
        uttr_activation=param[params_names["uttr_activation"]]
        input_emb_size = param[params_names["input_emb_size"]]
        input_emb_file = param[params_names["input_emb_file"]]
        emb_path_suffix = " --input_emb_file=" + input_emb_file if input_emb_file != "None" else ""
        emb_name_suffix = "_emb_" + "".join(input_emb_file.split(".")[1:3]) if input_emb_file != "None" else ""

        name_suffix=model_architecture+"_lr_"+learning_rate+"_nf_"+num_filters\
                  +"_fs_"+filter_sizes+"_uttrdr_"+uttr_hid_to_out_dropout+"_b_"+batch_size+\
                    "_act_"+uttr_activation+"_p_"+padding+"_str_"+stride+"_emb_"+input_emb_size+emb_name_suffix

        eval_folder = "" if not eval else " --eval_output_folder=experiments/eval_output/cnn"
        eval_input_path = "" if not eval else " --eval_input_path=experiments/data/test_input_uttr.txt"
        eval_target_path = "" if not eval else " --eval_target_path=experiments/data/test_target_uttr.txt"
        log = "logs_pcori/cnn/log_" + name_suffix if not eval else "logs_pcori/cnn/log_eval"
        command = "stdbuf -oL python main.py --train_input_path=experiments/data/train_input_uttr.txt " \
                  "--train_target_path=experiments/data/train_target_uttr.txt --val_input_path=experiments/data/val_input_uttr.txt " \
                  "--val_target_path=experiments/data/val_target_uttr.txt --vocab_path=experiments/data/0.00.1vocab.txt "+ \
                "--out_dir=experiments/out_folders/cnn/out_model_"+name_suffix+" --n_classes=27 --num_ckpt_epochs=1 --optimizer=adam --model_architecture="\
                  + model_architecture+ " --learning_rate=" + learning_rate +\
                " --filter_sizes="+ filter_sizes+" --num_filters="+ num_filters+" --padding="+padding+" --stride="+stride+\
                " --uttr_hid_to_out_dropout="+ uttr_hid_to_out_dropout+ \
                " --batch_size="+ batch_size+ " --num_epochs=" + num_epochs+ " --eval_batch_size="+batch_size+ \
                " --input_emb_size="+input_emb_size+" --uttr_activation="+uttr_activation+emb_path_suffix+\
                eval_folder+eval_input_path+eval_target_path+\
                " --gpu=" + str(int(gpu % 8)) + " > "+log + " 2>&1 &"
        if gpu%8==0:print("\n")
        gpu+=1
        print(command)


def print_rnn_params(eval=False):
    with open("experiments/rnn_params.json", "r") as f:
        params = json.load(f)

    param_combs = list(itertools.product(*params.values()))
    print params.items()
    params_names = {key: i for i, key in enumerate(params.keys())}

    gpu = 2
    # ffn
    print("Params combinations: %d" % len(param_combs))
    for param in param_combs:
        print param

    for param in param_combs:
        vocab_path=param[params_names["vocab_path"]]
        model_architecture = param[params_names["model_architecture"]]
        learning_rate = param[params_names["learning_rate"]]
        uttr_units = param[params_names["uttr_units"]]
        uttr_layers = param[params_names["uttr_layers"]]
        uttr_hid_to_out_dropout = param[params_names["uttr_hid_to_out_dropout"]]
        batch_size = param[params_names["batch_size"]]
        num_epochs = param[params_names["num_epochs"]]
        uttr_rnn_type=param[params_names["uttr_rnn_type"]]
        uttr_pooling=param[params_names["uttr_pooling"]]
        uttr_unit_type=param[params_names["uttr_unit_type"]]
        input_emb_size=param[params_names["input_emb_size"]]
        init_op=param[params_names["init_op"]]

        input_emb_file=param[params_names["input_emb_file"]]
        emb_path_suffix =" --input_emb_file="+input_emb_file if input_emb_file != "None" else ""
        emb_name_suffix = "_emb_"+"".join(input_emb_file.split(".")[1:3]) if input_emb_file != "None" else ""

        name_suffix = model_architecture + "_lr_" + learning_rate + "_u_" + uttr_units \
                      + "_uttrl_" + uttr_layers + "_utype_"+ uttr_unit_type+"_uttrdr_" + \
                      uttr_hid_to_out_dropout + "_b_" + batch_size + "_uty_"+\
                      uttr_rnn_type+"_po_"+uttr_pooling+"_emb_"+input_emb_size+"_inop_"+init_op+emb_name_suffix
        #+"_v_"+vocab_path.split("/")[-1].replace("vocab.txt","")
        eval_folder = "" if not eval else " --eval_output_folder=experiments/eval_output/rnn"
        eval_input_path = "" if not eval else " --eval_input_path=experiments/data/test_input_uttr.txt"
        eval_target_path = "" if not eval else " --eval_target_path=experiments/data/test_target_uttr.txt"
        log="logs_pcori/rnn/log_" + name_suffix if not eval else "logs_pcori/rnn/log_eval"
        command = "stdbuf -oL python main.py --train_input_path=experiments/data/train_input_uttr.txt " \
                  "--train_target_path=experiments/data/train_target_uttr.txt --val_input_path=experiments/data/val_input_uttr.txt " \
                  "--val_target_path=experiments/data/val_target_uttr.txt --vocab_path="+vocab_path+ \
                  " --out_dir=experiments/out_folders/rnn/out_model_" + name_suffix + " --n_classes=27 --num_ckpt_epochs=1 --optimizer=adam --model_architecture=" \
                  + model_architecture + " --learning_rate=" + learning_rate + \
                  " --uttr_units=" + uttr_units + " --uttr_layers=" + uttr_layers + emb_path_suffix + \
                  " --uttr_hid_to_out_dropout=" + uttr_hid_to_out_dropout + " --uttr_rnn_type="+uttr_rnn_type+ " --uttr_unit_type="+uttr_unit_type+\
                  " --uttr_pooling="+uttr_pooling+ " --input_emb_size="+input_emb_size+\
                  " --batch_size=" + batch_size + " --num_epochs=" + num_epochs + \
                  eval_folder+eval_input_path+eval_target_path+\
                  " --gpu=" + str(int(gpu % 8)) + " > "+log + " 2>&1 &"
        if gpu % 8 == 0: print("\n")
        gpu += 1
        print(command)


def name_from_model_architecture(model_architecture):
    return "_".join(model_architecture.split("-"))


def print_h_rnn_rnn_params(eval=False, sess_size=100, bin_type=None):
    if sess_size is None: sess_size="full"
    with open("experiments/h_rnn_rnn_params.json", "r") as f:
        params = json.load(f)

    param_combs = list(itertools.product(*params.values()))
    print params.items()
    params_names = {key: i for i, key in enumerate(params.keys())}

    gpu = 6
    # ffn
    print("Params combinations: %d" % len(param_combs))
    for param in param_combs:
        print param

    if bin_type=="sgmt_bin":
        train_target_path = "experiments/data/train_target_sess_"+str(sess_size)+"_sgmt_bin.txt"
        val_target_path = "experiments/data/val_target_sess_" + str(sess_size) + "_sgmt_bin.txt"
        test_target_path = "experiments/data/test_target_sess_" + str(sess_size) + "_sgmt_bin.txt"
        bin_suffix = "_sgmt_bin"
    elif bin_type=="sgmt_mult":
        train_target_path = "experiments/data/train_target_sess_" + str(sess_size) + "_sgmt_mult.txt"
        val_target_path = "experiments/data/val_target_sess_" + str(sess_size) + "_sgmt_mult.txt"
        test_target_path = "experiments/data/test_target_sess_" + str(sess_size) + "_sgmt_mult.txt"
        bin_suffix = "_sgmt_mult"
    else:
        train_target_path = "experiments/data/val_target_sess_" + str(sess_size) + ".txt"
        val_target_path = "experiments/data/val_target_sess_" + str(sess_size) + ".txt"
        test_target_path = "experiments/data/val_target_sess_" + str(sess_size) + ".txt"
        bin_suffix = ""

    if bin_type is None:
        n_classes=27
    elif bin_type=="sgmt_bin":
        n_classes=2
    else:
        n_classes = 54
    for param in param_combs:
        model_architecture = param[params_names["model_architecture"]]
        learning_rate = param[params_names["learning_rate"]]
        uttr_units = param[params_names["uttr_units"]]
        uttr_layers = param[params_names["uttr_layers"]]
        uttr_hid_to_out_dropout = param[params_names["uttr_hid_to_out_dropout"]]
        uttr_rnn_type=param[params_names["uttr_rnn_type"]]
        uttr_pooling=param[params_names["uttr_pooling"]]
        uttr_unit_type=param[params_names["uttr_unit_type"]]

        sess_units = param[params_names["sess_units"]]
        sess_layers = param[params_names["sess_layers"]]
        sess_hid_to_out_dropout = param[params_names["sess_hid_to_out_dropout"]]
        sess_rnn_type = param[params_names["sess_rnn_type"]]
        sess_unit_type = param[params_names["sess_unit_type"]]

        connect_inp_to_out = param[params_names["connect_inp_to_out"]]

        batch_size = param[params_names["batch_size"]]
        num_epochs = param[params_names["num_epochs"]]
        input_emb_size=param[params_names["input_emb_size"]]
        init_op=param[params_names["init_op"]]
        input_emb_file = param[params_names["input_emb_file"]]
        emb_path_suffix = " --input_emb_file=" + input_emb_file if input_emb_file != "None" else ""
        emb_name_suffix = "_emb_" + "".join(input_emb_file.split(".")[1:3]) if input_emb_file != "None" else ""
        connect_inp_to_out_suff = "_inp_" if connect_inp_to_out=="True" else ""
        model_architecture_name=name_from_model_architecture(model_architecture)
        name_suffix = model_architecture_name + "_lr_" + learning_rate + "_u_" + uttr_units \
                      + "_uttrl_" + uttr_layers + "_utype_"+ uttr_unit_type+"_uttrdr_" + \
                      uttr_hid_to_out_dropout + "_b_" + batch_size +"_uty_"+\
                      uttr_rnn_type+"_su_"+sess_units+"_sl_"+sess_layers+"_sdr_"+sess_hid_to_out_dropout+"_sty_"+\
                      sess_rnn_type+"_su_"+sess_unit_type+"_po_"+uttr_pooling+"_emb_"+input_emb_size+"_inop_"+init_op+\
                     emb_name_suffix+connect_inp_to_out_suff+"sess_"+str(sess_size)+bin_suffix

        eval_folder = "" if not eval else " --eval_output_folder=experiments/eval_output/"+model_architecture_name
        eval_input_path = "" if not eval else " --eval_input_path=experiments/data/test_input_sess_"+str(sess_size)+".txt"
        eval_target_path = "" if not eval else " --eval_target_path="+test_target_path
        log = "logs_pcori/"+model_architecture_name+"/log_" + name_suffix if not eval else "logs_pcori/"+model_architecture_name+"/log_eval"

        command = "stdbuf -oL python main.py --train_input_path=experiments/data/train_input_sess_"+str(sess_size)+".txt " \
                  "--train_target_path="+train_target_path+" --val_input_path=experiments/data/val_input_sess_"+str(sess_size)+".txt " \
                  "--val_target_path="+val_target_path+" --vocab_path=experiments/data/0.00.1vocab.txt "+ \
                  "--out_dir=experiments/out_folders/"+model_architecture_name+"/out_model_" + name_suffix + " --n_classes="+str(n_classes)+" --num_ckpt_epochs=1 --optimizer=adam --model_architecture=" \
                  + model_architecture + " --max_gradient_norm=5.0 --learning_rate=" + learning_rate + \
                  " --uttr_units=" + uttr_units + " --uttr_layers=" + uttr_layers + \
                  " --uttr_hid_to_out_dropout=" + uttr_hid_to_out_dropout + " --uttr_rnn_type="+uttr_rnn_type+ " --uttr_unit_type="+uttr_unit_type+\
                  " --uttr_pooling="+uttr_pooling+ " --sess_units=" + sess_units + " --sess_layers=" + sess_layers + \
                  " --sess_hid_to_out_dropout=" + sess_hid_to_out_dropout + " --sess_rnn_type="+sess_rnn_type+ " --sess_unit_type="+sess_unit_type+\
                  " --uttr_pooling="+uttr_pooling+\
                  " --input_emb_size="+input_emb_size+" --connect_inp_to_out="+connect_inp_to_out+\
                  " --batch_size=" + batch_size + " --num_epochs=" + num_epochs + \
                  " --eval_batch_size=" + str(4) + " --init_op="+init_op+emb_path_suffix+\
                  eval_folder+eval_input_path+eval_target_path+\
                  " --gpu=" + str(int(gpu % 8)) + " > "+log + " 2>&1 &"
        if gpu % 8 == 0: print("\n")
        gpu += 1
        print(command)

# print_h_rnn_ffn_params()
# print_h_rnn_rnn_params()
# print_rnn_params(eval=True)
print_h_rnn_rnn_params(eval=False,sess_size=400,bin_type="sgmt_mult")
# print_cnn_params()
#
# print_ffn_params()
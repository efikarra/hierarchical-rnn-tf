import io

import tensorflow as tf
import codecs
import os
import json


def get_file_name(hparams):
    filename = hparams.model_architecture+"_"+hparams.num_units+"_"+hparams.num_layers+"_"+hparams.in_to_hidden_dropout\
               +"_"+hparams.rnn_type+"_"+hparams.unit_type+"_"+hparams.emb_size+"_"+hparams.batch_size


def maybe_parse_standard_hparams(hparams, hparams_path):
  """Override hparams values with existing standard hparams config."""
  if not hparams_path:
    return hparams

  if tf.gfile.Exists(hparams_path):
    print("# Loading standard hparams from %s" % hparams_path)
    with tf.gfile.GFile(hparams_path, "r") as f:
      hparams.parse_json(f.read())

  return hparams


def save_hparams(out_dir, hparams):
  """Save hparams."""
  hparams_file = os.path.join(out_dir, "hparams")
  print("  saving hparams to %s" % hparams_file)
  with codecs.getwriter("utf-8")(tf.gfile.GFile(hparams_file, "wb")) as f:
    f.write(hparams.to_json())


def load_hparams(model_dir):
  """Load hparams from an existing model directory."""
  hparams_file = os.path.join(model_dir, "hparams")
  if tf.gfile.Exists(hparams_file):
    print("# Loading hparams from %s" % hparams_file)
    with codecs.getreader("utf-8")(tf.gfile.GFile(hparams_file, "rb")) as f:
      try:
        hparams_values = json.load(f)
        hparams = tf.contrib.training.HParams(**hparams_values)
      except ValueError:
        print("  can't load hparams file")
        return None
    return hparams
  else:
    return None


def print_hparams(hparams, skip_patterns=None):
  """Print hparams, can skip keys based on pattern."""
  values = hparams.values()
  for key in sorted(values.keys()):
    if not skip_patterns or all(
        [skip_pattern not in key for skip_pattern in skip_patterns]):
      print("  %s=%s" % (key, str(values[key])))


def ensure_compatible_hparams(hparams,default_hparams,flags):
    """Make sure the loaded hparams is compatible with new changes."""
    default_hparams = maybe_parse_standard_hparams(
        default_hparams,flags.hparams_path)

    # For compatible reason, if there are new fields in default_hparams,
    #   we add them to the current hparams
    default_config = default_hparams.values()
    config = hparams.values()
    for key in default_config:
        if key not in config:
            hparams.add_hparam(key,default_config[key])

    # Make sure that the loaded model has latest values for the below keys
    updated_keys = [
        "out_dir","num_ckpt_epochs","num_epochs","gpu","batch_size","eval_input_path","eval_target_path",
        "eval_output_folder","eval_batch_size","predict_batch_size","predictions_filename"
    ]
    for key in updated_keys:
        if key in default_config and getattr(hparams,key) != default_config[key]:
            print("# Updating hparams.%s: %s -> %s" %
                            (key,str(getattr(hparams,key)),str(default_config[key])))
            setattr(hparams,key,default_config[key])
    return hparams


def get_config_proto(log_device_placement=False, allow_soft_placement=True):
    config_proto = tf.ConfigProto(log_device_placement=log_device_placement,
                                  allow_soft_placement = allow_soft_placement)
    # allocate as much GPU memory as is needed, based on runtime allocations.
    config_proto.gpu_options.allow_growth = False
    return config_proto


def save_file(filepath, data):
    with io.open(filepath, 'w', encoding="utf-8") as f:
        newline = ""
        for d in data:
            f.write(unicode(newline+str(d)))
            newline="\n"


def load_file(filepath):
    with io.open(filepath, "r", encoding="utf-8") as f:
        data=f.read().splitlines()
    return data


def save_sess_words_to_file(filepath, sessions, uttr_delimiter="#"):
    with io.open(filepath, 'w', encoding="utf-8") as f:
        newline = ""
        for sess in sessions:
            sess_line = ""
            newuttr = ""
            for uttr in sess:
                uttr=" ".join(uttr)
                sess_line+=newuttr+uttr
                newuttr = uttr_delimiter
            f.write(unicode(newline+sess_line))
            newline="\n"


def save_to_file(filepath, data):
    with io.open(filepath, 'w', encoding="utf-8") as f:
        newline = ""
        for d in data:
            f.write(unicode(newline+str(d)))
            newline="\n"


def save_sess_labels_to_file(filepath, data, uttr_delimiter="#"):
    with io.open(filepath, 'w', encoding="utf-8") as f:
        newline = ""
        for d in data:
            f.write(unicode(newline+uttr_delimiter.join(d)))
            newline="\n"
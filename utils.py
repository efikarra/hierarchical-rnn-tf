import io

import tensorflow as tf
import codecs
import os
import json
import cPickle
import numpy as np

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


def get_config_proto(log_device_placement=False, allow_soft_placement=True):
    config_proto = tf.ConfigProto(log_device_placement=log_device_placement,
                                  allow_soft_placement = allow_soft_placement)
    # allocate as much GPU memory as is needed, based on runtime allocations.
    config_proto.gpu_options.allow_growth = False
    return config_proto


def save_object(filepath, obj):
    with open(filepath,"wb") as f:
        cPickle.dump(obj,f)


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


def load_file_split_lines(filepath, delimiter=" "):
    with io.open(filepath, "r", encoding="utf-8") as f:
        lines=f.read().splitlines()
        for i,l in enumerate(lines):
            lines[i]=l.split(delimiter)
    return lines


def save_sess_uttrs_to_file(filepath, sessions, uttr_delimiter="#"):
    with io.open(filepath, 'w', encoding="utf-8") as f:
        newline = ""
        for sess in sessions:
            sess_line = ""
            newuttr = ""
            for uttr in sess:
                sess_line+=newuttr+uttr
                newuttr = uttr_delimiter
            f.write(unicode(newline+sess_line))
            newline="\n"


def save_sess_labels_to_file(filepath, data, uttr_delimiter="#"):
    with io.open(filepath, 'w', encoding="utf-8") as f:
        newline = ""
        for d in data:
            f.write(unicode(newline+uttr_delimiter.join(d)))
            newline="\n"


def save_to_file(filepath, data):
    with io.open(filepath, 'w', encoding="utf-8") as f:
        newline = ""
        for d in data:
            f.write(unicode(newline+str(d)))
            newline="\n"


def get_lab_arr(lablist, n_labels=None):
    if not n_labels:
        maxlab = max(lablist)
        n_labels = maxlab + 1
    labarr = np.zeros((len(lablist), n_labels))
    labarr[range(len(lablist)), lablist] = 1
    return labarr


def flatten_nested_labels(nested_labs, lab_idx=None):
    if lab_idx is None:
        return [nested_labs[i][j] for i in range(len(nested_labs)) for j in range(len(nested_labs[i]))]
    else:
        return [nested_labs[i][lab_idx][j] for i in range(len(nested_labs)) for j in range(len(nested_labs[i][lab_idx]))]


def get_nested_labels(flattened_labs, len_list):
    fr, to = 0, 0
    nested_labs = []
    for slen in len_list:
        to = fr + slen
        nested_labs.append(flattened_labs[fr:to])
        fr = to
    return nested_labs


def save_sq_mat_with_labels(mat, lid2shortname, filename):
    import csv
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([""] + lid2shortname)
        writer.writerows([[lid2shortname[i]]+row for i, row in enumerate(mat.tolist())])


def load_labs_from_text_sess(labs_file, lab_delimiter=" "):
    labs = []
    with open(labs_file, "rb") as f:
        for line in f.readlines():
            uttr_labs = line.split(lab_delimiter)
            uttr_labs = [int(lab) for lab in uttr_labs]
            labs.append(uttr_labs)
    return labs


def convert_to_one_hot(labs):
    rows = labs.shape[0]
    cols = np.max(labs)+1
    labs_arr = np.zeros((rows,cols))
    labs_arr[np.arange(rows), labs] = 1
    return labs_arr


def probs_to_labs(predictions):
    return np.argmax(predictions, axis=1)


def sess_probs_to_labs(predictions):
    labels = []
    for i,sess in enumerate(predictions):
        labels.append(np.argmax(predictions[i], axis=1))
    return labels


def fix_uttr_count_rule(labs, excl_labs):
    import copy
    new_labs = copy.deepcopy(labs)
    for i,sess_labs in enumerate(new_labs):
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
                for k in range(j+1-topic2_c,j+1):
                    new_labs[i][k] = topic1
    return new_labs


def convert_to_broader_topic(labs, lab2ltr, lid2lab, lt2ltid):
    broader_topic_labels = []
    for sess_labs in labs:
        broader_sess_labs = []
        for j, uttr_lab in enumerate(sess_labs):
            broader_sess_labs.append(lt2ltid[lab2ltr[lid2lab[uttr_lab]]])
        broader_topic_labels.append(broader_sess_labs)
    return broader_topic_labels
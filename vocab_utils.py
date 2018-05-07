import tensorflow as tf
import codecs
import os

UNK_ID=1
UNK="<unk>"
PAD="<pad>"

def load_vocab(vocab_file):
  vocab = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
    vocab_size = 0
    for word in f:
      vocab_size += 1
      vocab.append(word.strip())
  return vocab, vocab_size


def check_vocab(vocab_file, out_dir, unk=None, pad=None):
    """ Check if vocab has the unk and pad symbols as first words.
    If not, create a new vocab file with these symbols as the first two words."""
    if tf.gfile.Exists(vocab_file):
        print(" Vocab file %s exists "% vocab_file)
        vocab, vocab_size = load_vocab(vocab_file)

        if not unk: unk = UNK
        if not pad: pad = PAD
        assert len(vocab)>=2
        # Extend vocabulary to include unk and pad symbols.
        if vocab[0] != pad or vocab[1] != unk:
            print("The first 2 vocab words [%s, %s]"
                            " are not [%s, %s]." %
                            (vocab[0], vocab[1], pad, unk))
            vocab = [pad, unk] + vocab
            vocab_size += 2
            new_vocab_file = os.path.join(out_dir, os.path.basename(vocab_file))
            with codecs.getwriter("utf-8")(tf.gfile.GFile(new_vocab_file, "wb")) as f:
                newline = ''
                for word in vocab:
                    f.write(newline + "%s" % word)
                    newline = "\n"
            print("Vocabulary was extended to include [%s, %s] and the extended file was saved to %s." %
                  (pad, unk,new_vocab_file))
            vocab_file = new_vocab_file
    else:
        raise ValueError("vocab_file does not exist.")
    vocab_size = len(vocab)
    return vocab_size, vocab_file

def create_vocab_table(vocab_file):
    return tf.contrib.lookup.index_table_from_file(vocab_file, default_value=UNK_ID)

def create_inverse_vocab_table(vocab_file, unk=UNK):
    return tf.contrib.lookup.index_to_string_table_from_file(vocab_file, default_value=unk)
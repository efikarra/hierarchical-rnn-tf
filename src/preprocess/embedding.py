import numpy as np
import tensorflow as tf
import src.utils.vocab_utils

# Loading the global embeddings and using the vocabulary to create the embeddings, and finally saving it
def save_embedding(vocab, embed_path, emb_outpath):
    """Create and save an embedding matrix for the given vocabulary and pretrained embeddings.
        vocab: input vocabulary.
        embed_path: path to the pretrained embeddings.
        emb_outpath: file path to save the embedding matrix.
    """
    # Create new embeddings and save in a matrix
    embeddings_index = {}
    # c=0
    if tf.gfile.Exists(embed_path):
        with open(embed_path) as f:
            for j,line in enumerate(f):
                # if c>100000:break
                # c+=1
                values = line.split(' ')
                word = values[0]
                embedding = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = embedding
    else:
        raise ValueError("Embedding File does not exist.")

    print('Length of Word embeddings:', len(embeddings_index))
    # Calculating embedding dim
    if len(embeddings_index) > 0:
        embedding_dim = len(embeddings_index[list(embeddings_index.keys())[0]])
    else:
        print('Embeddings are empty!!')
        embedding_dim = 300

    nb_words = len(vocab)
    word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)
    for i, word in enumerate(vocab):
        if word in embeddings_index:
            word_embedding_matrix[i] = embeddings_index[word]
        else:
            # If word not in the trained embeddings, create a random embedding for it
            new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
            embeddings_index[word] = new_embedding
            word_embedding_matrix[i] = new_embedding

    np.savetxt(emb_outpath, word_embedding_matrix, fmt='%.18e')

if __name__=="__main__":
    vocab_path="experiments/data/0.00.1vocab.txt"
    embed_path = "experiments/embeddings/glove.840B.300d.txt"
    out_path = "experiments/embeddings/"
    # embed_path = "experiments/embeddings/glove.6B.50d.txt"
    # embed_path = "experiments/embeddings/glove.6B.100d.txt"
    # embed_path = "experiments/embeddings/glove.6B.200d.txt"
    # embed_path = "experiments/embeddings/glove.6B.300d.txt"
    vocab, _ = src.utils.vocab_utils.load_vocab(vocab_path)
    unk= u"<unk>"
    pad=u"<pad>"
    if vocab[0] != pad or vocab[1] != unk:
        vocab = [pad, unk] + vocab
    save_embedding(vocab, embed_path ,out_path+
                   embed_path.split("/")[-1].replace(".txt","_")+vocab_path.split("/")[-1])
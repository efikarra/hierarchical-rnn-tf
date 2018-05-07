import numpy as np
import tensorflow as tf

# Not being used currently
# For creating embeddings
def create_embeddings(vocab, embed_path, unknown, pad, start = None, end = None):
    # Loading all embeddings
    embeddings_index = {}
    with open(embed_path) as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding

    print('Length of Word embeddings:', len(embeddings_index))
    # Calculating embedding dim
    if len(embeddings_index) > 0:
        embedding_dim = len(embeddings_index[embeddings_index.keys()[0]])
    else:
        print('Embeddings are empty!!')
        embedding_dim = 300

    # Creating embedding for the vocabulary provided
    word_embedding = {}
    missing_words = 0
    for word in vocab:
        if word in embeddings_index:
            word_embedding[word] = embeddings_index[word]
        else:
            embeddings_index[word] = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
            word_embedding[word] = embeddings_index[word]
            missing_words += 1

    # Creating embeddings for unknown, pad, start and end
    word_embedding[unknown] = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
    word_embedding[pad] = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
    if start is not None:
        word_embedding[start] = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
    if end is not None:
        word_embedding[end] = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))

    print('Total vocabulary size:', len(vocab))
    print('Number of missing words:', missing_words)

    return word_embedding

# Not being used currently
# For creating sentence embeddings
def sentence_embedding(sentence_vocab, word_embedding, unknown, embed_type = 'mean'):
    sentence_embed = []
    for word in sentence_vocab:
        if word in word_embedding:
            sentence_embed.append(word_embedding[word])
        else:
            sentence_embed.append(word_embedding[unknown])

    if embed_type is 'mean':
        output_embed = np.mean(sentence_embed, 0)
    else:
        print('CAUTION!! No other implementation, returning mean embedding only!')
        output_embed = np.mean(sentence_embed, 0)

    return output_embed

# Loading the global embeddings and using the vocabulary to create the embeddings, and finally saving it
def save_embedding(vocab, embed_path, emb_file):
    # Create new embeddings and save in a matrix
    embeddings_index = {}
    if tf.gfile.Exists(embed_path):
        with open(embed_path) as f:
            for j,line in enumerate(f):
                values = line.split(' ')
                word = values[0]
                embedding = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = embedding
    else:
        raise ValueError("Embedding File does not exist.")

    print('Length of Word embeddings:', len(embeddings_index))
    # Calculating embedding dim
    embedding_dim = 0
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

    np.savetxt(emb_file, word_embedding_matrix)

if __name__=="__main__":
    import vocab_utils
    vocab_path="experiments/data/0.00.1vocab_uttr.txt"
    embed_path = "experiments/embeddings/glove.840B.300d.txt"
    vocab, _ = vocab_utils.load_vocab(vocab_path)
    unk= u"<unk>"
    pad=u"<pad>"
    if vocab[0] != pad or vocab[1] != unk:
        vocab = [pad, unk] + vocab
    save_embedding(vocab, embed_path , "experiments/embeddings/emb_"+vocab_path.split("/")[-1])
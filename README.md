This repository implements several hierarchical and non-hierarchical Deep Learning models in Tensorflow for text classification of documents.
It also supports pre-trained word or character embeddings as input to the models.

The module "model/model.py" contains a Feed Forward Neural Network (FFN), a Recurrent Neural Network (RNN) and a Convolutional Neural Network (CNN).
The model "model/hierarchical_model.py" contains the following hierarchical models:
 1. A RNN in the word-level with a RNN in the sentence-level.
 2. A CNN in the word-level with a RNN in the sentence-level.
 3. A FFN in the word-level with a RNN in the sentence-level.
 4. A RNN in the word-level, a RNN in the sentence-level and a Conditional Random Field (CRF) as a final layer.
 
 <h3> Required packages</h3>
 
 1. Tensorflow 1.8
 2. numpy
 3. scipy
 4. matplotlib

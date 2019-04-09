import numpy as np
import pickle
import gensim
from collections import namedtuple
from preprocess import preprocess_twitter

# Define dataset container
Dataset = namedtuple('Dataset','train_vectors, Y_train, val_vectors, Y_val, test_vectors, Y_test')


def get_model(path=None):

    if path is None:
        path = "../word2vec/GoogleNews-vectors-negative300.bin.gz"

    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

    print("Model loaded.")

    return model


def generate_twitter_embeddings(model, save=True):
    """Generate word embeddings for the Twitter sentiment dataset. Saves to file by default."""

    # # Define dataset container
    # Dataset = namedtuple('Dataset','train_vectors, Y_train, val_vectors, Y_val, test_vectors, Y_test')

    # Load dataset
    X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocess_twitter()

    print("Twitter dataset loaded")

    # Get word embeddings. Delete variables between calls to free up memory
    train_vectors = embed_set(model, X_train)
    del X_train
    val_vectors = embed_set(model, X_val)
    del X_val
    test_vectors = embed_set(model, X_test)
    del X_test

    print("Vectors generated")

    embedded_data = Dataset(train_vectors, Y_train, val_vectors, Y_val, test_vectors, Y_test)

    if save:
        file_path = "../data/twitter_word2vec.pickle"
        pickle.dump(embedded_data, open(file_path, 'wb'))
        print("Word embeddings saved to: ", file_path, "\n Use pickle.load() to retrieve it.")

    return embedded_data




def embed_set(model, data):
    """Generate a list of word embeddings for each sentence in the dataset"""

    vectors = [None]*data.shape[0]

    for i in range(data.shape[0]):
        # Get sentence and initialize list of word embeddings
        sent = data[i]
        word_embeddings = []

        for w in sent.split():
            # If word in vocabulary, add embedding to list
            if w in model.vocab:
                word_embeddings.append(model.get_vector(w))

        # Save in vector container
        vectors[i] = word_embeddings

    return vectors
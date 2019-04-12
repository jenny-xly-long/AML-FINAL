import pandas as pd
import numpy as np
import re

def preprocess_twitter(preprocess = True):

    # loads the training set
    X_train = pd.read_csv('../data/X_twitter_train.csv', delimiter = "\n", header = None, encoding = 'latin-1').values.flatten()
    y_train = np.genfromtxt('../data/y_twitter_train.csv', delimiter = "\t", dtype = None, encoding = 'latin-1').astype(int)

    # Loads the test set
    X_test = pd.read_csv('../data/X_twitter_test.csv', delimiter = "\n", header = None, encoding = 'latin-1').values.flatten()
    y_test = np.genfromtxt('../data/y_twitter_test.csv', delimiter = "\t", dtype = None, encoding = 'latin-1').astype(int)

    if preprocess:
        # Replace usernames with @USERNAME
        username = re.compile(r'@([A-Za-z0-9_]+)')
        X_train = [username.sub("@USERNAME", text) for text in X_train]
        X_test = [username.sub("@USERNAME", text) for text in X_test]
        # Replace urls with URL
        url = re.compile(r'http\S+')
        X_train = [url.sub("URL.", text) for text in X_train]
        X_test = [url.sub("URL.", text) for text in X_test]

        # Replace repeated characters
        X_train = [re.sub(r'(\w)\1+',r'\1\1', text) for text in X_train]
        X_test = [re.sub(r'(\w)\1+', r'\1\1', text) for text in X_test]

        # converts back to numpy arrays
        X_train = pd.DataFrame(X_train).values.flatten()
        X_test = pd.DataFrame(X_test).values.flatten()

    return (X_train, y_train, X_test, y_test)

def preprocess_treebank():
    X_train = pd.read_csv("../data/X_treebank_train.txt", delimiter = "\n", header = None)
    y_train = np.genfromtxt("../data/y_treebank_train.txt", delimiter = "\t", dtype=None, encoding = 'ascii')

    X_train_phrases = pd.read_csv("../data/X_treebank_train_phrases.txt", delimiter = "\n", header = None)
    y_train_phrases = np.genfromtxt("../data/y_treebank_train_phrases.txt", delimiter = "\t", dtype=None, encoding = 'ascii')

    X_train = pd.concat([X_train, X_train_phrases], ignore_index=True).values.flatten()
    y_train = np.concatenate([y_train, y_train_phrases])
    y_train[(y_train != '0') & (y_train != '1')] = 1
    y_train = y_train.astype(int)

    X_test = pd.read_csv("../data/X_treebank_test.txt", delimiter = "\n", header = None).values.flatten()
    y_test = np.genfromtxt("../data/y_treebank_test.txt", delimiter = "\t", dtype=None, encoding = 'ascii')

    return (X_train, y_train, X_test, y_test)

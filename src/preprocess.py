import pandas as pd
import re


def preprocess_twitter():
    # loads the training set
    data = pd.read_csv("../data/training_twitter.csv", encoding="latin-1", header=None)
    train = data.sample(n=80000)
    val = data.sample(n=16000)

    X_train = train.iloc[:, -1].values
    y_train = train.iloc[:, 0].values
    y_train[y_train == 4] = 1

    X_val = val.iloc[:, -1].values
    y_val = val.iloc[:, 0].values
    y_val[y_val == 4] = 1

    # Loads the test set
    data = pd.read_csv("../data/test_twitter.csv", encoding="latin-1", header=None)
    X_test = data.iloc[:, 5].values
    y_test = data.iloc[:, 0].values
    y_test[y_test == 4] = 1

    # Replace usernames with @USERNAME
    username = re.compile(r'@([A-Za-z0-9_]+)')
    X_train = [username.sub("@USERNAME", text) for text in X_train]
    X_val = [username.sub("@USERNAME", text) for text in X_val]
    X_test = [username.sub("@USERNAME", text) for text in X_test]
    # Replace urls with URL
    url = re.compile(r'http\S+')
    X_train = [url.sub("URL.", text) for text in X_train]
    X_val = [url.sub("URL.", text) for text in X_val]
    X_test = [url.sub("URL.", text) for text in X_test]

    # Replace repeated characters
    X_train = [re.sub(r'(\w)\1+', r'\1\1', text) for text in X_train]
    X_val = [re.sub(r'(\w)\1+', r'\1\1', text) for text in X_val]
    X_test = [re.sub(r'(\w)\1+', r'\1\1', text) for text in X_test]

    return (X_train, y_train, X_val, y_val, X_test, y_test)

def preprocess_treebank():
    X_train = pd.read_csv("./data/X_treebank_train.txt", delimiter = "\n", header = None).values.flatten()
    y_train = pd.read_csv("./data/y_treebank_train.txt", delimiter = "\n", header = None).values.flatten()

    X_train_phrases = pd.read_csv("./data/X_treebank_train_phrases.txt", delimiter = "\n", header = None).values.flatten()
    y_train_phrases = pd.read_csv("./data/y_treebank_train_phrases.txt", delimiter = "\n", header = None).values.flatten()

    X_val = pd.read_csv("./data/X_treebank_val.txt", delimiter = "\n", header = None).values.flatten()
    y_val = pd.read_csv("./data/y_treebank_val.txt", delimiter = "\n", header = None).values.flatten()

    X_test = pd.read_csv("./data/X_treebank_test.txt", delimiter = "\n", header = None).values.flatten()
    y_test = pd.read_csv("./data/y_treebank_test.txt", delimiter = "\n", header = None).values.flatten()

    return (X_train, y_train, X_train_phrases, y_train_phrases, X_val, y_val, X_test, y_test)

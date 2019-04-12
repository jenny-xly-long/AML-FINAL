import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def features_extract(X_train, X_test, type = "binary", ngram_range = (1,2)):

    if type == "binary":
        counter = CountVectorizer(max_features = 5000, ngram_range = ngram_range)
        X_train = counter.fit_transform(X_train)
        X_test = counter.transform(X_test)
        return (X_train, X_test)

    elif type == "tf-idf":
        tf_idf = TfidfVectorizer(max_features = 5000, ngram_range = ngram_range)
        X_train = tf_idf.fit_transform(X_train)
        X_test = tf_idf.transform(X_test)
        return (X_train, X_test)

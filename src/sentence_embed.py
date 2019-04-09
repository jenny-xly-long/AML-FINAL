import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from tqdm import tqdm

def sentence_embed(X, method = "mean"):

    # Load spacy model
    nlp = spacy.load('en_core_web_lg')

    # initializes data structure that holds the embeddings
    N = X.shape[0]
    nlp_objects = list()
    nlp_vectors = np.zeros((N, 300))

    # converts the words of the sentences to their embeddings
    for i in tqdm(range(N)):
        nlp_objects.append(nlp(X[i]))

    if method == "mean":
        for i in range(N):
            nlp_vectors[i,:] = nlp_objects[i].vector

    elif method == "tf-idf":

        # computes the tfidf score of each word appearing in the data
        tfidf_vec = TfidfVectorizer()
        X = tfidf_vec.fit_transform(X)
        index_value={i[1]:i[0] for i in tfidf_vec.vocabulary_.items()}
        tfidf_scores = {index_value[column]:value for (column,value) in zip(X.indices,X.data)}

        # computes the weighted sum of the embedded vectors to yield the sentences embedding
        for i in range(N):
            # get the tokens of the current sentence
            tokens = [token.text for token in nlp_objects[i]]
            # get the vectors of the current sentence
            vectors = np.vstack([token.vector for token in nlp_objects[i]])
            # get the weights of the words o the sentence
            weights = np.array([tfidf_scores[token.lower()] if token.lower() in tfidf_scores else 0 for token in tokens])
            # computes the vector embeding of the sentence
            if np.any(weights):
                nlp_vectors[i,:] = np.average(vectors, axis = 0, weights=weights) #normalizes by sum of weights
    else:
        print("Unknown method")

    return nlp_vectors

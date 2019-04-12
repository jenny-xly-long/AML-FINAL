import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from Bernoulli_NBSVM import Bernoulli_NBSVM
from feedforward_NN import feedforward_NN
import numpy as np

class Classifier:

    def __init__(self, X, y, model = "log_reg", D_in=200):
        self.X = X
        self.y = y

        if model == "log_reg":
            self.model = LogisticRegression(solver = "liblinear")
        elif model == "Linear_SVM":
            self.model = LinearSVC(dual = False, max_iter = 10000)
        elif model == "Multinomial_NB":
            self.model = MultinomialNB()
        elif model == "Gaussian_NB":
            self.model = GaussianNB()
        elif model == "Bernoulli_NBSVM":
            self.model = Bernoulli_NBSVM()
        elif model == "feedforward_NN":
            self.model = feedforward_NN(D_in)

    def k_fold_cross_val(self, k=3):
        print ("  > " + str(k) + "Fold evaluating...")
        scores = cross_val_score(self.model, self.X, self.y, cv=k)
        score = scores.mean()
        print("    " + str(k) + "-fold cross validation accuracy: " + str(score))
        return score

    def fit(self):
        print ("  > Training...")
        self.model.fit(self.X, self.y)

    def score(self, X, y):
        print("  > Testing...")
        acc = self.model.score(X, y)
        print("    Test accuracy: " + str(acc))

    def grid_search(self, hyperparams, k=3):
        print("  > Evaluating...")
        gs = GridSearchCV(self.model, hyperparams, cv=k)
        gs.fit(self.X, self.y)
        print ("    Best parameters: " + str(gs.best_params_) + " with score: " + str(gs.best_score_))
        self.model = gs.best_estimator_

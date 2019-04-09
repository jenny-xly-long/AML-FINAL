import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectPercentile, chi2,mutual_info_classif,f_classif
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Multinomial NB
pclf = Pipeline([
    ('vect', CountVectorizer(binary=True, stop_words=None, ngram_range=(1,2))),
    ('norm', Normalizer()),
    ('clf', MultinomialNB()),
])

pclf.fit(X_train, y_train)
y_pred = pclf.predict(X_val)
print(metrics.classification_report(y_val, y_pred))


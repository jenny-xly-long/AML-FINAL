from classifiers import Classifier

def results(X_train, y_train, X_test, y_test, features = "binary", D_in=200):

    print("\n  > Logistic Regression: ")
    # performs logistic regression
    log_reg = Classifier(X_train, y_train, model = "log_reg")
    # determines the parameters used in the grid search
    hyperparams = {'C': [0.01, 1, 100], 'penalty': ['l1', 'l2']}
    # picks the best possible model using grid search
    log_reg.grid_search(hyperparams)
    # fully train the best model
    log_reg.fit()
    # tests the accuracy of the model
    log_reg.score(X_test, y_test)

    print("\n  > Linear SVM: ")
    # performs SVM
    Linear_SVM = Classifier(X_train, y_train, model = "Linear_SVM")
    # determines the parameters used in the grid search
    hyperparams = {'C': [0.01, 1, 100]}
    # picks the best possible model using grid search
    Linear_SVM.grid_search(hyperparams)
    # fully train the best model
    Linear_SVM.fit()
    # tests the accuracy of the model
    Linear_SVM.score(X_test, y_test)

    if features == "binary":
        print("\n  > Bernoulli Naive Bayes SVM: ")
        # performs Gaussian Naive Bayes
        Bernoulli_NBSVM = Classifier(X_train, y_train, model = "Bernoulli_NBSVM")
        # determines the parameters used in the grid search
        hyperparams = {'C': [0.01, 1, 100], 'beta': [0.25, 0.5, 0.75]}
        # picks the best possible model using grid search
        Bernoulli_NBSVM.grid_search(hyperparams)
        # fully train the best model
        Bernoulli_NBSVM.fit()
        # tests the accuracy of the model
        Bernoulli_NBSVM.score(X_test, y_test)

    if features == "sentence_embed":
        print("\n  > Feedforward NN:")
        # performs feeforward NN
        feedforward_NN = Classifier(X_train, y_train, "feedforward_NN", D_in)
        # determines the parameters used in the grid search
        #hyperparams = {'batch_size' : [128, 256, 512], 'epochs' : [10, 20, 50]}
        # picks the best possible model using grid search
        #feedforward_NN.grid_search(hyperparams)
        # fully train the best model
        feedforward_NN.fit()
        # tests the accuracy of the model
        feedforward_NN.score(X_test, y_test)

        print("\n  > Gaussian Naive Bayes: ")
        # performs Gaussian Naive Bayes
        Gaussian_NB = Classifier(X_train, y_train, model = "Gaussian_NB")
        # determines the parameters used in the grid search
        hyperparams = {'priors': [None, (0.25,0.75), (0.5, 0.5), (0.75, 0.25)]}
        # picks the best possible model using grid search
        Gaussian_NB.grid_search(hyperparams)
        # fully train the best model
        Gaussian_NB.fit()
        # tests the accuracy of the model
        Gaussian_NB.score(X_test, y_test)

        return (log_reg, Linear_SVM, Gaussian_NB)

    else:
        print("\n  > Multinomial Naive Bayes: ")
        # performs Gaussian Naive Bayes
        Multinomial_NB = Classifier(X_train, y_train, model = "Multinomial_NB")
        # determines the parameters used in the grid search
        hyperparams = {'class_prior': [None, (0.25,0.75), (0.5, 0.5), (0.75, 0.25)]}
        # picks the best possible model using grid search
        Multinomial_NB.grid_search(hyperparams)
        # fully train the best model
        Multinomial_NB.fit()
        # tests the accuracy of the model
        Multinomial_NB.score(X_test, y_test)

        return (log_reg, Linear_SVM, Multinomial_NB)

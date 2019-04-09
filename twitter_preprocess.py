import pandas as pd

def preprocess_twitter():

    # loads the training set
    data = pd.read_csv("./twitter/train.csv", encoding = "latin-1", header = None)
    train = data.sample(n=80000)
    val = data.sample(n=16000)

    X_train = train.iloc[:,-1].values
    y_train = train.iloc[:,0].values
    y_train[y_train == 4] = 1

    X_val = val.iloc[:,-1].values
    y_val = val.iloc[:,0].values
    y_val[y_val == 4] = 1

    # Loads the test set
    data = pd.read_csv("./twitter/test.csv", encoding = "latin-1", header = None)
    X_test = data.iloc[:,5].values
    y_test = data.iloc[:,0].values
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
    X_train = [re.sub(r'(\w)\1+',r'\1\1', text) for text in X_train]
    X_val = [re.sub(r'(\w)\1+', r'\1\1', text) for text in X_val]
    X_test = [re.sub(r'(\w)\1+', r'\1\1', text) for text in X_test]

    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)
    X_test = np.asarray(X_test)
    
    return (X_train, y_train, X_val, y_val, X_test, y_test)
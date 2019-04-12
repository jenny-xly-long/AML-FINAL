from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

def feedforward_NN(D_in=200):

    def model_fn(D_in):
    	# create model
    	model = Sequential()
    	model.add(Dense(units=64, input_dim=D_in, activation='sigmoid'))
    	model.add(Dense(units=32, activation='sigmoid'))
    	model.add(Dense(units=1, activation='sigmoid'))
    	# Compile model
    	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    	return model

    return KerasClassifier(build_fn=model_fn, epochs=10, batch_size=128, verbose=0)

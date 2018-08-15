import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from keras.model import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import kFold
from sklearn.preprocessing import StanddardScaler
from sklearn.pipeline import Pipeline
## Perfrom GridSearch on pickled training data

values = pickle.load(open("values.pkl", "rb"))
y = pickle.load(open("y.pkl", "rb"))

def search_mlregressor(values, y):
	params = {
	    'alpha': range(0, 2),
	    'beta_1': [i/10.0 for i in range(0, 10)],
	    'hidden_layer_sizes': range(50, 400, 50),
	    'activation': ['identity', 'logistic', 'tanh', 'relu'],
	    'solver': ['lbfgs', 'sgd','adam']
	}
	clf = GridSearchCV(MLPRegressor(), params, cv=5)
	clf.fit(values, y)
	clf.cv_results_['mean_test_score']

	print clf.best_score_
	print clf.best_params_

def search_keras_regressor(values, y):

	def train_model(train_x, train_y, batch_size, test_x=None, test_y=None):
	    model = Sequential()
	    model.add(LSTM(50, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
	    model.add(Dropout(0.2))
	    model.add(Dense(200))
	    model.add(Dropout(0.2))
	    model.add(Dense(1))
	    model.compile(loss='mean_squared_error', optimizer='adam')
		
	n_shift = 0
	batch_size = 10


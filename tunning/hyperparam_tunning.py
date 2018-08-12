import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

values = pickle.load(open("values.pkl", "rb"))
y = pickle.load(open("y.pkl", "rb"))

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
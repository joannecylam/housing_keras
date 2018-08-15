import pandas as pd
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from preprocessing import PrepareData
from keras.layers import TimeDistributed
from keras.callbacks import ReduceLROnPlateau
import datetime
from keras.callbacks import TensorBoard
import random

def shuffle(X,Y):
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]

def splitData(X,Y,rate):
    X_train = X[int(X.shape[0]*rate):]
    Y_train = Y[int(Y.shape[0]*rate):]
    X_val = X[:int(X.shape[0]*rate)]
    Y_val = Y[:int(Y.shape[0]*rate)]
    return X_train, Y_train, X_val, Y_val

def buildTrain(train, timesteps=5):
    X_train, Y_train = [], []
    sales = train['SalePrice']
    train.drop("SalePrice", axis=1, inplace=True)
    for i in range(train.shape[0]-timesteps+1):
        X_train.append(np.array(train.iloc[i:i+timesteps]))
        Y_train.append(np.array(sales.iloc[i+timesteps-1: i+timesteps]))
    return np.array(X_train), np.array(Y_train)

def buildTest(test, timesteps=5):
    X_test = []
    for i in range(test.shape[0]-timesteps+1):
        X_test.append(np.array(test.iloc[i:i+timesteps]))
    return np.array(X_test)

def buildModel(shape):
    model = Sequential()
    model.add(LSTM(5, input_shape=(shape[1], shape[2]), return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(200))
    model.add(Dropout(0.5))
    model.add(Dense(200))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="tanh"))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.summary()
    return model

def train_model(X_train, Y_train, X_val=None, Y_val=None):
    batch_size = 5
    model = buildModel(X_train.shape)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
    early_stop = EarlyStopping(monitor="val_loss", patience=20, verbose=1, mode="auto")
    tf_board = TensorBoard(log_dir='./logs/{}'.format(datetime.datetime.now().strftime('%H_%M')), histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)
    if X_val is None or Y_val is None:
        kwargs = {}
    else:
        kwargs = {
            "validation_data": (X_val, Y_val)
        }
    history = model.fit(X_train, Y_train, epochs=200, shuffle=True, batch_size=batch_size, callbacks=[tf_board, reduce_lr, early_stop], **kwargs)
    return model, history

def convert_to_price(sales, predict_results):
    max_s = sales.SalePrice.max()
    min_s = sales.SalePrice.min()
    predicted_p = predict_results*(max_s - min_s) + min_s
    return predicted_p

if __name__ == "__main__"
    timesteps = 5
    ppd = PrepareData()
    df, sale_price = ppd.get_train_data(time_series=True)
    X_train_all, Y_train_all = buildTrain(df, timesteps=timesteps)
    X_train, Y_train = shuffle(X_train_all, Y_train_all)
    X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)


    model, history = train_model(X_train, Y_train, X_val, Y_val)
    pred_train = model.predict(X_train)
    print 'RMSE (Training Data):', mean_squared_error(Y_train , pred_train)
    pred_test = model.predict(X_val)
    print 'RMSE (Testin Data):', mean_squared_error(Y_val , pred_test)

    df_padding = df.sort_index().head(timesteps-1)
    test_df, test_ids = ppd.get_test_data(time_series=True)
    test_df_padded = pd.concat([df_padding, test_df], ignore_index=True)
    
    model2, history2 = train_model(X_train_all, Y_train_all)
    
    X = buildTest(test_df_padded, timesteps=5)
    results = model2.predict(X)
    test_ids['SalePrice'] = convert_to_price(sale_price, results)
    test_results = test_ids.sort_values('Id')
    test_results.to_csv("submission_{}.csv".format(datetime.datetime.now().strftime("%H_%M")), index=False)


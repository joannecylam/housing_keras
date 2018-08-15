import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from preprocessing import PrepareData
from keras.callbacks import ReduceLROnPlateau
from preprocessing import PrepareData
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import datetime
from keras.callbacks import TensorBoard


def convert_to_price(sales, predict_results):
    max_s = sales.SalePrice.max()
    min_s = sales.SalePrice.min()
    predicted_p = predict_results*(max_s - min_s) + min_s
    return predicted_p

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    dff = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(dff.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(dff.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def split_data(x, y, ratio):
    n = int(ratio*x.shape[0])
    return (x[:n, :], x[n:, :], y[:n], y[n:])


def normalize_price(sale_price):
    # return normalized price
    width = sale_price.max() - sale_price.min()
    return (sale_price - sale_price.min())/width


def train_model(train_x, train_y, batch_size, test_x=None, test_y=None):
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(200))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
    tf_board = TensorBoard(log_dir='./logs/log_{}'.format(datetime.datetime.now().strftime("%H%M")), histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)
    if not test_x is None and not test_y is None:
        kwargs = {
            "validation_data": (test_x, test_y),
        }
    else:
        kwargs = {}
    history = model.fit(train_x, train_y, epochs=80, batch_size=batch_size, verbose=2, shuffle=True, callbacks=[reduce_lr, tf_board], **kwargs)
    return model, history

if __name__ == "__main__":
    ppd = PrepareData()
    df , sale_price = ppd.get_train_data(time_series=True, drop_cols = ['Id'] + ppd.sales_attrs)
    df.drop("SalePrice", axis=1, inplace=True)

    batch_size = 10
    n_shift = 0
    reframed = series_to_supervised(df.values, n_shift)
    values = reframed.values

    # shift the sale price by the number of days shifted
    sale_price_shifted = normalize_price(sale_price[n_shift:])

    train_x, test_x, train_y, test_y = split_data(values, sale_price_shifted, 0.8)
    # reshape to [samples, timesteps, features]
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

    model, history = train_model(train_x, train_y, batch_size, test_x, test_y)
    # make a prediction
    pred_train = model.predict(train_x)
    print 'RMSE (Training Data):', mean_squared_error(train_y , pred_train)
    pred_test = model.predict(test_x)
    print 'RMSE (Testin Data):', mean_squared_error(test_y , pred_test)

    with open('results.txt', 'a') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write('RMSE (Training Data):' + str(mean_squared_error(train_y , pred_train)) + "\n")
        f.write('RMSE (Testin Data):' + str(mean_squared_error(test_y , pred_test))+ "\n\n" )


    test_df, test_ids = ppd.get_test_data(time_series=True)
    test_data = test_df.values.reshape(test_df.shape[0], 1, test_df.shape[1])

    # append shifted data
    test_df_padded = pd.concat([df.head(n_shift), test_df], ignore_index=True)
    test_reframed = series_to_supervised(test_df_padded.values, n_shift)
    test_reframed_data = test_reframed.values.reshape(test_reframed.shape[0], 1, test_reframed.shape[1])

    # train model again
    all_train_data = values.reshape(values.shape[0], 1, values.shape[1])

    model2, history2 = train_model(all_train_data, sale_price_shifted, batch_size)

    predict_results = model.predict(test_reframed_data)
    price = convert_to_price(sale_price, predict_results)
    test_ids['SalePrice'] = price
    test_ids.to_csv("submissions/submission_lstm_{}.csv".format(datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")), index=False)


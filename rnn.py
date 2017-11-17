# Part 1. data preprocessing
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
# Importing the training set
dataset_train = pd.read_csv('/work/rnn-dataset/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values
training_set

#%%
# Feture scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

#%%
# Creating the data structure
X_train, y_train = [], [] 
for i in range(60, 1250):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


#%%
def rnn():
    """ Build network here
    """
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout

    # Initialising the RNN
    regressor = Sequential()

    # Adding LSTM layers
    layers = [
        {'layer': LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)), 'dropout': Dropout(0.2)},
        {'layer': LSTM(units=50, return_sequences=True), 'dropout': Dropout(0.2)},
        {'layer': LSTM(units=50, return_sequences=True), 'dropout': Dropout(0.2)},
        {'layer': LSTM(units=50), 'dropout': Dropout(0.2)},
        {'layer': Dense(units=1)}
        ]
    for layer in layers:
        regressor.add(layer['layer'])
        if 'dropout' in layer:
            regressor.add(layer['dropout'])

    regressor.compile(optimizer='adam', loss='mean_squared_error')
    regressor.fit(X_train, y_train, epochs=2, batch_size=32)
    return regressor

regressor = rnn()
#%% Part 3 - Making the prediction
dataset_test = pd.read_csv('/work/rnn/dataset/Google_Stock_Price_Test.csv')
dataset_test.iloc[:, 1:2].shape
real_stock_price = dataset_test.iloc[:, 1:2]
real_stock_price

#%%
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(X_test)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#%% Visualizing the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted')
plt.tilte('Google Stock Price Prediction')
plt.xlabel('Time')
plt.label('Google Stock Price')
plt.legend()
plt.show()

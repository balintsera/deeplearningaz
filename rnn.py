# Part 1. data preprocessing
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
# Importing the training set
dataset_train = pd.read_csv('/work/notebooks/rnn-dataset/Google_Stock_Price_Train.csv')
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

    # Adding the first LSTM layer
    regressor.add(LSTM(units=50, return_sequences=True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))

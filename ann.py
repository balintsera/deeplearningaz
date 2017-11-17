
# coding: utf-8

# ## Data processing for classification problem
# binary outcome of a banking problem: will the customer leave or stay

# In[1]:

import numpy as np
import pandas
import matplotlib.pyplot as plt
dataset = pandas.read_csv('/work/notebooks/dpt/Churn_Modelling.csv')


# In[2]:


dataset.head()


# these might have impact on independent variable: creditScore, Geography, Gender, Age, tenure, balance, numberOfProduct, HasCreditCard, EstimatedSalary
# 
# indexes: from 3 to 12

# In[26]:


X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
y[0]


# In[27]:


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
X[0]


# In[28]:


labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
X[0]


# In[29]:


# Dummy variable
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]
X


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[8]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

X_train.shape
X_train[1]


# In[9]:


import keras


# In[10]:

from keras.models import Sequential
from keras.layers import Dense



# ## kfold cross validation

# In[40]:


from keras.wrappers.scikit_learn import KerasClassifier


# In[41]:


from sklearn.model_selection import cross_val_score


# In[ ]:


def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6, input_shape=(11,), kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(6, activation = 'relu', kernel_initializer='uniform'))
    classifier.add(Dense(1, activation = 'sigmoid', kernel_initializer='uniform'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=10) # 100 epochs originally
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv=10, n_jobs = -1) 


#%%
from sklearn.model_selection import GridSearchCV
def build_classifier_params(optimizer):
    classifier = Sequential()
    classifier.add(Dense(6, input_shape=(11,), kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(6, activation = 'relu', kernel_initializer='uniform'))
    classifier.add(Dense(1, activation = 'sigmoid', kernel_initializer='uniform'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier_params) 
parameters = { 'batch_size': [25, 32], 
                'nb_epoch': [1, 5], 
                'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


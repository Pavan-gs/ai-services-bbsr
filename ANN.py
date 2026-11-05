
'''
conda create -n name_of_venv python=3.8
conda env list
conda activate name_of_venv
conda install tensorflow / pip install tensorflow
conda install keras/pip install keras
conda install spyder

pip install pandas
pip install matplotlib
pip install seaborn
pip install scikit-learn'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import History


# output = activation_function( w1*x1 + w2*x2 + ... + wn*xn + bias )

# Weights
# Bias
# Input Layers
# hidden Layers --> Neurons
# Forward Propagation
# Activation functions  : Sigmoid, RELU, SOFTMAX
# Loss fn/ cost fn
# Optimizers   : GD, SGD, Momentum, ADAGRAD, ADADELTA, ADAM, RMS Prop
# Back-propagation
# Weights adjustments (chain rule of differentiation)
 

# Loss functions : 
# MSE : sum(yi-e(y))2
# Binary Cross-entropy : -y*(log(y^)-(1-y)*log(1-y^))
# Multi-class cross entropy : -yij*log(yij^)
# Activation functions
# RELU : max(o,x)

from matplotlib import pyplot as plt
def relu_ex(x):
    return max(0.0,x)

inp = [i for i in range(-10,10)]
op = [relu_ex(i) for i in inp]
plt.plot(inp,op)
plt.show()

# Sigmoid : 1/1+exp(-z)

from math import exp

def sigmoid_ex(x):
    return 1.0/(1.0+exp(-x))
op_sig = [sigmoid_ex(i) for i in inp]
plt.plot(inp,op_sig)
plt.show()

# Tanh, Leaky Relu, Parameterised relu, exponential relu, 

# Softmax

def softmax_ex(x):
    y = np.exp(x) 
    y_ = y/y.sum()
    return y_

softmax_ex([0.8,1.2,3.1])
    
# Activation fns for hidden layers : RELU

# Activation fns for output layers

# 1. Binary classifiers : Sigmoid
# 2. Multi-class : Softmax
# 3. Regression  : RELU

# Optimizers :

# Gradient Descent : All the dataset is fed at a time
# Stochastic Gradient Descent : Each record is fed at a time
# Mini batch GD : Pass data by batches of records
# Momentum : Exponential weight average (taking previous learning rate by time steps into consideration)
# Adaptive learning rates : Taking learning rate into consideration
# ADAM : Combination Adaptive learning rates and momentum
# RMS Prop : Root mean squared error

#df = pd.read_csv("E://data/churn.csv")
df = pd.read_csv(("E://data/Boston.csv"))

x = df.iloc[:,1:14]
y = df['medv']
'''
x1 = pd.get_dummies(x,columns=['Geography','Gender'], drop_first=True)
x1 = x1.values'''

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state = 2)

y_train = y_train.values
y_test = y_test.values

# Here we scale the data for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the ANN model
model = Sequential()

model.add(Dense(6,activation='relu',kernel_initializer='uniform',input_shape=(X_train_scaled.shape[1],)))# First hidden layer
model.add(Droput=0.1)
model.add(Dense(6, activation='relu',kernel_initializer = 'uniform'))  # Second hidden layer
model.add(Dense(1, activation='relu', kernel_initializer = 'uniform'))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# Train the model and capture the training history
history = model.fit(X_train_scaled, y_train, epochs=500, batch_size=32, validation_split=0.2)

# Make predictions on the test set
#predictions = model.predict(X_test_scaled)
pred = model.predict(X_test_scaled)
#predictions = (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions
from sklearn.metrics import mean_squared_error
err = np.sqrt(mean_squared_error(y_test, pred))
pd.crosstab(y_test, predictions.reshape(1,-1))
X_test_scaled.shape

# Plot the training and validation loss
plt.figure(figsize=(12, 5))

# Plot for Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot for Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['mse'], label='Training precision')
plt.plot(history.history['val_mse'], label='Validation precision')
plt.title('error over Epochs')
plt.xlabel('Epochs')
plt.ylabel('precision')
plt.legend()

plt.tight_layout()
plt.show()

# Visualize the weights of the first hidden layer
weights, biases = model.layers[0].get_weights()
plt.figure(figsize=(10, 6))
plt.imshow(weights, aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('Weights of First Hidden Layer')
plt.xlabel('Neurons')
plt.ylabel('Input Features')
plt.show()

print("Model training complete and results visualized.")


# Tuning

df = pd.read_csv('cust_churning.csv')

dataset.describe()

dataset['Location'].value_counts()
sns.countplot(dataset['Left'])
dataset['Left'].value_counts()
pd.crosstab(dataset['Left'],dataset['Location'])

sns.countplot(dataset['Location'], hue=dataset['Left'])
sns.countplot(dataset['Gender'], hue=dataset['Left'])

dataset.isna().sum()
dataset.info()

df.info()
df.describe().transpose()
df.columns

sns.boxplot(x="Exited", y = 'CreditScore', data=df, hue="Gender")
sns.countplot(x="Exited", data=df)

sns.countplot(x="Exited", data=df, hue='Gender')
sns.countplot(x="Gender", data=df)
sns.countplot(x="Geography", data=df)

'''
X = dataset.iloc[:, 1:11].values
y = dataset['Left'].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
'''

x = df.iloc[:,1:11]
y = df['Left']


d = pd.get_dummies(x[['Gender','Location']], drop_first=True)

x.drop(['Gender','Location'], inplace = True, axis = 1)

x = pd.concat([d,x], axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import Dropout

model= Sequential()

model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
# model.add(Dropout(p = 0.1))

# Adding the second hidden layer
model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
# model.add(Dropout(p = 0.1))

# Adding the output layer
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size = 10, epochs = 10)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

y_test.head(5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
(1551+136)/2000

#Evaluating a ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def fn():
    model= Sequential()
    model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

model = KerasClassifier(build_fn= fn,  batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = model, X=X_train, y=y_train,cv = 10, n_jobs=-1)
mean = accuracies.mean()
variance = accuracies.std()
help(cross_val_score)

# Improving the ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

# Dropout regularisation technique


def fn(optimizer):
    model = Sequential()
    model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

model = KerasClassifier(build_fn = fn)
parameters = {"batch_size" : [25,35],
              "epochs" : [100,200],
              "optimizer" : ["adam","rmsprop"]}

grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = "accuracy", cv = 5)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
help(GridSearchCV)

















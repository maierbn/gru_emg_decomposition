import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow import keras
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from scipy.stats import zscore
from sklearn.utils import class_weight
from tensorflow.python.keras.callbacks import TensorBoard
import datetime
print(tensorflow.__version__)

tensorboard_callback = TensorBoard(log_dir='logs', histogram_freq=1)

# Read CSV using Pandas Framework and covert data to pandas DataFrame
data = pd.read_csv(r"C:\Users\SRIJAY\Desktop\Research Project\Data SET\20mus-50s\20mus-50s\electrodes.csv",
                   delimiter=',', header=0, na_values=['NA'], encoding='ISO-8859-9')
data = pd.DataFrame(data)

# Data Cleaning
time_data = data['t']
print(time_data)
electrode_data = data.drop(data.iloc[:, :2], axis=1)

# for adding gaussian noise
electrode_data = electrode_data + np.random.normal(loc=0.0, scale=0.1, size=electrode_data.shape)
print(electrode_data.shape)


# Function for Normalization and Standarization

def standardize(dataset):
    data_standard = dataset.apply(zscore)
    return data_standard


Standardize_data = standardize(electrode_data)

# Import Stimulated.log (Labels)
log_file = open(r'C:\Users\SRIJAY\Desktop\Research Project\Data SET\20mus-50s\20mus-50s\stimulation.log')
lines = log_file.read().splitlines()
lines = lines[1:]
log = []
for line in lines:
    lines = line[:-1].split(';')
    lines = [float(i) for i in lines]
    log.append(lines)
lab2 = pd.DataFrame(log)
lab_1 = lab2.sort_values(by=0)
lab = lab_1.drop_duplicates(0).drop(lab_1.columns[0:1], axis=1).replace(np.nan, 0)
print(lab.duplicated().any())


# Function to Round labels to nearest 0.5 as in input.
def round_labels(number):
    return round(number * 2) / 2


rounded_labels = round_labels(lab)
rounded_labels = rounded_labels.to_numpy()

# This part converts log file to labels.
lab1 = rounded_labels[:, 1:]  # labels are rounded
print(lab1)
labels = np.zeros(100000)
for i in range(len(lab1)):
    for j in range(len(lab1[i, :])):
        if lab1[i, j] != 0:
            labels[int((lab1[i, j]) * 2)] = i + 1
        else:
            labels[int((lab1[i, j]) * 2)] = 0
labels[0] = 1
print(labels[0:1000])
print(labels.shape)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(Standardize_data, labels, test_size=0.33, shuffle=False) # check for features
print(np.count_nonzero(y_train == 0))  # Look at Standardized or Normalized Data
print(np.count_nonzero(y_test == 0))

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

#  One hot coding using Keras Categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('X_train:', X_train.shape, 'y_train:', y_train.shape, type(X_train), type(y_train))
print('X_test:', X_test.shape, 'y_test:', y_test.shape, type(X_test), type(y_test))

# Reshape input to be 3D [samples, timesteps, features] as expected by GRU
look_back = 30
batch_size = 512
train_data_gen = TimeseriesGenerator(X_train, y_train, length=look_back, batch_size=batch_size, shuffle=False)
test_data_gen = TimeseriesGenerator(X_test, y_test, length=look_back, batch_size=batch_size, shuffle=False)

# Define Hyperparameter.
epochs = 400
learning_rate = 0.001

# Creating GRU model using tensorflow Keras
model = keras.Sequential([
    keras.layers.Bidirectional(keras.layers.GRU(units=80, return_sequences=True, activation='tanh', recurrent_dropout=0.5),
                               input_shape=(look_back, X_train.shape[1])),
    keras.layers.Dropout(0.5),
    keras.layers.Bidirectional(keras.layers.GRU(units=80, activation='tanh', recurrent_dropout=0.5)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(units=21, activation='softmax')
])

model.summary()

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.CategoricalAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc')
]

earlystop_callback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto',
    baseline=None, restore_best_weights=False)

opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=METRICS)
history = model.fit(train_data_gen, validation_data=test_data_gen, epochs=1, shuffle=False, callbacks=[tensorboard_callback, earlystop_callback])

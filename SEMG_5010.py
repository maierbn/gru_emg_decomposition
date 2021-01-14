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
print(tensorflow.__version__)
print(keras.__version__)
print(np.__version__)
# Read CSV using Pandas Framework and covert data to pandas DataFrame
data = pd.read_csv(r"C:\Users\SRIJAY\Desktop\Research Project\Data SET\50mus-10s\50mus-10s\electrodes.csv",
                   delimiter=';', header=0, na_values=['NA'], encoding='ISO-8859-9')
data = pd.DataFrame(data)
data['#timestamp'] = pd.to_datetime(data['#timestamp'], infer_datetime_format=True)
data['#timestamp'] = pd.Series(list(range(len(data))))

# Respective Data of Electrodes Value and Electrodes axis Position.
time_data = data['t']
electrode_data = data.drop(data.iloc[:, :1155], axis=1)
position_data = data.drop(data.iloc[:, 1155:1539], axis=1)
position_data = position_data.drop(position_data.iloc[:, 0:3], axis=1)

# Adding Gaussian Noise for Regularization
electrode_data = electrode_data + np.random.normal(loc=0.0, scale=0.1, size=electrode_data.shape)
print(electrode_data.shape)


# Function for z-Score Standardisation.
def standardize(dataset):
    data_standard = dataset.apply(zscore)
    return data_standard


Standardize_data = standardize(electrode_data)
Standardize_data = Standardize_data.to_numpy()
print(Standardize_data.dtype)

# Import .mat file.
mat = scipy.io.loadmat(r'C:\Users\SRIJAY\Desktop\Research Project\Data SET\50mus-10s\50mus-10s\emg_results.mat')

# Import Stimulated.log (Labels)
log_file = open(r'C:\Users\SRIJAY\Desktop\Research Project\Data SET\50mus-10s\50mus-10s\stimulation.log')
lines = log_file.read().splitlines()
lines = lines[1:]
log = []
for line in lines:
    lines = line[:-1].split(';')
    lines = [float(i) for i in lines]
    log.append(lines)
lab = pd.DataFrame(log)
lab = lab.sort_values(by=0)
lab = lab.drop_duplicates(0).drop(lab.columns[0:1], axis=1).replace(np.nan, 0)
print(lab.duplicated().any())


# Function to Round labels to nearest 0.5 as in input.
def round_labels(number):
    return round(number * 2) / 2


rounded_labels = round_labels(lab)
rounded_labels = rounded_labels.to_numpy()


# This part converts log file to labels.
lab1 = rounded_labels[:, 1:]  # labels are rounded
labels = np.zeros(20000)
for i in range(len(lab1)):
    for j in range(len(lab1[i, :])):
        if lab1[i, j] != 0:
            labels[int((lab1[i, j]) * 2)] = i + 1
        else:
            labels[int((lab1[i, j]) * 2)] = 0
labels[0] = 1

print(type(labels))


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(Standardize_data, labels, test_size=0.33, shuffle=False)
print(np.count_nonzero(y_test == 0))


print('X_train:', X_train.shape, 'y_train:', y_train.shape, type(X_train), type(y_train))
print('X_test:', X_test.shape, 'y_test:', y_test.shape, type(X_test), type(y_test))

# plt.hist(labels, bins=51)
# plt.show()


#  One hot coding using Keras Categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Reshape input to be 3D [samples, timesteps, features] as expected by GRU
look_back = 20
batch_size = 512
train_data_gen = TimeseriesGenerator(X_train, y_train, length=look_back, batch_size=batch_size, shuffle=False)
test_data_gen = TimeseriesGenerator(X_test, y_test, length=look_back, batch_size=batch_size, shuffle=False)

# Define Hyperparameter.
epochs = 100
learning_rate = 0.001

# Creating GRU model using tensorflow Keras
model = keras.Sequential([
    keras.layers.Bidirectional(keras.layers.GRU(units=80, return_sequences=True, activation='tanh'),
                               input_shape=(look_back, X_train.shape[1])),
    keras.layers.Dropout(0.5),
    keras.layers.Bidirectional(keras.layers.GRU(units=80, activation='tanh')),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(units=y_train.shape[1], activation='softmax')
])

model.summary()
opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=[tensorflow.keras.metrics.Precision()])
history = model.fit(train_data_gen, validation_data=test_data_gen, epochs=1, shuffle=False)  # add callback for early stopping


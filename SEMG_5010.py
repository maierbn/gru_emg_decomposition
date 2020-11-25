import keras as keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
# Read CSV using Pandas Framework and covert data to pandas DataFrame
from tensorflow.python.keras.optimizer_v2.adam import Adam

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


# print(time_data)
# print(electrode_data)


# Function for Normalizing the Data.
def normalize(dataset):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(dataset)
    return data_normalized


Norm_electrode_data = normalize(electrode_data)

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

# This part converts log file to labels.
lab1 = rounded_labels.iloc[:, 1:]  # labels are rounded

labels = np.zeros(20000)
for i in range(len(lab1)):
    for j in range(len(lab1.iloc[i, :])):
        if lab1.iloc[i, j] != 0:
            labels[int((lab1.iloc[i, j]) * 2)] = i + 1
        else:
            labels[int((lab1.iloc[i, j]) * 2)] = 0
labels[0] = 1

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(Norm_electrode_data, labels, test_size=0.33)
print('X_train:', X_train.shape, 'y_train:', y_train.shape)
print('X_test:', X_test.shape, 'y_test:', y_test.shape)

#plt.hist(labels, bins=51)
#plt.show()

#  One hot coding using Keras Categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape)

# Reshape input to be 3D [samples, timesteps, features] as expected by GRU
look_back = 80
batch_size = 512
train_data_gen = TimeseriesGenerator(X_train, y_train, length=look_back, batch_size=batch_size, shuffle=True)
test_data_gen = TimeseriesGenerator(X_test, y_test, length=look_back, batch_size=batch_size)

# Define Hyperparameters.
epochs = 400
learning_rate = 0.001

# Creating GRU model using tensorflow Keras

model = keras.Sequential([
    keras.layers.Bidirectional(keras.layers.GRU(units=80, return_sequences=True, activation='tanh'), input_shape=(80, 384)),
    keras.layers.Dropout(0.5),
    keras.layers.Bidirectional(keras.layers.GRU(units=80, activation='tanh')),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=51, activation='softmax')
])

model.summary()
# callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
history = model.fit(train_data_gen, validation_data=test_data_gen, steps_per_epoch=26, epochs=1)  # add callback for early stopping


# evaluate the model
_, train_acc = model.evaluate(train_data_gen, verbose=0)
_, test_acc = model.evaluate(test_data_gen, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))


# # plot loss during training
# plt.subplot(211)
# plt.title('Loss')
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# # plot accuracy during training
# plt.subplot(212)
# plt.title('Accuracy')
# plt.plot(history.history['accuracy'], label='train')
# plt.plot(history.history['val_accuracy'], label='test')
# plt.legend()
# plt.show()


# preds_classes = model.predict_classes(train_data_gen)
# for i in range(len(train_data_gen)):
#     print("Predicted=%s" % (preds_classes[i]))

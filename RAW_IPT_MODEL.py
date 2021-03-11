import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import scipy.io
import tensorflow
from tensorflow import keras
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from scipy.stats import zscore
from tensorflow.python.keras.callbacks import TensorBoard
import datetime
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import itertools
import sys
import os

print(tensorflow.__version__)
path = sys.argv[1]


def listDir(dir):
    filenames = os.listdir(dir)
    for filename in filenames:
        print('File Name: ' + filename)
        if filename == 'electrodes.csv':
            input_data = pd.read_csv(os.path.join(dir, filename), delimiter=',', header=0, na_values=['NA'],
                                     encoding='ISO-8859-9')
        elif filename == 'stimulation.log':
            output_data = open(os.path.join(dir, filename), 'r')
        elif filename == 'emg_results.mat':
            mat_file = scipy.io.loadmat(os.path.join(dir, filename))
    return input_data, output_data, mat_file


inp_data, out_data, IPT_mat_file = listDir(path)


# Read CSV using Pandas Framework and covert data to pandas DataFrame
# Data Cleaning
def generate_inputs(input_data):
    data = input_data
    data = pd.DataFrame(data)
    time_data = data['t']
    electrode_data = data.drop(data.iloc[:, :2], axis=1)
    electrode_data = data.iloc[:, 195:345]
    # for adding gaussian noise with mean 0, unit variance(0, 1)
    electrode_data = electrode_data + np.random.normal(loc=0.0, scale=0.1, size=electrode_data.shape)
    print(electrode_data.shape)
    # Function for Z-score standardization
    standardize_elec_data = electrode_data.apply(zscore)
    return standardize_elec_data


Standardize_data = generate_inputs(inp_data)
print(Standardize_data.shape)


# Import Stimulated.log (Labels)
# Clean the data for labels
# This part converts log file to labels with respect to input time step.
def generate_labels(output_data):
    log_file = output_data
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
    rounded_labels = round(lab * 2) / 2
    rounded_labels = rounded_labels.to_numpy()
    lab1 = rounded_labels[:, 1:]
    print(lab1)
    labels = np.zeros(Standardize_data.shape[0])
    for i in range(len(lab1)):
        for j in range(len(lab1[i, :])):
            if lab1[i, j] != 0:
                labels[int((lab1[i, j]) * 2)] = i + 1
            else:
                labels[int((lab1[i, j]) * 2)] = 0
    labels[0] = 1
    print(labels.shape)
    return lab1, labels


ground, labels = generate_labels(out_data)
print(labels[0:10])

def IPT_LABELS(mat_file):
    mat_df = pd.DataFrame(mat_file['IPTs'])
    mat_df = mat_df.iloc[:, :80000]
    print(mat_df)
    mat_df = mat_df + np.random.normal(loc=0.0, scale=0.1, size=mat_df.shape)
    mat_df_standardize = mat_df.apply(zscore)
    mat_df_standardize = mat_df_standardize.replace(np.nan, 0)
    mat_numpy = np.array(mat_df)
    return mat_numpy


mat_num = IPT_LABELS(IPT_mat_file)


# Prepare data for training
def train_test_data(input_data, output_data, look_back, bs):
    X_tra, X_tes, y_tra, y_tes = train_test_split(input_data, output_data, test_size=0.25, shuffle=False)
    X_tra = X_tra.to_numpy()
    X_tes = X_tes.to_numpy()
    y_tra = np.insert(y_tra, 0, 0, axis=0)
    y_tes = np.insert(y_tes, 0, 0, axis=0)
    y_tra = y_tra[1:len(y_tra)]
    y_tes = y_tes[1:len(y_tes)]
    print(np.unique(y_tra))
    print(np.unique(y_tes))
    print('X_train:', X_tra.shape, 'y_train:', y_tra.shape, type(X_tra), type(y_tra))
    print('X_test:', X_tes.shape, 'y_test:', y_tes.shape, type(X_tes), type(y_tes))
    # Reshape input to be 3D [samples, time steps, features] as expected by GRU
    # Used Timeseriesgenerator API to do the above reshaping
    train_data = TimeseriesGenerator(X_tra, y_tra, length=look_back, batch_size=bs, shuffle=False)
    test_data = TimeseriesGenerator(X_tes, y_tes, length=look_back, batch_size=bs, shuffle=False)
    return train_data, test_data, X_tra, y_tra, X_tes, y_tes


length = 400
batch = 512
print(np.transpose(mat_num).shape)
mat_num_trans = np.transpose(mat_num)
Standardize_data.shape
train_data_gen, test_data_gen, X_train, y_train, X_test, y_test = train_test_data(Standardize_data, mat_num_trans,
                                                                                  length,
                                                                                  batch)


def create_model(look_back, input_shape, output_shape):
    # Creating GRU model using tensorflow Keras
    model = keras.Sequential([
        keras.layers.Bidirectional(
            keras.layers.GRU(units=80, return_sequences=True, activation='tanh', recurrent_dropout=0.5),
            input_shape=(look_back, input_shape)),
        keras.layers.Bidirectional(keras.layers.GRU(units=80, activation='tanh', recurrent_dropout=0.5)),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(units=output_shape, activation='softmax')
    ])
    return model, model.summary()


epochs = 100
learning_rate = 0.001
batch_size = 512
input_s = X_train.shape[1]
output = y_train.shape[1]
model, model_summary = create_model(length, input_s, output)
print(model_summary)

# Early stopping callback when false negative are min.
earlystop_callback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0,
                                                              mode='min', baseline=None, restore_best_weights=False)

# Tensorboard Callback function
logs = "logs" + datetime.datetime.now().strftime("%y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logs, histogram_freq=1)

# Adam Optimizer
opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt, loss='mean_squared_error')
history = model.fit(train_data_gen, validation_data=test_data_gen, epochs=epochs,
                    shuffle=False, callbacks=[tensorboard_callback, earlystop_callback])


def raster_plot(label):
    predictions = model.predict(test_data_gen)
    np.savetxt('prediction.csv', predictions, delimiter=',')
    model.save('2040_IPT_MODEL.h5')

    IPT = []
    for i in range(len(predictions)):
        if np.where(predictions[i] > 0.25):
            IPT.append(np.where(predictions[i] > 0.25))
    threshold_IPT = np.array(IPT)
    IPT_pred = pd.DataFrame(threshold_IPT)
    np.savetxt('threshold_IPT.csv', threshold_IPT, delimiter=',')

    b = label[len(y_train):]
    vec = []
    for i in range(len(np.unique(b))):
        if np.where(b == i):
            IPT = np.where(b == i)
            vec.append(IPT)
    vec_new = pd.DataFrame(vec)
    IPT_ground = vec_new.iloc[1:, 0]
    print(IPT_ground)

    plt.rcParams["figure.figsize"] = (20, 7)
    plt.xlabel('Time')
    plt.ylabel('Motor units')
    plt.title('Rastor Plot')
    plt.eventplot(IPT_pred, color='red', linestyles=':', lineoffsets=1, linelengths=0.5, linewidths=1)
    plt.eventplot(IPT_ground, color='black', linestyles=':', lineoffsets=1, linelengths=0.5, linewidths=0.5)
    colors = ['red', 'black']
    lines = [Line2D([0], [0], color=c, linewidth=4, linestyle=':') for c in colors]
    labels_all = ['IPT_predicted', 'IPT Ground truth']
    plt.legend(lines, labels_all)
    plt.show();


raster_plot(labels)
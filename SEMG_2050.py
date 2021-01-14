import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

print(tensorflow.__version__)

# Tensorboard Callback function
logs = "logs" + datetime.datetime.now().strftime("%y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logs, histogram_freq=1)

# Read CSV using Pandas Framework and covert data to pandas DataFrame
data = pd.read_csv(r"C:\Users\SRIJAY\Desktop\Research Project\Data SET\20mus-50s\20mus-50s\electrodes.csv",
                   delimiter=',', header=0, na_values=['NA'], encoding='ISO-8859-9')
data = pd.DataFrame(data)

# Data Cleaning
time_data = data['t']
electrode_data = data.drop(data.iloc[:, :2], axis=1)
electrode_data = data.iloc[:, 195:345]

# for adding gaussian noise with mean 0, unit variance(0, 1)
electrode_data = electrode_data + np.random.normal(loc=0.0, scale=0.1, size=electrode_data.shape)
print(electrode_data.shape)


# Function for Z-score standardization
def standardize(dataset):
    data_standard = dataset.apply(zscore)
    return data_standard


Standardize_data = standardize(electrode_data)

# Import Stimulated.log (Labels)
# Clean the data for labels
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


# Function to Round labels to nearest 0.5ms as in input.
def round_labels(number):
    return round(number * 2) / 2


rounded_labels = round_labels(lab)
rounded_labels = rounded_labels.to_numpy()

# This part converts log file to labels with respect to input time step.
lab1 = rounded_labels[:, 1:]
print(lab1)
labels = np.zeros(100000)
for i in range(len(lab1)):
    for j in range(len(lab1[i, :])):
        if lab1[i, j] != 0:
            labels[int((lab1[i, j]) * 2)] = i + 1
        else:
            labels[int((lab1[i, j]) * 2)] = 0
labels[0] = 1
print(labels.shape)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(Standardize_data, labels, test_size=0.25,
                                                    shuffle=False)
print(np.count_nonzero(y_train == 0))
print(np.count_nonzero(y_test == 0))
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = np.insert(y_train, 0, 0, axis=0)
y_test = np.insert(y_test, 0, 0, axis=0)
y_train = y_train[0:75000]
y_test = y_test[0:25000]
#  One hot coding using Keras Categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('X_train:', X_train.shape, 'y_train:', y_train.shape, type(X_train), type(y_train))
print('X_test:', X_test.shape, 'y_test:', y_test.shape, type(X_test), type(y_test))

# Reshape input to be 3D [samples, time steps, features] as expected by GRU
# Used Timeseriesgenerator API to do the above reshaping

look_back = 400
batch_size = 1
train_data_gen = TimeseriesGenerator(X_train, y_train, length=look_back, batch_size=batch_size, shuffle=False)
test_data_gen = TimeseriesGenerator(X_test, y_test, length=look_back, batch_size=batch_size, shuffle=False)

# Define Hyperparameter.
epochs = 200
learning_rate = 0.001
dense_unit = 512

# Creating GRU model using tensorflow Keras

model = keras.Sequential([
    keras.layers.Bidirectional(
        keras.layers.GRU(units=80, return_sequences=True, activation='tanh', recurrent_dropout=0.5),
        input_shape=(look_back, X_train.shape[1])),
    keras.layers.Bidirectional(keras.layers.GRU(units=80, activation='tanh', recurrent_dropout=0.5)),
    keras.layers.Dense(dense_unit, activation='relu'),
    keras.layers.Dense(units=y_train.shape[1], activation='softmax')
])

model.summary()

# Metrics used

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.MeanIoU(name='IoU')
]

# Early stopping callback when false negative are min.
earlystop_callback = tensorflow.keras.callbacks.EarlyStopping(monitor='fn', min_delta=0, patience=20, verbose=0,
                                                              mode='min', baseline=None, restore_best_weights=False)

# Adam Optimizer
opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=METRICS)

# Class weight (Balanced)
y_integers = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights = dict(enumerate(class_weights))

# Custom class weights
# d_class_weights =  {0: 0.8, 1: 4.8, 2: 4.8, 3: 4.8, 4: 4.8, 5: 4.8, 6: 4.8, 7: 4.8, 8: 4.8,
#   9: 4.8, 10: 4.8, 11: 4.8, 12: 4.8, 13: 4.8, 14: 4.5, 15: 4.5, 16: 4.5, 17: 4.8, 18: 4.8, 19: 4.8, 20: 4.8}

history = model.fit(train_data_gen, validation_data=test_data_gen, epochs=epochs, shuffle=False, batch_size=batch_size,
                    class_weight=d_class_weights, callbacks=[tensorboard_callback, earlystop_callback], verbose=0)

print("Test Predictions")
print(model.evaluate(test_data_gen, verbose=0))
predictions = model.predict(test_data_gen)
y_pred = np.argmax(predictions, axis=1)
print(np.unique(y_pred))

print('Confusion Matrix')
y_test = np.argmax(y_test, axis=1)
y_test = y_test[:(len(y_test)-look_back)]
cm = confusion_matrix(y_test, y_pred)
print('Classification Report')
target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                '19', '20']
print(classification_report(y_test, y_pred, target_names=target_names))


# This function plots confusion matrix.
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.rcParams.update({'font.size': 15})
    plt.rcParams["figure.figsize"] = (25, 25)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('Confusion_Matrix.pdf')


plot_confusion_matrix(cm, target_names, title='confusion_matrix, With Class Weight')

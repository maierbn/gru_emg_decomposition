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
    return input_data, output_data


inp_data, out_data = listDir(path)

inp_data = ""

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
    # print(lab1)
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


# Prepare data for training
def train_test_data(input_data, output_data, look_back, bs):
    X_tra, X_tes, y_tra, y_tes = train_test_split(input_data, output_data, test_size=0.25, shuffle=False)
    print(np.count_nonzero(y_tra == 0))
    print(np.count_nonzero(y_tes == 0))
    X_tra = X_tra.to_numpy()
    X_tes = X_tes.to_numpy()
    y_tra = np.insert(y_tra, 0, 0, axis=0)
    y_tes = np.insert(y_tes, 0, 0, axis=0)
    y_tra = y_tra[0:75000]
    y_tes = y_tes[0:25000]
    #  One hot coding using Keras Categorical
    y_tra = y_tra[0:len(y_tra) - 1]
    y_tes = y_tes[0:len(y_tes) - 1]
    print('X_train:', X_tra.shape, 'y_train:', y_tra.shape, type(X_tra), type(y_tra))
    print('X_test:', X_tes.shape, 'y_test:', y_tes.shape, type(X_tes), type(y_tes))
    # Reshape input to be 3D [samples, time steps, features] as expected by GRU
    # Used Timeseriesgenerator API to do the above reshaping
    train_data = TimeseriesGenerator(X_tra, y_tra, length=look_back, batch_size=bs, shuffle=False)
    test_data = TimeseriesGenerator(X_tes, y_tes, length=look_back, batch_size=bs, shuffle=False)
    return train_data, test_data, X_tra, y_tra, X_tes, y_tes


length = 400
batch = 512
train_data_gen, test_data_gen, X_train, y_train, X_test, y_test = train_test_data(Standardize_data, labels, length,
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

# Metrics used
METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.MeanIoU(name='IoU', num_classes=21)
]

# Early stopping callback when false negative are min.
earlystop_callback = tensorflow.keras.callbacks.EarlyStopping(monitor='fn', min_delta=0, patience=20, verbose=0,
                                                              mode='min', baseline=None, restore_best_weights=False)

# Tensorboard Callback function
logs = "logs" + datetime.datetime.now().strftime("%y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logs, histogram_freq=1)

# Adam Optimizer
opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=METRICS)


def class_weight(Y_train):
    y_integers = np.argmax(Y_train, axis=1)
    class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
    weights = dict(enumerate(class_weights))
    # Custom class weights
    # weights =  {0: 0.8, 1: 4.8, 2: 4.8, 3: 4.8, 4: 4.8, 5: 4.8, 6: 4.8, 7: 4.8, 8: 4.8,
    #   9: 4.8, 10: 4.8, 11: 4.8, 12: 4.8, 13: 4.8, 14: 4.5, 15: 4.5, 16: 4.5, 17: 4.8, 18: 4.8, 19: 4.8, 20: 4.8}
    return weights


d_class_weights = class_weight(y_train)

history = model.fit(train_data_gen, validation_data=test_data_gen, epochs=epochs, shuffle=False, batch_size=batch_size,
                    class_weight=d_class_weights, callbacks=[tensorboard_callback, earlystop_callback], verbose=0)


def validation(Y_test):
    print("Test Predictions")
    print(model.evaluate(test_data_gen, verbose=0))
    predictions = model.predict(test_data_gen)
    y_prediction = np.argmax(predictions, axis=1)
    print(np.unique(y_prediction))
    np.savetxt('prediction.csv', predictions, delimiter=',')
    np.savetxt('y_prediction.csv', y_prediction, delimiter=',')
    print('Confusion Matrix')
    y_tst = np.argmax(Y_test, axis=1)
    y_tst = y_tst[:(len(y_tst) - length)]
    cm = confusion_matrix(y_tst, y_prediction)
    print('Classification Report')
    Motor_unit = len(np.unique(labels))
    target_names = [str(x) for x in range(Motor_unit)]
    cr = classification_report(y_tst, y_prediction, target_names=target_names)
    print(cr)
    return cm, cr, target_names, y_prediction


conf_matrix, class_report, target, y_pred = validation(y_test)


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


plot_confusion_matrix(conf_matrix, target, title='confusion_matrix, With Class Weight')


def Rastor_plot(predictions, ground_t):
    pred_num = np.array(predictions)
    vec = []
    for i in range(len(np.unique(pred_num))):
        if np.where(pred_num == i):
            IPT = np.where(pred_num == i)
            vec.append(IPT)
    vec_new = pd.DataFrame(vec)
    vec_num = vec_new.iloc[1:, 0]
    vec_plt = np.array(vec_num)
    plt.rcParams["figure.figsize"] = (20, 7)
    plt.xlabel('Time/sec')
    plt.ylabel('Motor units')
    plt.title('Ground Truth TEST LABEL')
    plt.eventplot(ground_t[:, 914:], color='red', linestyles=':', lineoffsets=0.5, linelengths=0.5, linewidths=0.5)
    plt.eventplot(vec_plt, color='green', linestyles='solid', lineoffsets=0.5, linelengths=0.5, linewidths=0.5)
    plt.axis([0, 5000, 0, 8])
    plt.legend()
    plt.show()


Rastor_plot(y_pred, ground)

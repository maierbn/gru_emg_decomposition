import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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
print(time_data)
print(electrode_data)


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
lab = lab.drop_duplicates(0).drop(lab.columns[1], axis=1).replace(np.nan, 0)
lab = lab.sort_values(by=0)
print(lab.duplicated().any())


# Function to Round labels to nearest 0.5 as in input.
def round_labels(number):
    return round(number * 2) / 2


rounded_labels = round_labels(lab)

# This part converts log file to labels.
lab1 = rounded_labels.iloc[:, 1:]
f = []
for i in range(len(lab1)):
    for j in range(len(lab1.iloc[i, :])):
        if (lab1.iloc[i, j] != 0).all():
            f.append(i + 1)
        else:
            f.append(0)
f[0] = 1
print(f)

labels = np.zeros(20000)
for i in range(len(lab)):
    for j in range(len(lab.iloc[i, :])):
        if lab.iloc[i, j] != 0:
            labels[int((lab.iloc[i, j]) * 2)] = i + 1

print(labels)

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(Norm_electrode_data, labels, test_size=0.33)
print(X_train.shape)
print(y_train.shape)


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from sklearn.preprocessing import MinMaxScaler

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
labels = pd.DataFrame(log)
labels = labels.drop_duplicates(0).drop(labels.columns[1], axis=1).replace(np.nan, 0)
labels = labels.sort_values(by=0)
print(labels.duplicated().any())


# Function to Round labels to nearest 0.5 as in input.
def round_labels(number):
    return round(number * 2) / 2


rounded_labels = round_labels(labels)
print(rounded_labels)


# Preparing Labels for training and testing

import zipfile
import numpy as np

zip_path = '/home/guest/Amritha/samples_generated (1).zip'
extract_folder = '/home/guest/Amritha/unzipped_sample'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

import os
import pickle

# Replace this with your actual folder path inside Google Drive
folder_path = '/home/guest/Amritha/unzipped_sample'
time_series_list = []
labels = []
filenames = []

"""is it necessary for the neural network to have data  and labels in lists"""


for filename in os.listdir(folder_path):
    if filename.endswith('.pkl') or filename.endswith('.pickle'):
        #print(f"Loading file: {filename}")  # 👈 This line shows the filename
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            sol = data["Time series"]  # shape: (10000, num_features)
            degree = data["degree"]    # class label: 0, 1, 2, or 3


             # Extract the actual array and shape it correctly
            sol_array = np.array(sol.y).T  # shape: (timesteps, features)
            time_series_list.append(sol_array)
            labels.append(degree)
            filenames.append(filename)

            #time_series_list.append(sol)
            #labels.append(degree)

print(f"Loaded {len(time_series_list)} samples.")
print(f"First sample shape: {time_series_list[0].shape}")
print(f"First label: {labels[1]}")


for i, ts in enumerate(time_series_list):
    print(f"Sample {i}: shape {ts.shape} filename: {filenames[i]}")


#Code to detect non-matching shapes:
expected_timesteps = 10000  # The number of time steps you're expecting

for i, ts in enumerate(time_series_list):
    if ts.shape[0] != expected_timesteps:
        print(f"Sample {i}: shape {ts.shape}, file: {filenames[i]}")

#Code to delete non-matching shaped samples
for i, ts in enumerate(time_series_list):
    if ts.shape[0] != expected_timesteps:
        file_to_delete = os.path.join(folder_path, filenames[i])
        print(f"Deleting: {file_to_delete}")
        os.remove(file_to_delete)
# Reload the time series samples.
time_series_list = []
labels = []
filenames = []


"""is it necessary for the neural network to have data  and labels in lists"""


for filename in os.listdir(folder_path):
    if filename.endswith('.pkl') or filename.endswith('.pickle'):
        print(f"Loading file: {filename}")    #This line shows the filename
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            sol = data["Time series"]  # shape: (10000, num_features)
            degree = data["degree"]    # class label: 0, 1, 2, or 3

             # Extract the actual array and shape it correctly
            sol_array = np.array(sol.y).T  # shape: (timesteps, features)
            time_series_list.append(sol_array)
            labels.append(degree)
            filenames.append(filename)

            #time_series_list.append(np.array(sol))
            #time_series_list.append(sol)  # No need to wrap again


            #labels.append(degree)

print(f"Loaded {len(time_series_list)} samples.")
print(f"First sample shape: {time_series_list[0].shape}")
print(f"First label: {labels[0]}")



for i, ts in enumerate(time_series_list):
    print(f"Sample {i}: shape {ts.shape} filename: {filenames[i]}")


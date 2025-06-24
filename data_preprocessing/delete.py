from data_preprocessing.extract import time_series_list
from data_preprocessing.extract import filenames
from data_preprocessing.extract import folder_path
import os
import pickle
import numpy as np


##########################################################################################
####################Code to detect non-matching shapes:###################################


expected_timesteps = 10000  # The number of time steps you're expecting
"""
for i, ts in enumerate(time_series_list):
    if ts.shape[0] != expected_timesteps:
        print(f"Sample {i}: shape {ts.shape}, file: {filenames[i]}")
"""
#########################################################################################
###################delete non-matching shaped samples####################################

deleted_count = 0

for i, ts in enumerate(time_series_list):
    if ts.shape[0] != expected_timesteps:
        file_to_delete = os.path.join(folder_path, filenames[i])
        print(f"Deleting: {file_to_delete}")
        os.remove(file_to_delete)
        deleted_count += 1

print(f"\nTotal deleted samples: {deleted_count}")
#################################################################################



########################################################################################
##################### Reload the time series samples.###################################

time_series_list = []
labels = []
filenames = []





for filename in os.listdir(folder_path):
    if filename.endswith('.pkl') or filename.endswith('.pickle'):
        #print(f"Loading file: {filename}")    #This line shows the filename
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
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
print(f"First label: {labels[1]}")



#for i, ts in enumerate(time_series_list):
#    print(f"Sample {i}: shape {ts.shape} filename: {filenames[i]}")

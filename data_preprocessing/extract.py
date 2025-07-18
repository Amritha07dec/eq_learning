import zipfile
import numpy as np

############################################################
#########EXTRACT ZIP files##################################
"""
zip_path = '/home/guest/Amritha/samples.zip'
import os

extract_folder = '/home/guest/Amritha/unzipped_sample832'
os.makedirs(extract_folder, exist_ok=True)


with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)
"""
import os
import pickle


# Replace this with your actual pickle folder path 
folder_path = '/home/guest/Amritha/pickle_files_val_104'

time_series_list = []
labels = []
filenames = []


for filename in os.listdir(folder_path):
    if filename.endswith('.pkl') or filename.endswith('.pickle'):
        #print(f"Loading file: {filename}")  # 👈 This line shows the filename
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

            #time_series_list.append(sol)
            #labels.append(degree)

print(f"Loaded {len(time_series_list)} samples.")
print(f"First sample shape: {time_series_list[0].shape}")
print(f"First label: {labels[1]}")

#def get_values():
#    return time_series_list, labels, filenames

#get_values()
"""
for i, ts in enumerate(time_series_list):
    print(f"Sample {i}: shape {ts.shape} filename: {filenames[i]}")
"""
print(f"Loaded {len(time_series_list)} samples.")
print(f"First sample shape: {time_series_list[0].shape}")
print(f"First label: {labels[1]}")

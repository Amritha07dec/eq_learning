from data_preprocessing.delete import time_series_list
from data_preprocessing.delete import filenames
from data_preprocessing.delete import folder_path
from train import pad_samples
import os
import pickle
import numpy as np

padded_samples = pad_samples(time_series_list, target_dim=6)

def has_no_dynamics(sample, threshold=1e-12):
    derivative = np.abs(np.gradient(sample, axis=0))
    return np.any(derivative < threshold) #der<thr true only for non dynamic cases.
    #So the fn returns true only when  non dynamic cases exist in the padded samples
    
count=0
for i, sample in enumerate(padded_samples):
    if not has_no_dynamics(sample):
        no_dynamics_samples = [i]
        count=count+1
        print(f"Samples with no dynamics: {no_dynamics_samples} file:{filenames[i]}")
print(count)


"""
def is_constant_dynamics(padded_samples, diff_threshold=1e-8):
    no_dynamics_indices = []
    for i, sample in enumerate(padded_samples):
        diff = np.diff(sample, axis=0)  # differences over time axis within sample
        if np.all(np.abs(diff) < diff_threshold):
            no_dynamics_indices.append(i)
            print(f"Sample {i} shows no dynamics.")
    return no_dynamics_indices

no_dynamics = is_constant_dynamics(padded_samples, diff_threshold=1e-8)
print(f"Samples with no dynamics: {no_dynamics}")
"""
######################################################################################
"""
no_dynamics_threshold = 1e-6  # Adjust based on your data's scale

for i, ts in enumerate(time_series_list):
    # Calculate range (max - min) over the time series for each feature (column)
    ranges = np.ptp(ts, axis=0)  # ptp = peak to peak = max - min
    
    # If all features vary less than threshold, this sample has no dynamics
    if np.all(ranges < no_dynamics_threshold):
        print(f"Sample {i} shows no dynamics: shape {ts.shape}, file: {filenames[i]}")
"""
#print(np.diff(padded_samples[50], axis=0)[:10])

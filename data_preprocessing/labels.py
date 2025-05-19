#from data_preprocessing.delete import time_series_list
from data_preprocessing.delete import labels
from analyze_labels import analyze_labels


print("Unique labels:", set(labels))

analyze_labels(labels)
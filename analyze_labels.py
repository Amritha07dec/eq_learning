# analyze_data.py
from data_preprocessing.delete import labels
from collections import Counter

def analyze_labels(labels):
    label_counts = Counter(labels)
    print("📊 Label Distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"Class {label}: {count} samples")

analyze_labels(labels)

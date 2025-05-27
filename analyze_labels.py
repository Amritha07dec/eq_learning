# analyze_data.py
from data_preprocessing.delete import labels
from collections import Counter
from split import X_train, X_val, y_train, y_val
"""
def analyze_labels(labels):
    label_counts = Counter(labels)
    print("📊 Label Distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"Class {label}: {count} samples")

analyze_labels(labels)


def analyze_labels(name, labels):
    label_counts = Counter(labels)
    print(f"\n📊 Label Distribution in {name}:")
    for label, count in sorted(label_counts.items()):
        print(f"Class {label}: {count} samples")

# Run analysis
analyze_labels("Validation Set", y_val)
analyze_labels("Training Set", y_train)
"""

def analyze_multiple_label_sets(label_sets):
    """
    label_sets: List of tuples in the form (name, labels)
    """
    for name, labels in label_sets:
        label_counts = Counter(labels)
        print(f"\n📊 Label Distribution in {name}:")
        for label, count in sorted(label_counts.items()):
            print(f"Class {label}: {count} samples")

# Example usage
analyze_multiple_label_sets([
    ("Full Dataset", labels),
    ("Training Set", y_train),
    ("Validation Set", y_val)
])
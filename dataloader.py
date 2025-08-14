# data_loader.py
import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np

class ODEDataset(Dataset):
    """
    A PyTorch Dataset class for loading and preprocessing ODE time series data
    from pickle files, including padding and scaling.
    This version loads from a specified directory, assuming it contains either
    training OR testing data. The split is now managed by having separate folders.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.raw_sequences_info = [] # Stores original y, t, degree, system_name from pickles
        self.all_degrees_raw = [] # Used to determine global degree mapping
        self.max_dims = 0       # Max features observed
        self.max_timesteps = 0  # Max timesteps observed

        self._load_raw_data() # Load raw data and determine max_dims/max_timesteps

        # These will be set by create_dataloaders after global maxes are known
        self.padded_sequences = []
        self.mapped_degrees = []
        self.scaler = None 
        self.degree_to_label = {}
        self.label_to_degree = {}

        print(f"ODEDataset initialized for directory: {self.data_dir} (raw data loaded).")

    def _load_raw_data(self):
        """
        Loads raw data from pickle files in the specified directory,
        extracts time series (y, t), and determines max dimensions and timesteps observed.
        """
        pickle_files = [f for f in os.listdir(self.data_dir) if f.endswith('.pkl')]
        if not pickle_files:
            raise FileNotFoundError(f"No pickle files found in directory: {self.data_dir}")

        print(f"Loading {len(pickle_files)} raw pickle files from {self.data_dir}...")
        
        for pkl_file in pickle_files:
            file_path = os.path.join(self.data_dir, pkl_file)
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                time_series_ode_result = data['Time series']
                degree = data['degree']
                
                # sol.y has shape (num_features, num_timesteps). Transpose to (num_timesteps, num_features)
                y_values = torch.tensor(time_series_ode_result.y.T, dtype=torch.float32)
                t_values = torch.tensor(time_series_ode_result.t, dtype=torch.float32)

                # Check for sufficient time points (important for later CDE interpolation)
                if y_values.shape[0] < 2:
                    print(f"Warning: Skipping {pkl_file}: Not enough time points ({y_values.shape[0]}). Minimum 2 required.")
                    continue
                
                # Ensure time values are strictly increasing and sort if necessary
                if not torch.all(t_values[1:] >= t_values[:-1]):
                    # print(f"Warning: Time values not strictly increasing for {pkl_file}. Sorting.") # Can be noisy
                    sorted_indices = torch.argsort(t_values)
                    t_values = t_values[sorted_indices]
                    y_values = y_values[sorted_indices]

                self.raw_sequences_info.append({'y': y_values, 't': t_values, 'degree': degree})
                self.all_degrees_raw.append(degree) # Collect all degrees to build a global mapping

                # Update max dimensions and timesteps found across all samples
                self.max_dims = max(self.max_dims, y_values.shape[-1])
                self.max_timesteps = max(self.max_timesteps, y_values.shape[0])

            except Exception as e:
                print(f"Error loading or processing {pkl_file}: {e}")
                continue
        
        if not self.raw_sequences_info:
            raise ValueError(f"No valid raw data loaded from {self.data_dir} after filtering. Check your pickle files and data structure.")

        print(f"Loaded {len(self.raw_sequences_info)} raw samples from {self.data_dir}.")
        print(f"Max dimensions observed in {self.data_dir}: {self.max_dims}")
        print(f"Max timesteps observed in {self.data_dir}: {self.max_timesteps}")

    def _pad_and_process_sequences(self, target_max_timesteps, target_max_dims, 
                                   global_degree_to_label, global_label_to_degree):
        """
        Pads sequences to global target dimensions and timesteps, and maps degrees to integer labels.
        This function is called by create_dataloaders after global max_dims/timesteps are known.
        """
        self.padded_sequences = []
        self.mapped_degrees = []
        self.degree_to_label = global_degree_to_label # Use the globally defined mapping
        self.label_to_degree = global_label_to_degree

        for seq_data in self.raw_sequences_info:
            original_y = seq_data['y']
            original_t = seq_data['t']
            degree = seq_data['degree']

            # 1. Pad features (last dimension) to target_max_dims
            # Ensure padded_y_features always has target_max_dims in its last dimension
            current_features_dim = original_y.shape[-1]
            if current_features_dim < target_max_dims:
                padding_needed_features = target_max_dims - current_features_dim
                padded_y_features = torch.nn.functional.pad(original_y, (0, padding_needed_features), 'constant', 0)
            else:
                padded_y_features = original_y # No padding needed or already correct size


            # 2. Pad time (first dimension) to target_max_timesteps
            # This is crucial for torch.stack in DataLoader.collate_fn
            current_timesteps_dim = padded_y_features.shape[0]
            if current_timesteps_dim < target_max_timesteps:
                padding_needed_time = target_max_timesteps - current_timesteps_dim
                # Pad only the time dimension with zeros (values should be considered 0 if padded)
                # The pad signature is (padding_left, padding_right, padding_top, padding_bottom, ...)
                # For 2D tensor (timesteps, features), we pad along first dim (timesteps)
                padded_y_full = torch.nn.functional.pad(padded_y_features, (0, 0, 0, padding_needed_time), 'constant', 0)
            else: # If it's already target_max_timesteps or longer, just slice it
                padded_y_full = padded_y_features[:target_max_timesteps, :]


            # 3. Re-generate t_values to match the target_max_timesteps
            # We assume original_t covers the relevant time range.
            # If original_t is empty or has only one point, handle gracefully to avoid linspace errors.
            t_min = original_t.min().item() if original_t.numel() > 0 else 0.0
            t_max = original_t.max().item() if original_t.numel() > 0 else 0.0
            if target_max_timesteps > 1:
                padded_t = torch.linspace(t_min, t_max, target_max_timesteps, dtype=torch.float32)
            else: # Handle case where only 1 timestep is needed (or to avoid linspace(a,b,1) edge cases)
                padded_t = torch.tensor([t_min], dtype=torch.float32)

            self.padded_sequences.append({'y': padded_y_full, 't': padded_t})
            self.mapped_degrees.append(self.degree_to_label[degree])
        
        print(f"ODEDataset for {self.data_dir}: {len(self.padded_sequences)} samples padded to ({target_max_timesteps}, {target_max_dims}).")


    def set_scaler(self, scaler_instance):
        """Allows setting the scaler externally after it's fitted on training data."""
        self.scaler = scaler_instance

    def __len__(self):
        return len(self.padded_sequences)

    def __getitem__(self, idx):
        seq_data = self.padded_sequences[idx]
        padded_y = seq_data['y'] # This is already padded in _pad_and_process_sequences
        padded_t = seq_data['t'] # This is already generated/padded in _pad_and_process_sequences
        degree_label = self.mapped_degrees[idx]

        # Apply the fitted scaler to normalize y_values
        if self.scaler is None:
            raise RuntimeError("Scaler not set for ODEDataset. Call set_scaler() first.")
        
        # StandardScaler expects 2D array: (n_samples, n_features)
        # Here, we treat each time point's feature vector as a sample for scaling.
        scaled_y_flat = self.scaler.transform(padded_y.numpy()) 
        scaled_y = torch.tensor(scaled_y_flat, dtype=torch.float32)

        return padded_t, scaled_y, torch.tensor(degree_label, dtype=torch.long)

def create_dataloaders(train_data_dir, test_data_dir, batch_size=32):
    """
    Loads all data from specified directories and splits into training and testing DataLoaders.
    Handles padding to a global max_dims and max_timesteps.
    """
    # 1. Load raw data for both datasets to determine dataset-specific max_dims/max_timesteps
    train_raw_dataset = ODEDataset(train_data_dir)
    test_raw_dataset = ODEDataset(test_data_dir)

    # 2. Determine global max_dims and max_timesteps across both datasets
    overall_max_dims = max(train_raw_dataset.max_dims, test_raw_dataset.max_dims)
    overall_max_timesteps = max(train_raw_dataset.max_timesteps, test_raw_dataset.max_timesteps)
    
    print(f"\nGlobal maximum dimensions (features) across all data: {overall_max_dims}")
    print(f"Global maximum timesteps across all data: {overall_max_timesteps}")

    # 3. Create a global degree to label mapping based on ALL unique degrees found
    all_unique_degrees = sorted(list(set(train_raw_dataset.all_degrees_raw + test_raw_dataset.all_degrees_raw)))
    global_degree_to_label = {deg: i for i, deg in enumerate(all_unique_degrees)}
    global_label_to_degree = {i: deg for i, deg in enumerate(all_unique_degrees)}
    print(f"Global unique degrees and labels: {global_degree_to_label}")

    # 4. Process (pad/map labels) for both datasets using global maxes and mapping
    train_raw_dataset._pad_and_process_sequences(overall_max_timesteps, overall_max_dims,
                                                 global_degree_to_label, global_label_to_degree)
    test_raw_dataset._pad_and_process_sequences(overall_max_timesteps, overall_max_dims,
                                                global_degree_to_label, global_label_to_degree)

    # 5. Fit StandardScaler using ONLY the processed TRAINING data
    training_y_values_for_scaler = []
    for seq_data in train_raw_dataset.padded_sequences:
        training_y_values_for_scaler.append(seq_data['y'].numpy()) # These are now already padded
    
    if training_y_values_for_scaler:
        # Stack to create a single (total_time_points, num_features) array for scaler fitting
        training_y_values_for_scaler_stacked = np.vstack(training_y_values_for_scaler)
        scaler = StandardScaler()
        scaler.fit(training_y_values_for_scaler_stacked)
        print("StandardScaler fitted successfully on processed TRAINING data.")
    else:
        raise ValueError("No training data available to fit the StandardScaler.")

    # 6. Set the fitted scaler for both datasets
    train_raw_dataset.set_scaler(scaler)
    test_raw_dataset.set_scaler(scaler)

    # 7. Create DataLoaders
    train_loader = DataLoader(train_raw_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_raw_dataset, batch_size=batch_size, shuffle=False)

    input_channels = overall_max_dims # Model input features
    num_classes = len(global_label_to_degree) # Number of unique degrees (classes)
    
    return train_loader, test_loader, input_channels, num_classes, global_label_to_degree

if __name__ == '__main__':
    train_data_directory_path = 'pickle_files_train'
    test_data_directory_path = 'pickle_files_unseen_test' 
    try:
        train_loader, test_loader, input_c, num_c, label_map = create_dataloaders(train_data_directory_path, test_data_directory_path)
        print(f"\nLoaded {len(train_loader.dataset)} training samples.")
        print(f"Loaded {len(test_loader.dataset)} unseen test samples.")
        print(f"Number of input channels (after padding): {input_c}")
        print(f"Number of classes: {num_c}")
        print(f"Label to Degree Map: {label_map}")

        print(f"\nExample: First batch from training loader:")
        for i, (times, y_vals, labels) in enumerate(train_loader):
            print(f"Batch {i+1}:")
            print(f"  Times shape: {times.shape}") # Should be (batch_size, overall_max_timesteps)
            print(f"  Y values shape: {y_vals.shape}") # Should be (batch_size, overall_max_timesteps, overall_max_dims)
            print(f"  Labels shape: {labels.shape}") # Should be (batch_size,)
            break 

        print(f"\nExample: First batch from testing loader:")
        for i, (times, y_vals, labels) in enumerate(test_loader):
            print(f"Batch {i+1}:")
            print(f"  Times shape: {times.shape}") 
            print(f"  Y values shape: {y_vals.shape}") 
            print(f"  Labels shape: {labels.shape}") 
            print(f"  First sample's true degree (label): {label_map[labels[0].item()]}")
            break 

    except Exception as e:
        print(f"An error occurred during data loading: {e}")
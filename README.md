# Project Name: eq_learning

## 📁 Project Structure

- `main.py` – Entry point for running the training or testing workflow.Contains model training, data loading, and evaluation.
- `train.py` – Contains Training utilities such as functions for confusion matrix, padding samples and training.
- `ode_models_dictionary.py` – Contains the definitions of various ODE systems.
- `ode_simulation/ode_simulator.py`–  Simulates ODEs and includes plotting and simulating utilities (functions).
- `ode_simulation/data.py` – 
\- `samples/` – Stores generated time series data samples.
- `lstm_model.pth` – Trained LSTM model weights (ignored from GitHub).
- `confusion_matrix.png` – Visualization of model's performance.

## 📝 Notes

- Use `main.py` to simulate, train, and test.
- Data cleaning scripts are inside `sample_generator/` (if exists).
- To ignore large files in version control, use `.gitignore`.

## 🔧 Setup

```bash
source myenv/bin/activate  # or use Windows path
pip install -r requirements.txt

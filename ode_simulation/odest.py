import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend suitable for SSH
import matplotlib.pyplot as plt
import os

from ode_models_dictionary import ode_systems
from ode_simulation.ode_simulator import simulate_ode_system
from ode_simulation.ode_simulator import plot_phase_space
from ode_simulation.ode_simulator import plot_trajectories
import numpy as np

# Example of how to access the rhs function and parameters_and_IC for the Lorenz system
#system_name = 'Damped_Oscilllator'
#system_name = 'Lorenz'                             #DCF=('Poly', 2, 0)
#system_name = 'Van_der_Pol'                        #DCF=('Poly', 3, 0)
#system_name = 'Lorenz96'                           #DCF=('Poly', 2, 0)
#system_name = 'Rossler'                            #DCF=('Poly', 2, 0)
#system_name = 'Linear_1D'                          #DCF=('Poly', 1, 0)
#system_name = 'Linear_2D_Harmonic_Oscillator'      #DCF=('Poly', 1, 0)
#system_name = 'Linear_3D_Coupled_Oscillators'      #DCF=('Poly', 1, 0)
#system_name = 'Linear_4D_Coupled_Oscillators'      #DCF=('Poly', 1, 0)
#system_name = 'Linear_5D_Coupled_Oscillators'      #DCF=('Poly', 1, 0)
#system_name = 'Duffing_Oscillator'                 #DCF=('Poly', 3, 0)
#system_name = 'Quartic_Oscillator'                 #DCF=('Poly', 4, 0)
#system_name = 'Lotka_Volterra_Cubic'               #DCF=('Poly', 3, 0)

import os
import pickle
import glob
import numpy as np



# ==== USER SETTINGS ====
num_samples = 120            # How many perturbations per (param, IC) combo
perturb_direction = "negative"  # Choose "positive" or "negative"
# ========================

# Generate perturbation factors list
step = 0.01

if perturb_direction == "positive":
    perturbation_factors = [step * i for i in range(1, num_samples + 1)]
elif perturb_direction == "negative":
    perturbation_factors = [-step * i for i in range(1, num_samples + 1)]
else:
    raise ValueError("perturb_direction must be 'positive' or 'negative'")


# Extract the specific system details
system_data = ode_systems[system_name]
rhs_func = system_data['rhs_function']
parameters_and_IC = system_data['parameters_and_IC']
degree = system_data['DCF_values'][1]

# Folder to save pickle files
folder_name = "pickle_files"
os.makedirs(folder_name, exist_ok=True)

# Counter for generated samples
sample_count = 0

# Loop over each parameter & initial condition set
for idx, (params, initial_conditions, description) in enumerate(parameters_and_IC):
    for factor in perturbation_factors:
        # Perturb initial conditions
        perturbed_ic = [ic + factor * ic for ic in initial_conditions]

        print(f"\nSimulating {system_name} (Set {idx + 1}) with perturbation factor {factor}")
        print(f"Parameters: {params}")
        print(f"Initial conditions: {perturbed_ic}")
        print(f"Expected behavior: {description}")

        # Solve the system
        t_span = (0, 20)
        t_eval = np.linspace(t_span[0], t_span[1], 10000)
        sol = simulate_ode_system(rhs_func, t_span, perturbed_ic, params, solver='RK45', t_eval=t_eval)

        # Plot (optional)
        # Define and create output folder
        output_folder = 'plots'
        os.makedirs(output_folder, exist_ok=True)
        plot_trajectories(sol)

        #plot_trajectories(sol)
        traj_filename = os.path.join(output_folder,f"{system_name}_set{idx+1}_trajectory.png")
        plt.savefig(traj_filename)
        plt.close()


        '''
        # Save sample
        time_series_sample = {
            "Time series": sol,
            "degree": degree
        }

        # Construct filename
        #degree_str = "_".join(map(str, degree))
        params_str = "_".join(map(str, params))
        ic_str = "_".join(f"{x:.2f}" for x in perturbed_ic)  # Keep decimals reasonable
        file_name = os.path.join(folder_name, f"{system_name}_Set-{idx + 1}_Deg-{degree}_Params-{params_str}_IC-{ic_str}.pkl")

        with open(file_name, "wb") as f:
            pickle.dump(time_series_sample, f)

        print(f"Pickle file saved at: {file_name}")
        sample_count += 1

# List all saved pickle files
pickle_files = glob.glob(f"{folder_name}/*.pkl")
print("\nAll saved pickle files:")
print("\n".join(pickle_files))

#List All Pickle Files in the Folder

import glob

# Get all .pkl files inside the folder
#pickle_files = glob.glob(f"{folder_name}/*.pkl")

# List all saved pickle files
#pickle_files = glob.glob(f"{folder_name}/*.pkl")
#print("\nAll saved pickle files:")
#print("\n".join(pickle_files))



'''





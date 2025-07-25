




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
#system_name = 'Quadratic_Damped_Oscillator'        #DCF=('Poly', 2, 0)
#system_name = 'SIR'                                #DCF=('Poly', 2, 0)
#system_name = 'Quartic_FitzHugh_Nagumo'            #DCF=('Poly', 4, 0)
#system_name = 'Neuron_Cubic_Model'                 #DCF=('Poly', 3, 0)
#system_name = 'Rössler_Cubic'                      #DCF=('Poly', 3, 0)
#system_name = 'Chemical_Kinetics'                  #DCF=('Poly', 4, 0)
#system_name = 'FitzHugh_Nagumo'                    #DCF=('Poly', 3, 0)
#system_name = 'Quartic_Potential_2D'               #DCF=('Poly', 4, 0)
#system_name = 'Quartic_Coupled_2D'                 #DCF=('Poly', 4, 0)
#system_name = 'Anharmonic_Oscillator'              #DCF=('Poly', 4, 0)
#system_name = 'Modified_Duffing'                   #DCF=('Poly', 4, 0)
#system_name = 'Cubic_Damped_Oscillator'            #DCF=('Poly', 3, 0)
#system_name = 'Cubic_Reaction'                     #DCF=('Poly', 3, 0)
#system_name = 'Cubic_Pendulum'                     #DCF=('Poly', 3, 0)            
#system_name = 'Brusselator'                        #DCF=('Poly', 2, 0)
#system_name = 'Van_der_Pol_Quadratic'              #DCF=('Poly', 2, 0)
#system_name = 'Simple_Quadratic_Oscillator'        #DCF=('Poly', 2, 0)
#system_name = 'Linear_2D_Damped_Harmonic_Oscillator' #DCF=('Poly', 1, 0)
#system_name = 'Linear_3D_Rotational_System'        #DCF=('Poly', 1, 0)
#system_name = 'Linear_4D_Chain'                    #DCF=('Poly', 1, 0)
#system_name = 'Linear_2D_Cross_Coupled'            #DCF=('Poly', 1, 0)
#system_name = 'Linear_ND_Chain_Oscillator'         #DCF=('Poly', 1, 0)
import os
import pickle
import glob

# Counter for the number of samples generated
sample_count = 0


# Perturbation factors2
#0,-0.05,0.05,-0.1, 0.1,-0.15,0.15,-0.2,0.2,-0.25, 0.25,-0.3,0.3,-0.35,0.35,-0.4,0.4,-0.45,0.45, -0.5, 0.5,-0.55,0.55,-0.6,0.6,-0.65,0.65,-0.7,0.7, 0.75, -0.75,-0.8,0.8,-0.85,0.85,-0.9,0.9, -0.95,0.95,1,-1.05,1.05,-1.1, 1.1,-1.15,1.15,-1.2,1.2, -1.25, 1.25,-1.3,1.3,-1.35,1.35,-1.4,1.4,-1.45,1.45, -1.5, 1.5,-1.55,1.55,-1.6,1.6,-1.65,1.65,-1.7,1.7,-1.75, 1.75,-1.8,1.8,-1.85,1.85,-1.9,1.9, -1.95,1.95, -2, 2
 # Original, +25%, -25%  -0.25, 0.0, 0.25, -0.5, 0.5, 0.75, -0.75, -1, 1, -1.25, 1.25, -1.5, 1.5, -1.75, 1.75, -2, 2
perturbation_factors = [-0.25, 0.0, 0.25, -0.5, 0.5, 0.75, -0.75, 1, -1.25, 1.25, -1.5, 1.5, -1.75, 1.75, -2, 2] 

# **LOOP OVER ALL SYSTEMS AND PARAMETER SETS**
for system_name, system_data in ode_systems.items():
    rhs_func = system_data['rhs_function']
    parameters_and_IC = system_data['parameters_and_IC']
    degree = system_data['DCF_values'][1]
#rhs_func = ode_systems[system_name]['rhs_function']
#parameters_and_IC = ode_systems[system_name]['parameters_and_IC']

# Accessing a specific pair of parameters and initial conditions for the selected system
#param_IC_index = 0



#params = parameters_and_IC[param_IC_index][0]  # Parameter values
#initial_conditions = parameters_and_IC[param_IC_index][1]  # Initial conditions
#description = parameters_and_IC[param_IC_index][2]  # Behavior description

#print(f"Simulating {system_name} system with parameters: {params}")
#print(f"Initial conditions: {initial_conditions}")
#print(f"Expected behavior: {description}")



    for idx, (params, initial_conditions, description) in enumerate(parameters_and_IC):
        for factor in perturbation_factors:
            perturbed_ic = [ic + factor * ic for ic in initial_conditions]
            print(f"\nSimulating {system_name} (Set {idx})with perturbation factor {factor}")
            print(f"Parameters: {params}")
            print(f"Initial conditions: {perturbed_ic}")
            print(f"Expected behavior: {description}")



         # Solve the system
            t_span = (0, 20)
            t_eval = np.linspace(t_span[0], t_span[1], 10000)
            sol = simulate_ode_system(rhs_func, t_span, perturbed_ic, params, solver='BDF', t_eval=t_eval)
            params_str = "_".join(map(str, params))
            ic_str = "_".join(map(str, perturbed_ic))
        # Plot results
            # Define and create output folder
            output_folder = '4_deg_plots'
            os.makedirs(output_folder, exist_ok=True)
            #plot_phase_space(sol)
            plot_trajectories(sol)

            print(f"plotting trajectories of set:{idx}")
            traj_filename = os.path.join(output_folder,f"{system_name}_Set-{idx}_Deg-{degree}_Perturb-{factor}_Params-{params_str}_IC-{ic_str}_trajectory.png")
            plt.savefig(traj_filename)
            plt.close()

# Simulating the Lorenz system with first set of parameters and initial conditions
#t_span = (0, 20)
#t_eval = np.linspace(t_span[0], t_span[1], 10000)

#sol = simulate_ode_system(rhs_func, t_span, initial_conditions, params, solver='RK45', t_eval=t_eval)

# Plot phase space and trajectories
#plot_phase_space(sol)
#plot_trajectories(sol)



        # Save as pickle file
            time_series_sample = {
            "Time series": sol,
            "degree": degree
        }


        # Define the folder name for pickle file (generated sample)
            folder_name = "4_deg_pickle_files"
        # Create the folder if it doesn't exist
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

        # Convert lists to strings for filenames
            #degree_str = "_".join(map(str, degree))
            #params_str = "_".join(map(str, params))
            #ic_str = "_".join(map(str, perturbed_ic))

        # Store the relevant details in the file name
            file_name = os.path.join(folder_name, f"{system_name}_Set-{idx}_Deg-{degree}_Perturb-{factor}_Params-{params_str}_IC-{ic_str}.pkl")
 # Save the pickle file in the folder

            with open(file_name, "wb") as f:
                pickle.dump(time_series_sample, f)

            print(f"Pickle file saved at: {file_name}")

        # Increment sample counter
            sample_count += 1



# List all saved pickle files
#pickle_files = glob.glob(f"{folder_name}/*.pkl")
#print("\nAll saved pickle files:")
#print("\n".join(pickle_files))

#List All Pickle Files in the Folder

import glob

# Get all .pkl files inside the folder
pickle_files = glob.glob(f"{folder_name}/*.pkl")






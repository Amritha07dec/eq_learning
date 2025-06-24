"""
This code will let you visualise the time series for a given system name. for a single given parameter
"""
# importing models dictionary
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
system_name = 'Rossler'                            #DCF=('Poly', 2, 0)
#system_name = 'Linear_1D'                          #DCF=('Poly', 1, 0)
#system_name = 'Linear_2D_Harmonic_Oscillator'      #DCF=('Poly', 1, 0)
#system_name = 'Linear_3D_Coupled_Oscillators'      #DCF=('Poly', 1, 0)
#system_name = 'Linear_4D_Coupled_Oscillators'      #DCF=('Poly', 1, 0)
#system_name = 'Linear_5D_Coupled_Oscillators'      #DCF=('Poly', 1, 0)
#system_name = 'Duffing_Oscillator'                 #DCF=('Poly', 3, 0)
#system_name = 'Quartic_Oscillator'                 #DCF=('Poly', 4, 0)
#system_name = 'Lotka_Volterra_Cubic'               #DCF=('Poly', 3, 0)
#system_name = 'Quadratic_Damped_Oscillator'        #DCF=('Poly,2,0)
#system_name = 'SIR'                                #DCF=('Poly,2,0)
#system_name = 'Quartic_FitzHugh_Nagumo'            #DCF=('Poly', 4, 0)
#system_name = 'Neuron_Cubic_Model'                 #DCF=('Poly', 3, 0)
#system_name = 'Rössler_Cubic'                      #DCF=('Poly', 3, 0)
#system_name = 'Chemical_Kinetics'                  #DCF=('Poly', 4, 0)
#system_name = 'FitzHugh_Nagumo'
                            


rhs_func = ode_systems[system_name]['rhs_function']
parameters_and_IC = ode_systems[system_name]['parameters_and_IC']

# Accessing a specific pair of parameters and initial conditions for the selected system
param_IC_index = 0


params = parameters_and_IC[param_IC_index][0]  # Parameter values
initial_conditions = parameters_and_IC[param_IC_index][1]  # Initial conditions
description = parameters_and_IC[param_IC_index][2]  # Behavior description

print(f"Simulating {system_name} system with parameters: {params}")
print(f"Initial conditions: {initial_conditions}")
print(f"Expected behavior: {description}")

# Simulating the Lorenz system with first set of parameters and initial conditions
t_span = (0, 20)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

sol = simulate_ode_system(rhs_func, t_span, initial_conditions, params, solver='RK45', t_eval=t_eval)

# Plot phase space and trajectories

# Define and create output folder
output_folder = 'amritha'
os.makedirs(output_folder, exist_ok=True)

#plotting for visualization
#plot_phase_space(sol)
#print(f"plotting phase space of {param_IC_index}")
#phase_filename = os.path.join(output_folder,f"{system_name}_set{param_IC_index}_phase_space.png")
#plt.savefig(phase_filename)
#plt.close()
plot_trajectories(sol)
print(f"plotting trajectories of {param_IC_index}")
traj_filename = os.path.join(output_folder,f"{system_name}_set{param_IC_index}_trajectory.png")
plt.savefig(traj_filename)
plt.close()
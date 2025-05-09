# importing models dictionary
from ode_models_dictionary import ode_systems
from ode_simulation.ode_simulator import simulate_ode_system
from ode_simulation.ode_simulator import plot_phase_space
from ode_simulation.ode_simulator import plot_trajectories
import numpy as np


# Example of how to access the rhs function and parameters_and_IC for the Lorenz system
system_name = 'Lorenz'
# system_name = 'Van_der_Pol'
# system_name = 'Lorenz96'
# system_name = 'Lotka_Volterra_Cubic'

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
plot_phase_space(sol)
plot_trajectories(sol)
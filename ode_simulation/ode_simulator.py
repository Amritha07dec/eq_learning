
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import yaml

from scipy.integrate import solve_ivp
import os

def simulate_ode_system(rhs_func, t_span, y0, params, solver='LSODA', t_eval=None):
    """
    Simulate the ODE system.

    Parameters:
    - rhs_func: The right-hand side of the ODE as a function of (t, y, params).
    - t_span: Tuple (t_0, t_final), the time span for the simulation.
    - y0: Initial conditions as an array.
    - params: Parameters required by the rhs_func.
    - solver: The ODE solver method ('RK45', 'RK23', 'DOP853', 'LSODA', etc.).
    - t_eval: Array of time points at which to store the solution.

    Returns:
    - sol: Solution object containing times and states.
    """
    # Define the ODE system as a lambda function to pass the parameters
    def ode_func(t, y):
        return rhs_func(t, y, params)

    # Solve the ODE system
    sol = solve_ivp(ode_func, t_span, y0, method=solver, t_eval=t_eval)
    
    return sol


# Updated phase space plot function
def plot_phase_space(sol, state_indices=(0, 1, 2)):
    """
    Plots the phase space of the solution. Automatically switches to 3D if there are more than two states.
    
    Args:
    sol: Solution object from the ODE solver (such as the one returned by scipy's solve_ivp).
    state_indices: Tuple specifying which state variables to plot (default is (0, 1)).
                   For a 3D plot, pass 3 indices, for example (0, 1, 2).
    """
    y = sol.y
    num_states = y.shape[0]
    
    # 2D phase space
    if num_states == 2 or len(state_indices) ==2:
        plt.figure(figsize=(8, 6))
        plt.plot(y[state_indices[0]], y[state_indices[1]], lw=0.8)
        plt.xlabel(f"State {state_indices[0]}")
        plt.ylabel(f"State {state_indices[1]}")
        plt.title("2D Phase Space")
        plt.grid(True)
        plt.show()
    
    # 3D phase space if the system has more than two states
    elif len(state_indices) == 3 and num_states >= 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(y[state_indices[0]], y[state_indices[1]], y[state_indices[2]], lw=0.8)
        ax.set_xlabel(f"State {state_indices[0]}")
        ax.set_ylabel(f"State {state_indices[1]}")
        ax.set_zlabel(f"State {state_indices[2]}")
        ax.set_title("3D Phase Space")
        plt.show()
    
    else:
        print("State indices must be 2 or 3 for phase space plotting.")
def plot_trajectories(sol):
    """
    Plot the trajectories of all state variables over time.
    
    Parameters:
    - sol: Solution object from solve_ivp.
    """
    plt.figure(figsize=(8, 6))
    for i in range(sol.y.shape[0]):
        plt.plot(sol.t, sol.y[i], label=f'State {i}')
    plt.xlabel('Time')
    plt.ylabel('State Variables')
    plt.title('State Variables Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
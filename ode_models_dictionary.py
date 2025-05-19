import numpy as np

"""
Dictionary Structure:
----------------------
This dictionary contains entries for various dynamical systems. Each key in the dictionary corresponds to the name of a
 system (e.g., 'Lorenz', 'Van_der_Pol', 'Rössler').

Each system entry is a dictionary with the following structure:

- DCF_values:  A list of DCF vssalues where the first entry represents the type of functions that appear on the 
    right-hand side, for example, 'Poly' for polynomials, 'Rat' for rational functions, etc. The second entry represents 
    the highest degree of terms appearing on the RHS; for example, for polynomials, a value of 2 means a second-degree
     polynomial. The third entry represents the number of hidden states.

- rhs_function: A lambda function that computes the right-hand side of the ordinary differential equation (ODE) system. 
  It takes the following arguments:
    - t (float): The time variable.
    - y (list): The state variables of the system.
    - params (list): The parameters of the system.

- parameters: A list of tuples, where each tuple contains:
    - A list of parameters (list): The parameters for the system.
    - A list of initial conditions (list): The initial conditions for the system.
    - A string description (str): A description of the behavior corresponding to  the corresponding pair of parameters, 
      and initial conditions. This description can be one of the following:
        - 'chaotic': The system exhibits chaotic behavior.
        - 'cyclic': The system exhibits cyclic behavior.
        - 'fixed point': The system converges to a fixed point.
        - 'NA': The behavior is unknown.
"""


ode_systems = {
    # """
    #
    # Linear systems
    # ------------------
    #
    # """
    'Linear_1D': {
        'DCF_values': ['Poly', 1, 0],
        'rhs_function': lambda t, y, params: [params[0] * y[0]],  # Single state: dx/dt = a * x (no cyclic behavior)
        'parameters_and_IC': [
            ([0.5], [1.0], 'growth'),  # Exponential growth
            ([-0.5], [1.0], 'decay'),  # Exponential decay
        ]
    },
    'Linear_2D_Harmonic_Oscillator': {
        'DCF_values': ['Poly', 1, 0],
        'rhs_function': lambda t, y, params: [
            y[1],  # dx/dt = y
            -params[0] * y[0]  # dy/dt = -omega^2 * x
        ],
        'parameters_and_IC': [
            ([1.0], [1.0, 0.0], 'cyclic (simple harmonic oscillator)'),  # Cyclic behavior (ω = 1)
            ([0.25], [1.0, 0.0], 'cyclic (slower oscillation)'),  # Slower cyclic behavior (ω = 0.5)
        ]
    },
    'Linear_3D_Coupled_Oscillators': {
        'DCF_values': ['Poly', 1, 0],
        'rhs_function': lambda t, y, params: [
            params[0] * y[1],  # dx1/dt = a * x2
            params[1] * y[2],  # dx2/dt = b * x3
            -params[2] * y[0]  # dx3/dt = -c * x1
        ],
        'parameters_and_IC': [
            ([1.0, 1.0, 1.0], [1.0, 0.0, 0.0], 'cyclic'),  # Cyclic with clear oscillations
            ([0.5, 0.5, 0.5], [1.0, 1.0, 1.0], 'slower cyclic')  # Slower oscillatory behavior
        ]
    },
    'Linear_4D_Coupled_Oscillators': {
        'DCF_values': ['Poly', 1, 0],
        'rhs_function': lambda t, y, params: [
            params[0] * y[1],  # dx1/dt = a * x2
            params[1] * y[2],  # dx2/dt = b * x3
            params[2] * y[3],  # dx3/dt = c * x4
            -params[3] * y[0]  # dx4/dt = -d * x1
        ],
        'parameters_and_IC': [
            ([1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0], 'cyclic'),  # Cyclic behavior with balanced coupling
            ([0., 0.5, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0], 'slower cyclic')  # Slower cyclic behavior
        ]
    },
    'Linear_5D_Coupled_Oscillators': {
        'DCF_values': ['Poly', 1, 0],
        'rhs_function': lambda t, y, params: [
            params[0] * y[1],  # dx1/dt = a * x2
            params[1] * y[2],  # dx2/dt = b * x3
            params[2] * y[3],  # dx3/dt = c * x4
            params[3] * y[4],  # dx4/dt = d * x5
            -params[4] * y[0]  # dx5/dt = -e * x1
        ],
        'parameters_and_IC': [
            ([1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0], 'cyclic'),  # Cyclic with clear oscillations
            ([0.5, 0.5, 0.5, 0.5, 0.5], [1.0, 110.0, 110.0, 10.0, -20.0], 'slower cyclic')  # Slower cyclic behavior
        ]
    },
    # """
    #
    # Non-linear systems
    # ------------------
    #
    # """
    'Lorenz': {
        'DCF_values': ['Poly', 2, 0],
        'rhs_function': lambda t, y, params: [
            params[0] * (y[1] - y[0]),                  # dx/dt = sigma * (y - x)
            y[0] * (params[1] - y[2]) - y[1],           # dy/dt = x * (rho - z) - y
            y[0] * y[1] - params[2] * y[2]              # dz/dt = x * y - beta * z
        ],
        'parameters_and_IC': [
            ([10.0, 28.0, 8.0 / 3.0], [1.0, 1.0, 1.0], 'chaotic'),     # Chaotic behavior
            ([10.0, 15.0, 8.0 / 3.0], [0.5, 0.5, 0.5], 'fixed_point'),      # fixed_point
            ([10.0, 100.0, 8.0 / 3.0], [2.0, 2.0, 2.0], 'chaotic')     # Chaotic with different initial conditions
        ]
    },
    'Rossler': {
        'DCF_values': ['Poly', 2, 0],
        'rhs_function': lambda t, y, params: [
            -y[1] - y[2],                               # dx/dt = -y - z
            y[0] + params[0] * y[1],                    # dy/dt = x + a * y
            params[1] + y[2] * (y[0] - params[2])       # dz/dt = b + z * (x - c)
        ],
        'parameters_and_IC': [
            ([0.2, 0.2, 5.7], [1.0, 1.0, 1.0], 'chaotic'),  # Classic chaotic behavior
            ([0.1, 0.1, 6.0], [0.5, 0.5, 0.5], 'cyclic'),  # Cyclic with modified parameters
            ([0.2, 0.2, 10.0], [1.0, 0.0, 0.0], 'chaotic') # Chaotic with higher c
        ]
    },
    'Lorenz96': {
        'DCF_values': ['Poly', 2, 0],
        'rhs_function': lambda t, y, params: [
            (y[(i+1) % params[1]] - y[i-2]) * y[i-1] - y[i] + params[0]
            for i in range(params[1])
        ],  # N-dimensional Lorenz-96 system
        'parameters_and_IC': [
            ([10.0, 4], [0.0, 1.0, 2.0, 3.0], 'cyclic'),               # 4D cyclic system
            ([12.0, 6], [1.0, 0.5, 0.5, 0.5, 1.0, 0.0], 'chaotic')     # 6D chaotic system
        ]
    },
    'SIR': {
    'DCF_values': ['Poly', 2, 0],
    'rhs_function': lambda t, y, params: [
        -params[0] * y[0] * y[1],                      # dS/dt = -beta * S * I
        params[0] * y[0] * y[1] - params[1] * y[1],    # dI/dt = beta * S * I - gamma * I
        params[1] * y[1]                               # dR/dt = gamma * I
    ],
    'parameters_and_IC': [
        ([0.3, 0.1], [0.99, 0.01, 0.0], 'epidemic'),      # R0 = 3
        ([0.2, 0.2], [0.99, 0.01, 0.0], 'threshold'),     # R0 = 1
        ([0.1, 0.3], [0.99, 0.01, 0.0], 'fading_out'),     # R0 = 0.33
        ([0.1, 0.2], [0.99, 0.01, 0.0], 'fading_out'),
        ([0.5, 0.1], [0.99, 0.01, 1.0], 'fading_out'),  
    ]
},
'Quadratic_Damped_Oscillator': {
    'DCF_values': ['Poly', 2, 0],
    'rhs_function': lambda t, y, params: [
        y[1],
        -params[0] * y[1] - params[1] * y[0] - params[2] * y[0]**2
    ],
    'parameters_and_IC': [
        ([0.5, 1.0, 1.0], [1.0, 0.0], 'underdamped'),       # Less damping, oscillatory
        ([2.0, 1.0, 1.0], [1.0, 0.0], 'overdamped'),        # High damping, non-oscillatory
        ([0.1, 0.5, -1.0], [2.0, 0.0], 'unstable'),         # Negative nonlinearity, divergence
    ]
},


    'Van_der_Pol': {
        'DCF_values': ['Poly', 3, 0],
        'rhs_function': lambda t, y, params: [
            y[1],                                       # dx/dt = y
            params[0] * (1 - y[0]**2) * y[1] - y[0]     # dy/dt = mu * (1 - x^2) * y - x
        ],
        'parameters_and_IC': [
            ([0.5], [1.0, 0.0], 'cyclic'),              # Cyclic for small mu
            ([1.0], [0.0, 1.0], 'cyclic'),              # Moderate mu, cyclic
            ([10.0], [2.0, 0.0], 'cyclic'),             # Larger mu, slower periodicity
            ([20.0], [0.1, 0.1], 'cyclic'),              # Very large mu, still cyclic
        ]
    },
    
     'Duffing_Oscillator': {
         'DCF_values': ['Poly', 3, 0],
         'rhs_function': lambda t, y, params: [
             y[1],  # dx/dt = y
             -params[0] * y[1] - params[1] * y[0] - params[2] * y[0]**3 + params[3] * np.cos(params[4] * t)  # dy/dt = -delta * y - alpha * x - beta * x^3 + gamma * cos(omega * t)
         ],
         'parameters_and_IC': [
             ([0.2, 1.0, 0.5, 0.3, 1.0], [1.0, 0.0], 'cyclic'),  # Typical cyclic motion
             ([0.2, 1.0, 0.5, 0.8, 1.0], [0.5, 0.0], 'chaotic'),  # Chaotic motion
         ]
     },
    
    'Lotka_Volterra_Cubic': {
        'DCF_values': ['Poly', 3, 0],
        'rhs_function': lambda t, y, params: [
            params[0] * y[0] - params[1] * y[0] * y[1] - params[2] * y[0]**3,  # dx/dt = alpha * x - beta * x * y - gamma * x^3
            -params[3] * y[1] + params[4] * y[0] * y[1]**2  # dy/dt = -delta * y + epsilon * x * y^2
        ],
        'parameters_and_IC': [
            ([1.0, 0.5, 0.1, 1.0, 0.1], [0.5, 1.0], 'cyclic'),  # Cyclic predator-prey dynamics
            ([1.0, 0.5, 0.3, 1.0, 0.2], [0.7, 0.5], 'complex'),  # More complex dynamics due to cubic interaction
        ]
    },
    'Neuron_Cubic_Model': {
        'DCF_values': ['Poly', 3, 0],
        'rhs_function': lambda t, y, params: [
            y[1],
            -params[0] * y[0]**3 + params[1] * y[0]**2 - params[2] * y[1] + params[3]
        ],
        'parameters_and_IC': [
            ([1.0, -1.5, 0.5, 0.0], [0.0, 0.0], 'resting'),           # neuron at rest
            ([1.2, -1.8, 0.8, 0.1], [0.1, 0.0], 'spiking'),          # spike generation
            ([1.5, -2.0, 1.0, 0.5], [0.5, 0.0], 'bursting')          # bursting pattern
        ]
    },
    'Rössler_Cubic': {
        'DCF_values': ['Poly', 3, 0],
        'rhs_function': lambda t, y, params: [
            -y[1] - y[2],
            y[0] + params[0] * y[1],
            params[1] + y[2] * (y[0] - params[2]) + params[3] * y[0]**3
        ],
        'parameters_and_IC': [
            ([0.2, 0.2, 5.7, 0.0], [0.0, 1.0, 1.0], 'chaotic'),     # classic Rossler attractor
            ([0.15, 0.15, 4.0, 0.3], [0.5, 0.5, 0.5], 'cubic_mod'),# cubic modifies z dynamics
            ([0.1, 0.1, 3.5, 0.7], [1.0, 1.0, 1.0], 'complex')     # stronger cubic nonlinearity
        ]
    },
    'Quartic_Oscillator': {
        'DCF_values': ['Poly', 4, 0],
        'rhs_function': lambda t, y, params: [
            y[1],  # dx/dt = y
            -params[0] * y[0]**3 - params[1] * y[0]**4  # dy/dt = -x^3 - x^4
        ],
        'parameters_and_IC': [
            ([1.0, 0.5], [1.0, 0.0], 'cyclic'),  # Cyclic motion with quartic interaction
            ([1.0, 1.0], [0.5, 0.0], 'complex'),  # More complex motion due to quartic term
        ]
    },
    'Chemical_Kinetics': {
        'DCF_values': ['Poly', 4, 0],
        'rhs_function': lambda t, y, params: [
            -params[0] * y[0]**2 + params[1] * y[1]**4 - params[2] * y[2],
            params[0] * y[0]**2 - params[1] * y[1]**4,
            params[2] * y[1] - y[2]
        ],
        'parameters_and_IC': [
            ([1.0, 0.5, 0.2], [1.0, 0.5, 0.1], 'transient'),
            ([2.0, 1.0, 0.5], [2.0, 1.0, 0.1], 'nonlinear_flow'),
            ([0.8, 1.2, 0.4], [0.1, 0.9, 0.8], 'damped'),
            ([1.5, 0.7, 0.3], [0.9, 0.2, 0.6], 'bistable'),
            ([2.5, 1.3, 0.6], [1.1, 0.3, 0.1], 'spiking'),
            ([3.0, 2.0, 1.0], [0.5, 0.6, 0.9], 'rich_dynamics')
        ]
    },
    'Quartic_FitzHugh_Nagumo': {
        'DCF_values': ['Poly', 4, 0],
        'rhs_function': lambda t, y, params: [
            y[0] - y[0]**3 - y[1] + params[0] * y[0]**4,  # dv/dt
            params[1] * (y[0] - params[2] * y[1] + params[3])  # dw/dt
        ],
        'parameters_and_IC': [
            ([0.5, 0.08, 1.0, 0.7], [0.0, 0.0], 'stable'),        # small quartic term → near fixed point
            ([1.0, 0.2, 1.0, 0.7], [0.5, 0.0], 'oscillatory'),    # regular limit cycle oscillations
            ([2.0, 0.25, 0.9, 0.8], [1.0, 0.0], 'bursting'),      # stronger nonlinearity → spiking
            ([4.0, 0.3, 0.8, 0.5], [1.5, 0.0], 'chaotic-like')    # intense quartic injection → irregularity
        ]
    }
    


    }


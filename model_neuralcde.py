# model_neural_cde.py
import torch
import torch.nn as nn
from torchcde import CubicSpline, cdeint, natural_cubic_spline_coeffs

######################
# A CDE model looks like
#
# z_t = z_0 + \int_0^t f_\theta(z_s) dX_s
#
# Where X is your data and f_\theta is a neural network. So the first thing we need to do is define such an f_\theta.
# That's what this CDEFunc class does.
# Here we've built a small single-hidden-layer neural network, whose hidden layer is of width 128.
######################
class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    def forward(self,t, z):
        #print(z.shape)
        z = self.linear1(z)
        z = z.relu()
        #print(z.shape)
        z = self.linear2(z)
        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        z = z.tanh()
        #print(z.shape)
        ######################
        # Ignoring the batch dimensions, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        #print(z.shape)

        return z


######################
# Next, we need to package CDEFunc up into a model that computes the integral.
######################
class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(NeuralCDE, self).__init__()
        self.hidden_channels = hidden_channels

        self.func = CDEFunc(input_channels, hidden_channels)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)

    # --- IMPORTANT: The forward method signature and contents have been corrected ---
    def forward(self, times, X_data):
        """
        Forward pass for the Neural CDE.

        Args:
            times (torch.Tensor): Time points for the input sequence. Shape: (batch_size, sequence_length).
            X_data (torch.Tensor): The input time series data. Shape: (batch_size, sequence_length, input_channels).
                                   This is 'y_values' from the DataLoader.
        Returns:
            torch.Tensor: Predicted output (e.g., class probabilities/logits).
        """
        # Create a NaturalCubicSpline from the input data (X_data) and its time points (times).
        # This interpolates the discrete data into a continuous path.

        # Debugging prints
        print(f"DEBUG: times shape: {times[0].shape}") # Expected (batch_size, sequence_length) e.g., (32, 10000)
        print(f"DEBUG: X_data shape: {X_data.shape}") # Expected (batch_size, sequence_length, input_channels) e.g., (32, 10000, 6)

        coeffs = natural_cubic_spline_coeffs(X_data, t=times[0])
        print(f"DEBUG: coeffs shape: {coeffs.shape}") # <-- ADD THIS LINE
        spline = CubicSpline(coeffs, t=times[0]) # Note: CubicSpline also needs 't' here in 0.2.x versions

        ######################
        # Easy to forget gotcha: Initial hidden state should be a function of the first observation.
        # We take the first observation from X_data (first element along the time dimension for each batch sample)
        # and pass it through a linear layer to get the initial hidden state z0.
        ######################
        z0 = self.initial(spline.evaluate(times[0][0])) # X_data[:, 0, :] takes the first time point for all batch samples
        #print(f"DEBUG: z0 shape: {z0.shape}") # Expected (batch_size, hidden_channels) e.g., (32, 64)

        ######################
        # Actually solve the CDE.
        # dX_dt: The derivative of the path X, obtained from the spline.
        # z0: The initial hidden state.
        # func: The CDE function (f_theta, our CDEFunc neural network).
        # t: The time points at which to evaluate the CDE. For batching with cdeint,
        #    this `t` should be a 1D tensor of common time points for the whole batch.
        #    We assume the time points are consistent across samples within a batch and use `times[0]`.
        # atol, rtol: Absolute and relative tolerances for the ODE solver (cdeint uses an ODE solver internally).
        ######################

        t_for_cdeint = torch.stack([times[0][0], times[0][-1]])
        #print(f"DEBUG: t_for_cdeint shape: {t_for_cdeint.shape}") # Expected (2,)

        z_T = cdeint(
                     z0=z0,
                     func=self.func,
                     X=spline,
                     t=torch.stack([times[0][0], times[0][-1]]), # Use the time points from the first sample in the batch as the common evaluation grid.
                     atol=1e-4,
                     rtol=1e-4)
        ######################
        # Both the initial value and the terminal value are returned from cdeint;
        # extract just the terminal value (z_T[1]), which is the state at the final time point.
        # Then, apply a linear map (self.readout) for the final classification prediction.
        ######################


        #print(f"DEBUG: z_T after cdeint shape: {z_T.shape}") # Expected (2, batch_size, hidden_channels) e.g., (2, 32, 64)

        z_T = z_T[:,-1,:] # z_T will have shape (batch_size, hidden_channels) at the final time point
       
        #print(f"DEBUG: z_T after slicing (-1) shape: {z_T.shape}") # Expected (batch_size, hidden_channels) e.g., (32, 64)

        pred_y = self.readout(z_T) # pred_y will have shape (batch_size, output_channels)

        #print(f"DEBUG: pred_y shape: {pred_y.shape}") # Expected (batch_size, output_channels) e.g., (32, 4)

        return pred_y
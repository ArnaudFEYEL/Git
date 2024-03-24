import math
import numpy as np
from IPython.display import clear_output
from tqdm import tqdm_notebook as tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.color_palette("bright")
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torch import Tensor
from torch import nn
import torch.optim as optim
from torch.nn  import functional as F 
from torch.autograd import Variable
import streamlit as st 
import iteration

use_cuda = torch.cuda.is_available()

#Importing user's matrix, loss function and iterations
given_matrix = torch.load('data/matrix.pth')

import sys
sys.path.append("data/")
import iteration


    
def ode_solve(z0, t0, t1, f):
    """
    Simplest Euler ODE initial value solver
    Parameters:
    - z0: Initial condition
    - t0: Initial time
    - t1: Final time
    - f: Function defining the ODE (z' = f(z, t))
    
    Returns:
    - z: Solution of the ODE at time t1
    """
    
    h_max = 0.05
    n_steps = math.ceil((abs(t1 - t0)/h_max).max().item())

    h = (t1 - t0)/n_steps
    t = t0
    z = z0

    for i_step in range(n_steps):
        z = z + h * f(z, t)
        t = t + h
    return z

class ODEF(nn.Module):
    """
    Ordinary Differential Equation Function (ODEF) module.
    """
    
    def forward_with_grad(self, z, t, grad_outputs):
        """
        Compute f and its gradients w.r.t. z, t, and parameters.
        
        Parameters:
        - z: Input tensor
        - t: Time tensor
        - grad_outputs: Gradients w.r.t. the output
        
        Returns:
        - out: Output of the forward pass
        - adfdz: Gradient of f w.r.t. z
        - adfdt: Gradient of f w.r.t. t
        - adfdp: Gradients of f w.r.t. parameters
        """
        
        batch_size = z.shape[0]

        out = self.forward(z, t)

        a = grad_outputs
        adfdz, adfdt, *adfdp = torch.autograd.grad(
            (out,), (z, t) + tuple(self.parameters()), grad_outputs=(a),
            allow_unused=True, retain_graph=True
        )
        # grad method automatically sums gradients for batch items, we have to expand them back 
        if adfdp is not None:
            adfdp = torch.cat([p_grad.flatten() for p_grad in adfdp]).unsqueeze(0)
            adfdp = adfdp.expand(batch_size, -1) / batch_size
        if adfdt is not None:
            adfdt = adfdt.expand(batch_size, 1) / batch_size
        return out, adfdz, adfdt, adfdp

    def flatten_parameters(self):
        """
        Flatten and concatenate all parameters of the module.
        
        Returns:
        - flat_parameters: Flattened parameters
        """
        
        p_shapes = []
        flat_parameters = []
        for p in self.parameters():
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters)
    
class ODEAdjoint(torch.autograd.Function):
    """
    Backward pass for Ordinary Differential Equation (ODE) solver using adjoint method.
    """
    
    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func):
        
        """
        Forward pass to solve the ODE and save necessary tensors for the backward pass.
        
        Parameters:
        - ctx: Context to save tensors for backward pass
        - z0: Initial state tensor
        - t: Time tensor
        - flat_parameters: Flattened parameters tensor
        - func: ODEF instance
        
        Returns:
        - z: Solution tensor
        """
        
        assert isinstance(func, ODEF)
        bs, *z_shape = z0.size()
        time_len = t.size(0)

        with torch.no_grad():
            z = torch.zeros(time_len, bs, *z_shape).to(z0)
            z[0] = z0
            for i_t in range(time_len - 1):
                z0 = ode_solve(z0, t[i_t], t[i_t+1], func)
                z[i_t+1] = z0

        ctx.func = func
        ctx.save_for_backward(t, z.clone(), flat_parameters)
        return z

    @staticmethod
    def backward(ctx, dLdz):
        """
        Backward pass to compute gradients using adjoint method.
        
        Parameters:
        - ctx: Context to retrieve saved tensors
        - dLdz: Gradient of loss with respect to z
        
        Returns:
        - Gradients w.r.t. z, t, parameters, and None
        """
        
        func = ctx.func
        t, z, flat_parameters = ctx.saved_tensors
        time_len, bs, *z_shape = z.size()
        n_dim = np.prod(z_shape)
        n_params = flat_parameters.size(0)

        # Dynamics of augmented system to be calculated backwards in time
        def augmented_dynamics(aug_z_i, t_i):
            """
            Dynamics of augmented system to be calculated backwards in time.
            
            Parameters:
            - aug_z_i: Augmented state at time t_i
            - t_i: Time tensor
            
            Returns:
            - Augmented dynamics
            """
            z_i, a = aug_z_i[:, :n_dim], aug_z_i[:, n_dim:2*n_dim]  # ignore parameters and time

            # Unflatten z and a
            z_i = z_i.view(bs, *z_shape)
            a = a.view(bs, *z_shape)
            with torch.set_grad_enabled(True):
                t_i = t_i.detach().requires_grad_(True)
                z_i = z_i.detach().requires_grad_(True)
                func_eval, adfdz, adfdt, adfdp = func.forward_with_grad(z_i, t_i, grad_outputs=a)  # bs, *z_shape
                adfdz = adfdz.to(z_i) if adfdz is not None else torch.zeros(bs, *z_shape).to(z_i)
                adfdp = adfdp.to(z_i) if adfdp is not None else torch.zeros(bs, n_params).to(z_i)
                adfdt = adfdt.to(z_i) if adfdt is not None else torch.zeros(bs, 1).to(z_i)

            # Flatten f and adfdz
            func_eval = func_eval.view(bs, n_dim)
            adfdz = adfdz.view(bs, n_dim) 
            return torch.cat((func_eval, -adfdz, -adfdp, -adfdt), dim=1)

        dLdz = dLdz.view(time_len, bs, n_dim)  # flatten dLdz for convenience
        with torch.no_grad():
            ## Create placeholders for output gradients
            # Prev computed backwards adjoints to be adjusted by direct gradients
            adj_z = torch.zeros(bs, n_dim).to(dLdz)
            adj_p = torch.zeros(bs, n_params).to(dLdz)
            # In contrast to z and p we need to return gradients for all times
            adj_t = torch.zeros(time_len, bs, 1).to(dLdz)

            for i_t in range(time_len-1, 0, -1):
                z_i = z[i_t]
                t_i = t[i_t]
                f_i = func(z_i, t_i).view(bs, n_dim)

                # Compute direct gradients
                dLdz_i = dLdz[i_t]
                dLdt_i = torch.bmm(torch.transpose(dLdz_i.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

                # Adjusting adjoints with direct gradients
                adj_z += dLdz_i
                adj_t[i_t] = adj_t[i_t] - dLdt_i

                # Pack augmented variable
                aug_z = torch.cat((z_i.view(bs, n_dim), adj_z, torch.zeros(bs, n_params).to(z), adj_t[i_t]), dim=-1)

                # Solve augmented system backwards
                aug_ans = ode_solve(aug_z, t_i, t[i_t-1], augmented_dynamics)

                # Unpack solved backwards augmented system
                adj_z[:] = aug_ans[:, n_dim:2*n_dim]
                adj_p[:] += aug_ans[:, 2*n_dim:2*n_dim + n_params]
                adj_t[i_t-1] = aug_ans[:, 2*n_dim + n_params:]

                del aug_z, aug_ans

            ## Adjust 0 time adjoint with direct gradients
            # Compute direct gradients 
            dLdz_0 = dLdz[0]
            dLdt_0 = torch.bmm(torch.transpose(dLdz_0.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

            # Adjust adjoints
            adj_z += dLdz_0
            adj_t[0] = adj_t[0] - dLdt_0
        return adj_z.view(bs, *z_shape), adj_t, adj_p, None

class NeuralODE(nn.Module):
    """
    Neural Ordinary Differential Equation (NeuralODE) module.
    """
    
    def __init__(self, func):
        """
        Initialize NeuralODE module.

        Parameters:
        - func: ODEF instance
        """
        super(NeuralODE, self).__init__()
        assert isinstance(func, ODEF)
        self.func = func

    def forward(self, z0, t=Tensor([0., 1.]), return_whole_sequence=False):
        """
        Forward pass through the NeuralODE.

        Parameters:
        - z0: Initial state tensor
        - t: Time tensor (default: [0., 1.])
        - return_whole_sequence: Whether to return the whole sequence or just the final state

        Returns:
        - z or z[-1]: Sequence of states or the final state
        """
        t = t.to(z0)
        z = ODEAdjoint.apply(z0, t, self.func.flatten_parameters(), self.func)
        if return_whole_sequence:
            return z
        else:
            return z[-1]
        
class LinearODEF(ODEF):
    """
    Linear Ordinary Differential Equation Function (LinearODEF) module.
    Inherits from ODEF.
    """
    def __init__(self, W):
        """
        Initialize LinearODEF module.

        Parameters:
        - W: Weight matrix for the linear layer
        """
        super(LinearODEF, self).__init__()
        self.lin = nn.Linear(2, 2, bias=False)
        self.lin.weight = nn.Parameter(W)

    def forward(self, x, t):
        """
        Forward pass through the LinearODEF.

        Parameters:
        - x: Input tensor
        - t: Time tensor

        Returns:
        - Output of the linear transformation
        """
        return self.lin(x)
    
class SpiralFunctionExample(LinearODEF):
    """
    Example class for a Spiral Function.
    Inherits from LinearODEF.
    """
    def __init__(self):
        """
        Initialize SpiralFunctionExample.
        """
        super(SpiralFunctionExample, self).__init__(Tensor(given_matrix))
 
class RandomLinearODEF(LinearODEF):
    """
    Class for a random LinearODEF.
    Inherits from LinearODEF.
    """
    def __init__(self):
        """
        Initialize RandomLinearODEF with a random weight matrix.
        """
        super(RandomLinearODEF, self).__init__(torch.randn(2, 2)/2.)
        
class TestODEF(ODEF):
    """
    Test class for a custom ODE function.
    Inherits from ODEF.
    """
    def __init__(self, A, B, x0):
        """
        Initialize TestODEF with provided matrices and initial value.

        Parameters:
        - A: Weight matrix for the A linear layer
        - B: Weight matrix for the B linear layer
        - x0: Initial value
        """
        super(TestODEF, self).__init__()
        self.A = nn.Linear(2, 2, bias=False)
        self.A.weight = nn.Parameter(A)
        self.B = nn.Linear(2, 2, bias=False)
        self.B.weight = nn.Parameter(B)
        self.x0 = nn.Parameter(x0)

    def forward(self, x, t):
        """
        Forward pass through the TestODEF.

        Parameters:
        - x: Input tensor
        - t: Time tensor

        Returns:
        - Output of the differential equation
        """
        xTx0 = torch.sum(x*self.x0, dim=1)
        dxdt = torch.sigmoid(xTx0) * self.A(x - self.x0) + torch.sigmoid(-xTx0) * self.B(x + self.x0)
        return dxdt
    
class NNODEF(ODEF):
    """
    Neural Network Ordinary Differential Equation Function.
    Inherits from ODEF.
    """
    def __init__(self, in_dim, hid_dim, time_invariant=False):
        """
        Initialize NNODEF.

        Parameters:
        - in_dim: Input dimension
        - hid_dim: Hidden dimension
        - time_invariant: Flag for time invariance
        """
        super(NNODEF, self).__init__()
        self.time_invariant = time_invariant

        if time_invariant:
            self.lin1 = nn.Linear(in_dim, hid_dim)
        else:
            self.lin1 = nn.Linear(in_dim+1, hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.lin3 = nn.Linear(hid_dim, in_dim)
        self.LRelu = torch.nn.leaky_relu(approximate='none', inplace=True)
    
    def forward(self, x, t):
        """
        Forward pass through NNODEF.

        Parameters:
        - x: Input tensor
        - t: Time tensor

        Returns:
        - Output tensor
        """
        if not self.time_invariant:
            x = torch.cat((x, t), dim=-1)

        h = self.LRelu(self.lin1(x))
        h = self.LRelu(self.lin2(h))
        out = self.lin3(h)
        return out
    
def to_np(x):
    """
    Convert tensor to numpy array.

    Parameters:
    - x: Input tensor

    Returns:
    - Numpy array
    """
    return x.detach().cpu().numpy()
    
def plot_trajectories(obs=None, times=None, trajs=None, save=None, figsize=(16, 8)):
    """
    Plot trajectories.

    Parameters:
    - obs: Observations
    - times: Time values
    - trajs: Trajectories
    - save: Path to save the plot
    - figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    if obs is not None:
        if times is None:
            times = [None] * len(obs)
        for o, t in zip(obs, times):
            o, t = o.detach().cpu().numpy(), t.detach().cpu().numpy()
            for b_i in range(o.shape[1]):
                ax.scatter(o[:, b_i, 0], o[:, b_i, 1], c=t[:, b_i, 0], cmap=cm.plasma)

    if trajs is not None: 
        for z in trajs:
            z = z.detach().cpu().numpy()
            ax.plot(z[:, 0, 0], z[:, 0, 1], lw=1.5)
        if save is not None:
            plt.savefig(save)
    
link = "data/user_try/NEDO_Leaky_ReLU"

def conduct_experiment(ode_true, ode_trained, n_steps, name, plot_freq=10):
    """
    Conduct the experiment.

    Parameters:
    - ode_true: True ODE
    - ode_trained: Trained ODE
    - n_steps: Number of steps
    - name: Name of the experiment
    - plot_freq: Frequency of plotting
    """
    z0 = Variable(torch.Tensor([[0.6, 0.3]]))

    t_max = 6.29*5
    n_points = 200

    index_np = np.arange(0, n_points, 1, dtype=int)
    index_np = np.hstack([index_np[:, None]])
    times_np = np.linspace(0, t_max, num=n_points)
    times_np = np.hstack([times_np[:, None]])

    times = torch.from_numpy(times_np[:, :, None]).to(z0)
    obs = ode_true(z0, times, return_whole_sequence=True).detach()
    obs = obs + torch.randn_like(obs) * 0.01

    # Get trajectory of random timespan 
    min_delta_time = 1.0
    max_delta_time = 5.0
    max_points_num = 32
    def create_batch():
        t0 = np.random.uniform(0, t_max - max_delta_time)
        t1 = t0 + np.random.uniform(min_delta_time, max_delta_time)

        idx = sorted(np.random.permutation(index_np[(times_np > t0) & (times_np < t1)])[:max_points_num])

        obs_ = obs[idx]
        ts_ = times[idx]
        return obs_, ts_

    # Train Neural ODE
    # Define the total number of steps for the progress bar
    total_steps = n_steps // plot_freq

    # Create a tqdm progress bar
    progress_bar = st.progress(0)
    
    optimizer = torch.optim.Adam(ode_trained.parameters(), lr=0.01)
    
    # Define a learning rate scheduler
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    for i in range(n_steps):
        obs_, ts_ = create_batch()

        z_ = ode_trained(obs_[0], ts_, return_whole_sequence=True)
        loss = F.mse_loss(z_, obs_.detach())

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        #scheduler.step(loss)
        
        if i % plot_freq == 0:
            z_p = ode_trained(z0, times, return_whole_sequence=True)

            plot_trajectories(obs=[obs], times=[times], trajs=[z_p], save=f"{link}/test_user_{i}.png")
            clear_output(wait=True)
            
            # Update the progress bar
            progress_bar.progress((i // plot_freq + 1) / total_steps)
    
    # Close the progress bar
    progress_bar.empty()
        
def main():
    ode_true = NeuralODE(SpiralFunctionExample())
    ode_trained = NeuralODE(RandomLinearODEF())
    conduct_experiment(ode_true, ode_trained, int(iteration.user_it), "linear")
    
if __name__ == '__main__':
    main()
    
    
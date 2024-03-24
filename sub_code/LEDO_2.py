import time

import os
import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optax


matplotlib.rcParams.update({"font.size": 30})

class Func(eqx.Module):
    """
    A class representing a function that scales the output of an MLP (Multi-Layer Perceptron).
    """
    
    scale: jnp.ndarray
    mlp: eqx.nn.MLP

    def __call__(self, t, y, args):
        """
        Compute the scaled output of the MLP given input `y`.
        
        Args:
        - t: Time parameter (not used in the function but kept for compatibility).
        - y: Input data.
        - args: Additional arguments (not used).
        
        Returns:
        - Scaled output of the MLP.
        """
        return self.scale * self.mlp(y)
    
    
class LatentODE(eqx.Module):
    """
    A class representing a Latent Ordinary Differential Equation (ODE) model.
    """
    func: Func
    rnn_cell: eqx.nn.GRUCell

    hidden_to_latent: eqx.nn.Linear
    latent_to_hidden: eqx.nn.MLP
    hidden_to_data: eqx.nn.Linear

    hidden_size: int
    latent_size: int

    def __init__(
        self, *, data_size, hidden_size, latent_size, width_size, depth, key, **kwargs
    ):
        """
        Initialize the LatentODE model.
        
        Args:
        - data_size: Size of the input data.
        - hidden_size: Size of the hidden layer.
        - latent_size: Size of the latent space.
        - width_size: Width of the MLP layers.
        - depth: Depth of the MLP layers.
        - key: PRNGKey for initialization.
        """
        super().__init__(**kwargs)

        mkey, gkey, hlkey, lhkey, hdkey = jr.split(key, 5)

        scale = jnp.ones(())
        mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            final_activation=jnn.tanh,
            key=mkey,
        )
        self.func = Func(scale, mlp)
        self.rnn_cell = eqx.nn.GRUCell(data_size + 1, hidden_size, key=gkey)

        self.hidden_to_latent = eqx.nn.Linear(hidden_size, 2 * latent_size, key=hlkey)
        self.latent_to_hidden = eqx.nn.MLP(
            latent_size, hidden_size, width_size=width_size, depth=depth, key=lhkey
        )
        self.hidden_to_data = eqx.nn.Linear(hidden_size, data_size, key=hdkey)

        self.hidden_size = hidden_size
        self.latent_size = latent_size

    # Encoder of the VAE
    def _latent(self, ts, ys, key):
        """
        Encoder of the Variational Autoencoder (VAE) for the LatentODE model.
        
        Args:
        - ts: Time points.
        - ys: Input data.
        - key: PRNGKey for randomness.
        
        Returns:
        - Latent representation.
        - Mean and standard deviation of the latent representation.
        """
        data = jnp.concatenate([ts[:, None], ys], axis=1)
        hidden = jnp.zeros((self.hidden_size,))
        for data_i in reversed(data):
            hidden = self.rnn_cell(data_i, hidden)
        context = self.hidden_to_latent(hidden)
        mean, logstd = context[: self.latent_size], context[self.latent_size :]
        std = jnp.exp(logstd)
        latent = mean + jr.normal(key, (self.latent_size,)) * std
        return latent, mean, std

    # Decoder of the VAE
    def _sample(self, ts, latent):
        """
        Decoder of the VAE for the LatentODE model.
        
        Args:
        - ts: Time points.
        - latent: Latent representation.
        
        Returns:
        - Sampled data.
        """
        dt0 = 0.4  # selected as a reasonable choice for this problem
        y0 = self.latent_to_hidden(latent)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            ts[0],
            ts[-1],
            dt0,
            y0,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return jax.vmap(self.hidden_to_data)(sol.ys)

    @staticmethod
    def _loss(ys, pred_ys, mean, std):
        """
        Compute the loss for the LatentODE model.
        
        Args:
        - ys: True data.
        - pred_ys: Predicted data.
        - mean: Mean of the latent representation.
        - std: Standard deviation of the latent representation.
        
        Returns:
        - Total loss.
        """
        # -log p_θ with Gaussian p_θ
        reconstruction_loss = 0.5 * jnp.sum((ys - pred_ys) ** 2)
        # KL(N(mean, std^2) || N(0, 1))
        variational_loss = 0.5 * jnp.sum(mean**2 + std**2 - 2 * jnp.log(std) - 1)
        return reconstruction_loss + variational_loss

    # Run both encoder and decoder during training.
    def train(self, ts, ys, *, key):
        """
    Trains the LatentODE model using the encoder and decoder.
    
    Args:
    - ts: Time points.
    - ys: Input data.
    - key: PRNGKey for randomness.
    
    Returns:
    - Loss value.
    """
        latent, mean, std = self._latent(ts, ys, key)
        pred_ys = self._sample(ts, latent)
        return self._loss(ys, pred_ys, mean, std)

    # Run just the decoder during inference.
    def sample(self, ts, *, key):
        """
    Generates predictions using only the decoder of the LatentODE model.
    
    Args:
    - ts: Time points.
    - key: PRNGKey for randomness.
    
    Returns:
    - Predicted data.
    """
        latent = jr.normal(key, (self.latent_size,))
        return self._sample(ts, latent)
    
def get_data(dataset_size, *, key):
    """
    Generates synthetic data for training the LatentODE model.
    
    Args:
    - dataset_size: Number of data points to generate.
    - key: PRNGKey for randomness.
    
    Returns:
    - Time points and corresponding data.
    """
    ykey, tkey1, tkey2 = jr.split(key, 3)

    y0 = jr.normal(ykey, (dataset_size, 2))

    t0 = 0
    t1 = 2 + jr.uniform(tkey1, (dataset_size,))
    ts = jr.uniform(tkey2, (dataset_size, 20)) * (t1[:, None] - t0) + t0
    ts = jnp.sort(ts)
    dt0 = 0.1

    def func(t, y, args):
        return jnp.array([[-2, 0], [0, -3]]) @ y

    def solve(ts, y0):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(func),
            diffrax.Tsit5(),
            ts[0],
            ts[-1],
            dt0,
            y0,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys

    ys = jax.vmap(solve)(ts, y0)

    return ts, ys


def dataloader(arrays, batch_size, *, key):
    """
    Creates a generator that yields batches of data.
    
    Args:
    - arrays: List of data arrays.
    - batch_size: Size of each batch.
    - key: PRNGKey for randomness.
    
    Yields:
    - Batches of data.
    """
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while start < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def main(
    dataset_size=10000,
    batch_size=256,
    lr=1e-2,
    steps=250,
    save_every=50,
    hidden_size=16,
    latent_size=16,
    width_size=16,
    depth=2,
    seed=5678,
):
    """
    Main function to train the LatentODE model and visualize the results.
    
    Args:
    - dataset_size: Number of data points to generate.
    - batch_size: Size of each batch.
    - lr: Learning rate for optimization.
    - steps: Number of training steps.
    - save_every: Interval for saving plots.
    - hidden_size: Size of the hidden layer.
    - latent_size: Size of the latent space.
    - width_size: Width of the MLP layers.
    - depth: Depth of the MLP layers.
    - seed: Random seed.
    """
    
    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key, train_key, sample_key = jr.split(key, 5)

    ts, ys = get_data(dataset_size, key=data_key)

    model = LatentODE(
        data_size=ys.shape[-1],
        hidden_size=hidden_size,
        latent_size=latent_size,
        width_size=width_size,
        depth=depth,
        key=model_key,
    )

    @eqx.filter_value_and_grad
    def loss(model, ts_i, ys_i, key_i):
        batch_size, _ = ts_i.shape
        key_i = jr.split(key_i, batch_size)
        loss = jax.vmap(model.train)(ts_i, ys_i, key=key_i)
        return jnp.mean(loss)

    @eqx.filter_jit
    def make_step(model, opt_state, ts_i, ys_i, key_i):
        value, grads = loss(model, ts_i, ys_i, key_i)
        key_i = jr.split(key_i, 1)[0]
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return value, model, opt_state, key_i

    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    
    # Create folder for saving plots
    os.makedirs("latent_plots", exist_ok=True)

    # Plot results
    num_plots = 1 + (steps - 1) // save_every
    if ((steps - 1) % save_every) != 0:
        num_plots += 1
    fig, axs = plt.subplots(1, num_plots, figsize=(num_plots * 8, 8))
    axs[0].set_ylabel("x")
    axs = iter(axs)
    for step, (ts_i, ys_i) in zip(
        range(steps), dataloader((ts, ys), batch_size, key=loader_key)
    ):
        start = time.time()
        value, model, opt_state, train_key = make_step(
            model, opt_state, ts_i, ys_i, train_key
        )
        end = time.time()
        print(f"Step: {step}, Loss: {value}, Computation time: {end - start}")

        # Create a new figure for each plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_ylabel("x")
        ax.set_xlabel("t")

        # Sample over a longer time interval than we trained on. The model will be
        # sufficiently good that it will correctly extrapolate!
        sample_t = jnp.linspace(0, 12, 300)
        sample_y = model.sample(sample_t, key=sample_key)
        sample_t = np.asarray(sample_t)
        sample_y = np.asarray(sample_y)
        ax.plot(sample_t, sample_y[:, 0], label='x1')
        ax.plot(sample_t, sample_y[:, 1], label='x2')
        ax.legend()

        
        # Save plot in the "latent_plots" folder
        plt.savefig(os.path.join("data/user_try/latent_plots_2", f"latent_ode_{step}.png"))
        plt.close()
    

if __name__ == '__main__':
        main()
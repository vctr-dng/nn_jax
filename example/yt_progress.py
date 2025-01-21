# %%
import jax as jax
import jax.numpy as jnp

from activation_function import ReLU
from layer import Dense
from utils import RANDOM_KEY

# %% Data generation

# Generate x^2 with some noise

n_samples = 100
x = jnp.linspace(-1, 1, n_samples)
y = x**2 + 0.1 * jax.random.normal(RANDOM_KEY, (n_samples,))

# %%

nn_input = jnp.vstack((x[:5], y[:5])).T
layer1 = Dense(2, 5)
activation1 = ReLU()
l1_output = layer1.forward(nn_input)
a1_output = activation1.forward(l1_output)
print(a1_output)

import jax.random as random

global RANDOM_SEED
RANDOM_SEED = 0
global RANDOM_KEY
RANDOM_KEY = random.PRNGKey(RANDOM_SEED)


def set_seed(seed: int):
    global RANDOM_SEED
    RANDOM_SEED = seed


def update_random_key():
    global RANDOM_KEY
    RANDOM_KEY = random.PRNGKey(RANDOM_SEED)

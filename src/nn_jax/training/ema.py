from jax import tree_util


def initialize_ema[ParametersT](parameters: ParametersT) -> ParametersT:
    """Create an EMA pytree initialized to the current parameters."""
    return tree_util.tree_map(lambda parameter: parameter, parameters)


def update_ema[ParametersT](
    ema_parameters: ParametersT,
    parameters: ParametersT,
    decay: float,
) -> ParametersT:
    """Update a parameter pytree with exponential moving averages."""
    if not 0 <= decay <= 1:
        raise ValueError("decay must be in [0, 1]")

    return tree_util.tree_map(
        lambda ema_parameter, parameter: (
            decay * ema_parameter + (1 - decay) * parameter
        ),
        ema_parameters,
        parameters,
    )

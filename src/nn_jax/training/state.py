from jax import Array, tree_util


@tree_util.register_pytree_node_class
class TrainState[ParametersT, OptimizerStateT]:
    """All mutable inputs to a training step, represented explicitly."""

    def __init__(
        self,
        parameters: ParametersT,
        optimizer_state: OptimizerStateT,
        ema_parameters: ParametersT,
        rng_key: Array,
        step: Array,
    ):
        self.parameters = parameters
        self.optimizer_state = optimizer_state
        self.ema_parameters = ema_parameters
        self.rng_key = rng_key
        self.step = step

    def tree_flatten(self):
        children = (
            self.parameters,
            self.optimizer_state,
            self.ema_parameters,
            self.rng_key,
            self.step,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        parameters, optimizer_state, ema_parameters, rng_key, step = children
        return cls(parameters, optimizer_state, ema_parameters, rng_key, step)

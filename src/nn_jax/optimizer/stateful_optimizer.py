from abc import ABC, abstractmethod


class StatefulOptimizer[ParametersT, StateT](ABC):
    """An optimizer whose update state is passed explicitly."""

    @abstractmethod
    def init(self, parameters: ParametersT) -> StateT:
        """Create optimizer state for a parameter pytree."""

    @abstractmethod
    def update(
        self,
        parameters: ParametersT,
        gradients: ParametersT,
        state: StateT,
    ) -> tuple[ParametersT, StateT]:
        """Return updated parameters and optimizer state."""

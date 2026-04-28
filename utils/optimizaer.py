from abc import ABC, abstractmethod
import numpy as np
from models.nn.parameter import Parameter


class Optimizer(ABC):
    @abstractmethod
    def optimize(self, parameter: Parameter):
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, l2_penalty=0.0):
        self.learning_rate = learning_rate
        self.l2_penalty = l2_penalty

    def optimize(self, parameter: Parameter):
        grad = parameter.gradient
        if self.l2_penalty > 0.0:
            grad = grad + (self.l2_penalty * parameter.value)

        parameter.value -= self.learning_rate * grad


class SGDMomentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, l2_penalty=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2_penalty = l2_penalty
        self.velocities = {}

    def optimize(self, parameter: Parameter):
        param_id = parameter.id
        if param_id not in self.velocities:
            self.velocities[param_id] = np.zeros_like(parameter.value)

        grad = parameter.gradient
        if self.l2_penalty > 0.0:
            grad = grad + (self.l2_penalty * parameter.value)

        self.velocities[param_id] = (self.momentum * self.velocities[param_id]) + (self.learning_rate * grad)

        parameter.value -= self.velocities[param_id]

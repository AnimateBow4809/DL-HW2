import numpy as np
from abc import ABC, abstractmethod

from models.nn.parameter import Parameter
from utils.optimizaer import Optimizer


class Layer(ABC):
    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input_data):
        pass

    @abstractmethod
    def backward(self, output_gradient):
        pass

    @abstractmethod
    def update(self, optimizer: Optimizer):
        pass


class MLP(Layer):

    def __init__(self, input_size, output_size, layer_name="MLP",
                 weight_mu=0.0, weight_sigma=None, bias_value=0.0):
        super().__init__()
        if weight_sigma is None:
            sigma = np.sqrt(2. / input_size)
        else:
            sigma = weight_sigma

        w_val = np.random.normal(loc=weight_mu, scale=sigma, size=(input_size, output_size))
        b_val = np.full((1, output_size), fill_value=bias_value)

        self.weights = Parameter(value=w_val, name=f"{layer_name}_weights")
        self.biases = Parameter(value=b_val, name=f"{layer_name}_biases")


    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights.value) + self.biases.value
        return self.output

    def backward(self, output_gradient):
        self.weights.gradient = np.dot(self.input.T, output_gradient)
        self.biases.gradient = np.sum(output_gradient, axis=0, keepdims=True)
        input_gradient = np.dot(output_gradient, self.weights.value.T)
        return input_gradient

    def update(self, optimizer: Optimizer):
        optimizer.optimize(self.weights)
        optimizer.optimize(self.biases)


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        self.input = input_data
        self.output = np.maximum(0, self.input)
        return self.output

    def backward(self, output_gradient):
        input_gradient = output_gradient.copy()
        input_gradient[self.input <= 0] = 0
        return input_gradient

    def update(self, optimizer: Optimizer):
        pass


class Softmax(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        self.input = input_data
        shifted_input = input_data - np.max(input_data, axis=1, keepdims=True)
        exps = np.exp(shifted_input)
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient):
        input_gradient = np.empty_like(output_gradient)
        for i, (single_output, single_grad) in enumerate(zip(self.output, output_gradient)):
            single_output = single_output.reshape(-1, 1)
            jacobian = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            input_gradient[i] = np.dot(jacobian, single_grad)

        return input_gradient

    def update(self, optimizer: Optimizer):
        pass

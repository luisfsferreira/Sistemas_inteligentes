from abc import abstractmethod
from typing import Union

import numpy as np

from si.neural_networks.layers import Layer


class ActivationLayer(Layer):
    """
    Base class for activation layers.
    """

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error: float) -> Union[float, np.ndarray]:
        """
        Perform backward propagation on the given output error.

        Parameters
        ----------
        output_error: float
            The output error of the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output error of the layer.
        """
        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output of the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Derivative of the activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The derivative of the activation function.
        """
        raise NotImplementedError

    def output_shape(self) -> tuple:
        """
        Returns the output shape of the layer.

        Returns
        -------
        tuple
            The output shape of the layer.
        """
        return self._input_shape

    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return 0


class SigmoidActivation(ActivationLayer):
    """
    Sigmoid activation function.
    """

    def activation_function(self, input: np.ndarray):
        """
        Sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return 1 / (1 + np.exp(-input))

    def derivative(self, input: np.ndarray):
        """
        Derivative of the sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return self.activation_function(input) * (1 - self.activation_function(input))


class ReLUActivation(ActivationLayer):
    """
    ReLU activation function.
    """

    def activation_function(self, input: np.ndarray):
        """
        ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return np.maximum(0, input)

    def derivative(self, input: np.ndarray):
        """
        Derivative of the ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return np.where(input > 0, 1, 0)
    
class TanhActivation:
    """
    TanhActivation class represents the hyperbolic tangent activation function in neural networks.
    """

    def activation_function(self, input: np.ndarray):
        """
        Applies the hyperbolic tangent activation function to the input array.

        Parameters
        ----------
        input : np.ndarray
            The input array to which the activation function is applied.

        Returns
        -------
        np.ndarray
            The output array after applying the hyperbolic tangent activation function.
        """
        return np.exp(input) + np.exp(-input)/(np.exp(input) - np.exp(-input))

    def derivative(self, input: np.ndarray):
        """
        Computes the derivative of the hyperbolic tangent activation function with respect to the input.

        Parameters
        ----------
        input : np.ndarray
            The input array for which the derivative is calculated.

        Returns
        -------
        np.ndarray
            The derivative of the hyperbolic tangent activation function.
        """
        return 1 - self.activation_function(input)**2

class SoftmaxActivation:
    """
    SoftmaxActivation class represents the softmax activation function in neural networks.
    """
    
    def activation_function(self, input: np.ndarray):
        """
        Applies the softmax activation function to the input array.

        Parameters
        ----------
        input : np.ndarray
            The input array to which the activation function is applied.

        Returns
        -------
        np.ndarray
            The output array after applying the softmax activation function.
        """
        exp_values = np.exp(input - np.max(input, axis=-1, keepdims=True))  # Subtraimos o valor máximo calculado de cada elemento do array de entrada. Isso é feito para evitar grandes valores exponenciais. keepdims = Aqui mantém as dimensões do resultado, para que possa ser usado diretamente na operação seguinte. Se o array for bidimensional, o último eixo (axis = -1) será a coluna. 
        return exp_values / np.sum(exp_values, axis=-1, keepdims=True) # Calcula a soma dos valores exponenciais ao longo do último eixo (novamente, ao longo das colunas)


    def derivative(self, input: np.ndarray):
        """
        Computes the derivative of the softmax activation function with respect to the input.

        Parameters
        ----------
        input : np.ndarray
            The input array for which the derivative is calculated.

        Returns
        -------
        np.ndarray
            The derivative of the softmax activation function.
        """
        return self.activation_function(input) * (1- self.activation_function(input))
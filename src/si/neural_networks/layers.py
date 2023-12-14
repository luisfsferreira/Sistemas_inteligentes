import copy
from abc import abstractmethod

import numpy as np

from si.neural_networks.optimizers import Optimizer


class Layer:
    """
    Base class for neural network layers.
    """

    @abstractmethod
    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input, i.e., computes the output of a layer for a given input.
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
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, output_error: float) -> float:
        """
        Perform backward propagation on the given output error, i.e., computes dE/dX for a given dE/dY and update
        parameters if any.
        Parameters
        ----------
        output_error: float
            The output error of the layer.
        Returns
        -------
        float
            The input error of the layer.
        """
        raise NotImplementedError

    def layer_name(self) -> str:
        """
        Returns the name of the layer.
        Returns
        -------
        str
            The name of the layer.
        """
        return self.__class__.__name__

    @abstractmethod
    def output_shape(self) -> tuple:
        """
        Returns the shape of the output of the layer.
        Returns
        -------
        tuple
            The shape of the output of the layer.
        """
        raise NotImplementedError

    def set_input_shape(self, shape: tuple):
        """
        Sets the shape of the input to the layer.
        Parameters
        ----------
        shape: tuple
            The shape of the input to the layer.
        """
        self._input_shape = shape

    def input_shape(self) -> tuple:
        """
        Returns the shape of the input to the layer.
        Returns
        -------
        tuple
            The shape of the input to the layer.
        """
        return self._input_shape

    @abstractmethod
    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.
        Returns
        -------
        int
            The number of parameters of the layer.
        """
        raise NotImplementedError


class DenseLayer(Layer):
    """
    Dense layer of a neural network.
    """

    def __init__(self, n_units: int, input_shape: tuple = None):
        """
        Initialize the dense layer.
        Parameters
        ----------
        n_units: int
            The number of units of the layer, aka the number of neurons, aka the dimensionality of the output space.
        input_shape: tuple
            The shape of the input to the layer.
        """
        super().__init__()
        self.n_units = n_units
        self._input_shape = input_shape

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

    def initialize(self, optimizer: Optimizer) -> 'DenseLayer':
        # initialize weights from a 0 centered uniform distribution [-0.5, 0.5)
        self.weights = np.random.rand(self.input_shape()[0], self.n_units) - 0.5
        # initialize biases to 0
        self.biases = np.zeros((1, self.n_units))
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
        return self

    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.
        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

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
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward_propagation(self, output_error: np.ndarray) -> float:
        """
        Perform backward propagation on the given output error.
        Computes the dE/dW, dE/dB for a given output_error=dE/dY.
        Returns input_error=dE/dX to feed the previous layer.
        Parameters
        ----------
        output_error: numpy.ndarray
            The output error of the layer.
        Returns
        -------
        float
            The input error of the layer.
        """
        # computes the layer input error (the output error from the previous layer),
        # dE/dX, to pass on to the previous layer
        input_error = np.dot(output_error, self.weights.T)
        # computes the weight error: dE/dW = X.T * dE/dY
        weights_error = np.dot(self.input.T, output_error)
        # computes the bias error: dE/dB = dE/dY
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        # updates parameters
        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)
        return input_error

    def output_shape(self) -> tuple:
        """
        Returns the shape of the output of the layer.
        Returns
        -------
        tuple
            The shape of the output of the layer.
        """
        return (self.n_units,)
    
class Dropout(Layer):
    """
        Implementation of a Dropout layer in a neural network.

        Parameters
        ----------
        probability : float
            The probability of deactivating a neuron during training. A value between 0 and 1.

    """

    def __init__(self, probability:float):
        """
            Initializes the Dropout layer.

            Parameters
            ----------
            probability : float
                The probability of deactivating a neuron during training.
        """    
        super().__init__()
        self.probability = probability
        self.mask = None
        self.input = None
        self.output = None
        
    def forward_propagation (self, input:np.ndarray, training: bool)-> np.ndarray:
            """
            Performs forward propagation on the layer during training or inference.

            Parameters
            ----------
            input : np.ndarray
                The input to the layer.
            training : bool
                Indicates whether the layer is in training mode.

            Returns
            -------
            np.ndarray
                The output of the layer after applying Dropout.
            """
            if training:
                scale = 1/(1-self.probability)
                self.mask = np.random.binomial(1, 1 - self.probability, size=input.shape)  # matriz binária que pode conter valores de 0 ou 1 (neuronio desativo e ativo, respetivamente)
                self.input = input
                self.output = self.input * self.mask * scale                                   # esta matriz tem valores positivos, negativos e nulos. Nulos representa o neuronio desativado, positivo significa que o neuronio contribuiu positivamente para a saída e negativo significa que contribui negativamente.

            else:
                self.input = input
                self.output = self.input

            return self.output
    def backward_propagation (self, output_error:np.ndarray)-> np.ndarray:
            """
            Performs backward propagation on the layer.

            Parameters
            ----------
            output_error : np.ndarray
                The output error of the layer.

            Returns
            -------
            np.ndarray
                The input error of the layer.
            """
            return self.mask * output_error  #Cada valor da matriz representa se o erro está sendo propagado de volta para a camada anterior após a aplicação do dropout

        
    def output_shape(self) -> tuple:
            """
            Returns the shape of the layer's output.

            Returns
            -------
            tuple
                The shape of the layer's output.
            """
            if self.output is None:
                print(f"Warning: {self.layer_name()} output is None. Check forward_propagation.")
                return tuple()  # Return an empty tuple for now, you might want to adjust this based on your needs
            else:
                return self.output.shape
        
    def parameters(self) -> int:
            """
            Returns the number of parameters in the layer.

            Returns
            -------
            int
                The number of parameters in the layer. In this case, it is always 0 as Dropout has no parameters.
            """
            return 0
        
if __name__== "__main__":
    
    # from si.neural_networks.layers import Dropout
    
    # Create an instance of the Dropout layer with a probability of 0.5
    dropout_layer = Dropout(probability=0.5)

    # Generate a random input array
    random_input = np.random.randn(4, 4)  # valores podem ser qualquer número real, positivo ou negativo, com uma probabilidade maior de valores próximos a 0

    # Test forward propagation during training
    output_train = dropout_layer.forward_propagation(random_input, training=True) # esta matriz tem valores positivos, negativos e nulos. Nulos representa o neuronio desativado, positivo significa que o neuronio contribuiu positivamente para a saída e negativo significa que contribui negativamente.
    print("Output during training:")
    print(output_train)

    # Test forward propagation during inference
    output_inference = dropout_layer.forward_propagation(random_input, training=False)
    print("\nOutput during inference:")
    print(output_inference)

    # Test backward propagation with a dummy output_error
    dummy_output_error = np.ones_like(output_train)                            #Simula um cenário onde todas as saídas da camada de dropout contribuíram de igual forma para a função de perda. Matriz com valores 1.
    backward_result = dropout_layer.backward_propagation(dummy_output_error)   #Utilizo o erro calculado em cima para fazer a backpropagation.
    print("\nBackward propagation result:")
    print(backward_result)                                             #Cada valor da matriz representa se o erro está sendo propagado de volta para a camada anterior após a aplicação do dropout, se for 1 está,se for 0 não.

    # Output shape and parameters
    print("\nOutput shape:", dropout_layer.output_shape())
    print("Parameters:", dropout_layer.parameters())


        
import numpy as np

class Dropout:
    def __init__(self, probability: float):
        self.probability = probability
        self.mask = None
        self.input = None
        self.output = None

    def forward_propagation(self, input: np.ndarray, training: bool):
        if training:
            scale = 1 / (1 - self.probability)
            self.mask = np.random.binomial(1, 1 - self.probability, size=input.shape)
            print("aa:", self.mask) 	
            self.output = input * self.mask * scale
            print(self.output)
            return self.output
            
        else:
            self.output = input
            return self.output

    def backward_propagation(self, output_error):
        return self.mask * output_error
    
    def output_shape(self):
        return self.output.shape
        
    def parameters(self):
        return 0
# Exemplo simples de uso
if __name__ == "__main__":
    # Cria uma instância da camada de Dropout com uma probabilidade de 0.5
    dropout_layer = Dropout(probability=0.5)


    # Gera uma matriz de entrada aleatória
    random_input = np.random.randn(4, 4)
    # print(random_input)

    # Testa a propagação para frente durante o treinamento
    output_train = dropout_layer.forward_propagation(random_input, training=True)
    # print("Output during training:", output_train)

    # Simula um cenário onde todas as saídas da camada de dropout contribuíram de igual forma para a função de perda
    dummy_output_error = np.ones_like(output_train)   #O
    backward_result = dropout_layer.backward_propagation(dummy_output_error)   
    print("dummy:", dummy_output_error)
    print("\nBackward propagation result:\n", backward_result)

    # Output shape and parameters
    print("\nOutput shape:", dropout_layer.output_shape())
    print("Parameters:", dropout_layer.parameters())
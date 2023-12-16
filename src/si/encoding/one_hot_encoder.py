import numpy as np

class OneHotEncoder:
    def __init__(self, padder:str, max_length: int):
        """
        Initialize the OneHotEncoder.

        Parameters:
        - padder (str): The character used for padding.
        - max_length (int): The maximum length of sequences after padding.
        """
        self.padder = padder
        self.max_length = max_length

        self.alphabet = None
        self.char_to_index = {}
        self.index_to_char = {}

    def fit (self, data:list):
        """
        Fit the encoder on the given data.

        Parameters:
        - data (list): List of input sequences.
        """

        self.alphabet = set(''.join(data)) # Conjunto único de caracteres presentes nas sequências de entrada.


        self.char_to_index = {char: index for index, char in enumerate(self.alphabet)}
        self.index_to_char = {index: char for index, char in enumerate(self.alphabet)}
 
        if self.max_length is None:
            self.max_length = max(len(seq) for seq in data)

    def transform(self, sequence: str):
        """
        Transform a single sequence into its one-hot encoded representation.

        Parameters:
        - sequence (str): The input sequence.

        Returns:
        - list: List of one-hot encoded vectors.
        """
        one_hot_encoding = []
    
        # Trim the sequences to max_length
        seq_max = sequence[:self.max_length]  #Obtemos o tamanho máximo da nossa sequencia, todas os caracteres que passarem deste comprimentos, sao preenchidos 

        # Pad the sequences with the padding character
        seq = seq_max.ljust(self.max_length, self.padder)  #str.ljust(width, fillchar). width: Comprimento máximo que iremos impor às sequências de entrada. fillchar: É o caracter de preenchimento que será usado ( _ ). Se este argumento não for fornecido, é utilizado um espaço em branco como caractere padrão.

        for char in seq:
            #if char in self.char_to_index.items():   # o char é uma tupla da key (caracter) e do valor (numero)
            if char in self.char_to_index:            # char aqui é a key ou seja o caracter
                one_hot_vector = [0] * len(self.alphabet)              #criar vetores com apenas 0 do tamanho dos caracteres unicos
                one_hot_vector[self.char_to_index[char]] = 1           # Obtém o índice do caractere atual na sequência através da representacao do caracter [char]. Por exemplo, se o alfabeto for {'a', 'b', 'c', 'd'}, e o caractere atual (char) for 'c', e supondo que self.char_to_index seja algo como {'a': 0, 'b': 1, 'c': 2, 'd': 3}, então self.char_to_index[char] seria 2.
                one_hot_encoding.append(one_hot_vector)

        return one_hot_encoding

    def fit_transform (self, data: list):
        """
        Fit the encoder on the given data and transform it.

        Parameters:
        - data (list): List of input sequences.

        Returns:
        - list: List of lists containing one-hot encoded vectors.
        """
        self.fit(data)
        result = []
        for seq in data:
            result.append(self.transform(seq))
        return result
        
    def inverse_transform(self, encoded_matrices: list):
        """
        Inverse transform the one-hot encoded matrices back to sequences.

        Parameters:
        - encoded_matrices (list): List of lists containing one-hot encoded vectors.

        Returns:
        - list: List of decoded sequences.
        """
        return [
        ''.join(self.index_to_char[np.argmax(char_one_hot)]     #o argmax encontra o índice do valor mais alto, depois temos self.index_to_char[x] que vai nos dar o value (caracter) correspondente à key em x
            for char_one_hot in sample                          #por cada vetor na matriz
            if self.index_to_char[np.argmax(char_one_hot)] != self.padder)  #para evitar conflitos com os valores de preenchimentos
        for sample in encoded_matrices]                         #por cada matriz na lista de matrizes

if __name__ == "__main__":

    data = ["hello", "world", "blue"]

    encoder = OneHotEncoder(padder='_', max_length=8)

    encoded_data = encoder.fit_transform(data)
    print()
    print("Alphabet:", encoder.alphabet)	
    print()

    print("Sequências Codificadas:")
    for seq in encoded_data:
        print(seq)
    print()

    decoded_sequences = encoder.inverse_transform(encoded_data)
    print("Sequências descodificadas:", decoded_sequences)





            


        


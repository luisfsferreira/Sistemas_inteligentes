from typing import Tuple

import numpy as np

from si.data.dataset import Dataset
from icecream import ic
from si.io.csv_file import read_csv 


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test

def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the dataset into training and testing sets in a stratified manner.

    Parameters
    ----------
    dataset: Dataset
        The dataset to be split.
    test_size: float
        The proportion of the dataset to be used for testing.
    random_state: int
        The seed for random number generation.

    Returns
    -------
    train: Dataset
        The training dataset.
    test: Dataset
        The testing dataset.
    """
    
    np.random.seed(random_state)
    
    unique_labels, label_counts = np.unique(dataset.y, return_counts=True)   # O np.unique é usada para obter os rotulos das classes sem haver duplicações e diz-nos as contagens para cada classe se for preciso usar. EX: A classe 2 têm dois rotulos, entao tenho duas amostras ([0,1,1,1,2,2])

    train_idxs = []
    test_idxs = []

    for label in unique_labels:
        class_indices = np.where(dataset.y == label)[0]  # Obtenho o indice para quando a condição é verdadeira e retiro apenas o primeiro elemento da tupla, sendo algo deste genero a saida (array([0, 2, 4]),))
        n_test = int(len(class_indices) * test_size)    # Obtenho o numero de rotulos/amostras para a classe(label) para ser incluidas no test set
        np.random.shuffle(class_indices)  # Vamos obter o numero de indices aleatorios consoante o tamanho do n_test. Altera o array inicial
        # idx_rand = np.random.permutation(class_indices)  # Vamos obter o numero de indices aleatorios consoante o tamanho do n_test. Cria um novo array, sem alterar o original.

        idx_select = class_indices[:n_test]   #seleciona os index consoante o tamanho do n_test
        test_idxs.extend(idx_select)

        # Exemplo da diferença entre o uso do extend e do uso append

        # test_idxs_append = []
        # test_idxs_append.append([1, 2, 3])  # Adiciona uma lista de índices
        # test_idxs_append.append([4, 5, 6])  # Adiciona outra lista de índices
        # Resultado: [[1, 2, 3], [4, 5, 6]]

        # # Exemplo com extend
        # test_idxs_extend = []
        # test_idxs_extend.extend([1, 2, 3])  # Adiciona os índices individualmente
        # test_idxs_extend.extend([4, 5, 6])  # Adiciona os índices individualmente
        # Resultado: [1, 2, 3, 4, 5, 6]

        idx_remaining = class_indices[n_test:] # seleciona os index maiores que o tamanho do n_test
        train_idxs.extend(idx_remaining)
    
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label) # dataset.y[train_idxs] são os rótulos correspondentes aos índices no dataset.x[train_idxs]
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label =dataset.label ) # dataset.y[test_idxs] são os rótulos correspondentes aos índices no dataset.x[test_idxs]

    return train, test


if __name__ == '__main__':
        
        path= "C:/Users/luis-/Documents/GitHub/Sistemas_inteligentes/datasets/iris/iris.csv"
        dataset =  read_csv(path, features=True, label=True)
        dataset.shape()  # 150 linhas e 4 colunas
        dataset.get_classes() # 3 classes

        train, test = stratified_train_test_split(dataset)
        print(train.shape())
        print(test.shape())

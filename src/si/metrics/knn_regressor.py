from typing import Callable, Union
from si.data.dataset import Dataset
from si.statistics.euclidean_distance import euclidean_distance
from si.metrics.mse import mse
from si.metrics.rmse import rmse
import numpy as np
import pandas as pd
from si.model_selection.split import train_test_split
from si.io.csv_file import read_csv



class KNNRegressor:
    """
    KNNRegressor class for regression problems using the k-nearest neighbors algorithm.

    Parameters
    ----------
    k: int, default = 1
        The number of k nearest examples to consider.
    distance: Callable, default = euclidean_distance
        A function that calculates the distance between a sample and the samples in the training dataset.

    Attributes
    ----------
    k: int
        The number of k nearest examples to consider.
    distance: Callable
        A function that calculates the distance between a sample and the samples in the training dataset.
    dataset: Dataset
        Stores the training dataset.

    """

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance):
        """
        Initializes the KNNRegressor.

        Parameters
        ----------
        k: int, default = 1
            The number of k nearest examples to consider.
        distance: Callable, default = euclidean_distance
            A function that calculates the distance between a sample and the samples in the training dataset.
        """
        self.k = k
        self.distance = distance
        self.dataset = None

    def fit(self, dataset:Dataset) -> "KNNRegressor":
        """
        Stores the training dataset.

        Parameters
        ----------
        dataset: Dataset
            The training dataset.
        """
        self.dataset = dataset
        return self
    
    def _get_closest_label(self, sample: np.ndarray) -> Union[int, str]:
        """
        It returns the closest label of the given sample

        Parameters
        ----------
        sample: np.ndarray
            The sample to get the closest label 

        Returns
        -------
        closest_label : Union[int, str]
            The average label value of the k-nearest neighbors.
        """
        all_distances = self.distance (sample, self.dataset.X) #Calcular a distância entre cada amostra e várias amostras no conjunto de treinamento:
        KNN = np.argsort(all_distances)[:self.k]  # O argsort ordena o array inicial por ordem crescente, através do indice. Ex. Array original: [3 1 4 1 5 9 2 6 5 3 5], Índices ordenados: [ 1  3  6  0  9  4  8 10  2  7  5]. Assim podemos obter a distância mais pequena, que fica no inicio.
        KNN_label = self.dataset.y[KNN]     #Label da amostra de cima
        unique_labels, label_counts = np.unique(KNN_label, return_counts=True)   # O np.unique é usada para obter os rotulos das classes sem haver duplicações e diz-nos as contagens para cada classe se for preciso usar. EX: A classe 2 têm dois rotulos, entao tenho duas amostras ([0,1,1,1,2,2])
        closest_label = np.average(KNN_label) # Em problemas de regressão, os rótulos representam valores numéricos. Como tal, obtemos a média das labels dos k vizinhos 
                              # O valor numérico previsto está associado a uma nova amostra (característica) e é determinado com base nos valores das k amostras mais próximas.
        return closest_label 

    def predict(self, dataset: Dataset)-> np.ndarray:
        """
        Predicts the labels/values for a given dataset using the KNNRegressor algorithm.

        Parameters
        ----------
        dataset: Dataset
            The dataset for which predictions are to be made.

        Returns
        -------
        predictions: np.ndarray
            An array containing the predicted labels/values for each sample in the input dataset.
        """
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X) # aplica o get_closest_label em todas as linhas do dataset

    def score(self, dataset: Dataset) -> float:
        """
        It returns the rmse of the model on the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on

        Returns
        -------
        rmse: float
            The rmse(error) of the model
        """       
        predictions= self.predict(dataset)
        return rmse(dataset.y, predictions) # compara os previstos com os valores reais
    
if __name__ == '__main__':

    # Caminho para o arquivo CSV
    path = "C:/Users/luis-/Documents/GitHub/Sistemas_inteligentes/datasets/cpu/cpu.csv"
    
    # Ler o conjunto de dados a partir do arquivo CSV
    df = pd.read_csv(path)

    #dataset_ = Dataset.from_dataframe(df, label='perf') #porquê que não dá para fazer logo assim?

    # Assumindo que "perf" é a coluna que se pretende prever
    label_column = 'perf'
    
    # Obter as colunas de características (features)
    features_columns = df.columns[df.columns != label_column]

    # Criar um objeto Dataset
    dataset_ = Dataset(X=df[features_columns].to_numpy(), y=df[label_column].to_numpy())

    # Dividir o conjunto de dados em treino e teste
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.25)

    # Criar um modelo de regressão k-NN (K-Nearest Neighbors)
    knn = KNNRegressor(k=5)

    # Treinar o modelo
    knn.fit(dataset_train)

    # Calcular e imprimir o RMSE score
    score = knn.score(dataset_test)
    print(f'The RMSE of the model is: {score}')





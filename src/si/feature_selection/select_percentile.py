from typing import Callable
import numpy as np
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification
from si.io.csv_file import read_csv
from icecream import ic

class SelectPercentile: 

    """
    Select features according to percentiles.
    Feature ranking is performed by computing the scores of each feature using a scoring function:
        - f_classification: ANOVA F-value between label/feature for classification tasks.
        - f_regression: F-value obtained from F-value of r's pearson correlation coefficients for regression tasks.

    Parameters
    ----------
    score_func: callable
        Function taking dataset and returning a pair of arrays (scores, p_values)
    percentile: float, default = 0.5
            The percentile of features to select.

    Attributes
    ----------
    F: array, shape (n_features,)
        F scores of features.
    p: array, shape (n_features,)
        p-values of F-scores.
    """
    
    def __init__(self, score_func: Callable = f_classification, percentile: float = 0.5):
        """
        Select features according to the percentile.

        Parameters
        ----------
        score_func: callable
            Function taking dataset and returning a pair of arrays (scores, p_values).
        percentile: float, default = 0.5
            The percentile of features to select. 
        """
        self.percentile = percentile
        self.score_func = score_func
        self.F = None
        self.p = None     

    def fit(self, dataset: Dataset) -> "SelectPercentile": 

        """
        It fits SelectPercentile to compute the F scores and p-values.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        self: object
            Returns self.
        """
        self.F, self.p = self.score_func(dataset)   #vamos definir o F com o F score e o p value para o dataset
        return self
    
    def transform(self, dataset: Dataset) -> Dataset:
        """
        It transforms the dataset by selecting features based on the specified percentile.
        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            A labeled dataset with features selected based on the specified percentile.
        """
        if self.F is None:
            raise ValueError("Model not fitted")
        k = int(len(dataset.features) * self.percentile) # Calcula o número de características a serem mantidas. Total de carateristicas * percentagem que queremos das carateristicas.
        ic(k)
        if k == 0: k = 1  # se o k for igual a 0, vai selecionar todas as features e não é suposto
        sorted_indices = np.argsort(self.F)   #ordena os elementos do array por ordem crescente mas retorna o indice, ao em vez do elemento. X = [3 2 1 4], np.argsort(X) = [2,1,0,3], lembrando que o index começa em 0. 
        idxs = sorted_indices[-k:]  #Vamos obter os ultimos scores da lista, que vao ser considerados os scores mais altos para o k considerado.
        features = np.array(dataset.features)[idxs] #obtemos as features correspondentes ao/aos index/indexes obtidos no idxs.
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fit the SelectPercentile method and transform the dataset by selecting features based on the specified percentile.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset.

        Returns
        -------
        dataset: Dataset
            A labeled dataset with features selected based on the specified percentile of feature importance.
        """
        self.fit(dataset)
        return self.transform(dataset)
    

if __name__ == '__main__':
        
        path= "C:/Users/luis-/Documents/GitHub/Sistemas_inteligentes/datasets/iris/iris.csv"
        dataset =  read_csv(path, features=True, label=True)

        selector = SelectPercentile(f_classification, percentile= 0.25)
        selector = selector.fit(dataset)
        dataset = selector.transform(dataset)
        print(dataset.features)

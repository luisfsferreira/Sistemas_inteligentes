from typing import Tuple, Sequence, Union

import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray = None, features: Sequence[str] = None, label: str = None) -> None:
        """
        Dataset represents a tabular dataset for single output classification.

        Parameters
        ----------
        X: numpy.ndarray (n_samples, n_features)
            The feature matrix
        y: numpy.ndarray (n_samples, 1)
            The label vector
        features: list of str (n_features)
            The feature names
        label: str (1)
            The label name
        """
        if X is None:
            raise ValueError("X cannot be None")
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if features is not None and len(X[0]) != len(features):
            raise ValueError("Number of features must match the number of columns in X")
        if features is None:
            features = [f"feat_{str(i)}" for i in range(X.shape[1])]
        if y is not None and label is None:
            label = "y"
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the dataset
        Returns
        -------
        tuple (n_samples, n_features)
        """
        return self.X.shape

    def has_label(self) -> bool:
        """
        Returns True if the dataset has a label
        Returns
        -------
        bool
        """
        return self.y is not None

    def get_classes(self) -> np.ndarray:
        """
        Returns the unique classes in the dataset
        Returns
        -------
        numpy.ndarray (n_classes)
        """
        if self.has_label():
            return np.unique(self.y)
        else:
            raise ValueError("Dataset does not have a label")

    def get_mean(self) -> np.ndarray:
        """
        Returns the mean of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmean(self.X, axis=0)

    def get_variance(self) -> np.ndarray:
        """
        Returns the variance of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanvar(self.X, axis=0)

    def get_median(self) -> np.ndarray:
        """
        Returns the median of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmedian(self.X, axis=0)

    def get_min(self) -> np.ndarray:
        """
        Returns the minimum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmin(self.X, axis=0)

    def get_max(self) -> np.ndarray:
        """
        Returns the maximum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmax(self.X, axis=0)

    def summary(self) -> pd.DataFrame:
        """
        Returns a summary of the dataset
        Returns
        -------
        pandas.DataFrame (n_features, 5)
        """
        data = {
            "mean": self.get_mean(),
            "median": self.get_median(),
            "min": self.get_min(),
            "max": self.get_max(),
            "var": self.get_variance()
        }
        return pd.DataFrame.from_dict(data, orient="index", columns=self.features)

#Exercício 2.1

    def dropna (self):

        identificar = np.all(~np.isnan(self.X), axis=1)  # np.isnan(self.X) Obtenho um boleano com os verdadeiros sendo iguais a NA mas o ~inverte a logica e agora quando é igual a NA, é falso. Na função toda, obtenho um boleano sendo os verdadeiros, as linhas que nao tem nenhum NA.
        self.X = self.X[identificar]  #Nova matriz ondas as linhas de identificar são true. Removo as linhas falsas
        self.y = self.y[identificar]    #Removo os rotulas das amostras que forem falsas.

        return self

#Exercício 2.2

    def fillna (self, valor):
        identificar = np.isnan(self.X)  #Boleano em que os valores de NA são considerados verdadeiros.

        if valor == "mean":
            média = np.nanmean(self.X, axis=0) #faz a média, ignorando os valores de NA
            self.X = np.where(identificar, média, self.X)   # identificar é a matriz boleana em que os seus valores true são substituidos pela média. Self.X é a matriz original em que se o elemento for false, o valor vai se manter

        elif valor == "median":
            mediana = np.nanmedian(self.X, axis=0) #faz a mediana, ignorando os valores de NA
            self.X = np.where(identificar, mediana, self.X)   # substitui os valores True pela mediana

        else:
            self.X[identificar] = valor

        return self
    
#Exercício 2.3
    
    def remove_by_index (self, index):
        self.X = np.delete(self.X, index, axis = 0)   #vai ser removida a linha correspondente ao index. Aqui o axis= 0 é a linha e axis = 1 é a coluna
        self.y = np.delete(self.y, index)             #vai ser removido o rotulo correspondente a esta linha (amostra)
        return self

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, label: str = None):
        """
        Creates a Dataset object from a pandas DataFrame

        Parameters
        ----------
        df: pandas.DataFrame
            The DataFrame
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        if label:
            X = df.drop(label, axis=1).to_numpy()
            y = df[label].to_numpy()
        else:
            X = df.to_numpy()
            y = None

        features = df.columns.tolist()
        return cls(X, y, features=features, label=label)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataset to a pandas DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        if self.y is None:
            return pd.DataFrame(self.X, columns=self.features)
        else:
            df = pd.DataFrame(self.X, columns=self.features)
            df[self.label] = self.y
            return df

    @classmethod
    def from_random(cls,
                    n_samples: int,
                    n_features: int,
                    n_classes: int = 2,
                    features: Sequence[str] = None,
                    label: str = None):
        """
        Creates a Dataset object from random data

        Parameters
        ----------
        n_samples: int
            The number of samples
        n_features: int
            The number of features
        n_classes: int
            The number of classes
        features: list of str
            The feature names
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return cls(X, y, features=features, label=label)


if __name__ == '__main__':
    X = np.array([[1, 2, 3], [4,  np.nan, np.nan],[5, 10, 3]])
    y = np.array([1, 2,3])
    features = np.array(['a', 'b', 'c'])
    label = 'y'
    dataset = Dataset(X, y, features, label)

#Forma como os dados estão organizados

 # a   |   b   |   c   |   y
 #-----|-----|-----|-----
 # 1   |   2   |   3   |   1
 # 4   |   NA  |   NA  |   2    
 # 5   |   10  |   3   |   3 

    print(dataset.X.shape[0])
    print()
    print(dataset.has_label())
    print(dataset.get_classes())
    print(dataset.get_mean())
    print(dataset.get_variance())
    print(dataset.get_median())
    print(dataset.get_min())
    print(dataset.get_max())
    print(dataset.summary())
    print()
    # dataset.dropna()
    # dataset.fillna(5)
    print(dataset.remove_by_index(0))
    print(dataset.X)
    print(dataset.y)
    
   

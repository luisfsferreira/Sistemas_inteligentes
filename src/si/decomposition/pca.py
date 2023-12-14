import numpy as np
from si.data.dataset import Dataset 

class PCA:
    def __init__(self, n_components: int) :
        """
        Initialize the PCA model with the specified number of components.

        Parameters
        ----------
        n_components: int
            The number of principal components to retain for dimensionality reduction.
        """

        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, dataset: Dataset):
        """ 
        It fits PCA computing principal components and explained variance.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        self: object
            Returns self.
        """

        # Centralizar os dados
        self.mean = np.mean(dataset.X, axis=0) # média de cada característica (ao longo da coluna)
        Dataset_centered = dataset.X - self.mean

        # Calcular a SVD
        U, S, VT = np.linalg.svd(Dataset_centered, full_matrices=False)

        # Inferir os Componentes Principais
        self.components = VT[:self.n_components]

        # Inferir a Variância Explicada
        EV = (S ** 2) / (len(dataset.X)-1) # n corresponde ao numero de amostras (Dataset.shape[0]) antes estava assim (dataset.X.shape[0] - 1)
        self.explained_variance = EV[:self.n_components] 
        return self

    def transform(self, dataset: Dataset):
        """
        Transform the dataset by reducing its dimensionality using PCA.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        reduced_dataset : Dataset
            A labeled dataset with reduced dimensionality.
        """

        X_centered = dataset.X - self.mean
        x_reduced = np.dot(X_centered, np.transpose(self.components)) 
        reduced_dataset = Dataset(x_reduced, dataset.y, dataset.features[:self.n_components], dataset.label)
        return reduced_dataset
    
    def fit_transform(self, dataset:Dataset):
        """
        Fit the PCA model and transform the dataset.

        Parameters
        ----------
        dataset : Dataset
            A labeled dataset.

        Returns
        -------
        reduced_dataset : Dataset
            A labeled dataset with reduced dimensionality.
        """

        return self.fit(dataset).transform(dataset)

#testing
if __name__ == '__main__':

    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([0, 1, 0])
    print(X.shape)
    
    label = 'target'
    features = ['feature1', 'feature2', 'feature3']
    dataset = Dataset(X, y, features, label)
    
    #Sklearn
    sklearn_pca = PCA(n_components=2)
    sklearn_pca.fit(dataset)
    sklearn_transformed_data = sklearn_pca.transform(dataset)

    #PCA
    my_pca = PCA(n_components=2)
    my_pca.fit(dataset)
    my_transformed_dataset = my_pca.transform(dataset)

    print("scikit-learn Transformed Data:")
    print(sklearn_transformed_data.X)
    print("My PCA Transformed Data:")
    print(my_transformed_dataset.X)


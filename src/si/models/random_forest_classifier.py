import numpy as np
from si.data.dataset import Dataset
from typing import Literal
from decision_tree_classifier import DecisionTreeClassifier
from si.metrics.accuracy import accuracy


class RandomForestClassifier:
    """
    Ensemble machine learning technique that combines multiple decision trees to improve prediction accuracy and reduce overfitting.
    
    Parameters
    ----------
    n_estimators (int): Number of decision trees to use in the forest.
    max_features (int or None): Maximum number of features to use per tree. If None, set to int(np.sqrt(n_features)).
    min_sample_split (int): Minimum samples allowed in a split.
    max_depth (int): Maximum depth of the decision trees.
    mode (Literal['gini', 'entropy']): Impurity calculation mode.
    seed (int): Random seed for reproducibility.

    Attributes
    ----------
    trees (list): List to store the trained decision trees.
    """

    def __init__(self, n_estimators:int = 500, max_features: int = None, min_sample_split:int = 2,  max_depth: int = 10, mode: Literal['gini', 'entropy'] = 'gini', seed:int = 42):
        """
        Initialize the Random Forest Classifier.

        Parameters
        ----------
        n_estimators (int): Number of decision trees to use in the forest.
        max_features (int or None): Maximum number of features to use per tree. If None, set to int(np.sqrt(n_features)).
        min_sample_split (int): Minimum samples allowed in a split.
        max_depth (int): Maximum depth of the decision trees.
        mode (Literal['gini', 'entropy']): Impurity calculation mode.
        seed (int): Random seed for reproducibility.
        """

        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed

        self.trees = []
        
    def fit(self, dataset:Dataset)-> "RandomForestClassifier":
        """
        Train the random forest on the provided dataset.

        Parameters
        ----------
        dataset: Dataset
            The training dataset.

        Returns
        -------
        self: RandomForestClassifier
            The trained random forest
        """

        np.random.seed(self.seed)
    
        n_features = dataset.shape[1]
        n_samples = dataset.shape[0]

        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))
        else:
            self.max_features

        for x in range (self.n_estimators):

            index = np.random.choice(n_samples, size = n_samples, replace = True)
            self.max_features = np.random.choice(n_features, size = n_features)
   
            bootstrap_X = dataset.X[index] [:, self.max_features]  # quero apenas as linhas presentes em index e a outra condição é, quero todas as linhas presentes na étapa anterior mas só as colunas em self.max_features
            bootstrap_y = dataset.y[index]  # seleciona a classe para cada linha (amostra)
            
            random_dataset = Dataset (dataset.X[bootstrap_X], dataset.y[bootstrap_y])

            tree = DecisionTreeClassifier(min_sample_split = self.min_samples_split, max_depth = self.max_depth, mode =self.mode )

            tree.fit(random_dataset)

            self.tree.append(bootstrap_y,tree)

            return self
    
    def predict (self, dataset:Dataset)-> np.ndarray:
        """
        Make predictions using the trained random forest.

        Parameters
        ----------
        dataset: Dataset: 
            The dataset for which predictions are to be made.

        Returns
        -------
        predictions: np.ndarray
            Array of predicted class labels.
        """
        
        predictions = []
        for features, tree in self.trees:
            X_subset = dataset.X[:, features]   # Obtenho todas as linhas mas apenas as colunas (features) presente na tree.
            tree_pred = tree.predict(X_subset)  # Vou usar a tree com os dados treinados para prever os dados do meu dataset.
            predictions.append((features, tree_pred))       # Adiciono o resultado à lista vazia

        predictions = np.array(predictions)

        features, counts = np.unique(predictions[:,1], return_counts=True)  # [:, 1] diz que queremos todas as entrada da segunda coluna, que contém as previsões da árvore

        return features[np.argmax(counts)]        
    
    def score (self, dataset: Dataset) -> float:
        """
        Calculate the accuracy of the random forest on the provided dataset.

        Parameters
        ----------
        dataset: Dataset 
            The dataset for which accuracy is to be calculated..

        Returns
        -------
        accuracy: float
            Accuracy of the random forest on the provided dataset.
        """

        predictions = self.predict(dataset)

        return accuracy(dataset.y, predictions)




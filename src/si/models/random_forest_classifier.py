import numpy as np
from si.data.dataset import Dataset
from typing import Literal
from decision_tree_classifier import DecisionTreeClassifier
from si.metrics.accuracy import accuracy
from si.io.csv_file import read_csv
from si.model_selection.split import stratified_train_test_split


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
    
        n_samples, n_features = dataset.shape()

        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))
        else:
            self.max_features

        for x in range (self.n_estimators):

            index = np.random.choice(n_samples, size = n_samples, replace = True)
            features_used = np.random.choice(n_features, size = self.max_features, replace = False)
   
            bootstrap_X = dataset.X[index, :] [:, features_used]  # quero apenas as linhas presentes em index e a outra condição é, quero todas as linhas presentes na étapa anterior mas só as colunas em self.max_features
            bootstrap_y = dataset.y[index]  # seleciona a classe para cada linha (amostra)
            
            random_dataset = Dataset(bootstrap_X, bootstrap_y)

            tree = DecisionTreeClassifier(min_sample_split = self.min_samples_split, max_depth = self.max_depth, mode =self.mode )

            tree.fit(random_dataset)

            self.trees.append((features_used, tree))

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

        tree_pred_list = []

        for features, tree in self.trees:
            X_subset = dataset.X[:, features]    # Obtenho todas as linhas mas apenas as colunas (features) presente na tree.
            tree_pred = tree.predict(Dataset(X=X_subset)) # Vou usar a tree com os dados treinados para prever os dados do meu dataset.
            
            tree_pred_list.append(tree_pred)   # Adiciono o resultado à lista vazia

        tree_pred_array = np.array(tree_pred_list)

        # Now, you can use features_array and tree_pred_array as needed.

        features, counts = np.unique(tree_pred_array, return_counts=True)

        return features[np.argmax(counts)]     #Obtemos a feature que aparece mais vezes na lista de previsões.    
    
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

if __name__ == "__main__":
    path= "C:/Users/luis-/Documents/GitHub/Sistemas_inteligentes/datasets/iris/iris.csv"
    dataset =  read_csv(path, features=True, label=True)

    dataset_train, dataset_test = stratified_train_test_split(dataset, test_size=0.33, random_state=42)

    # X_train, y_train = dataset_train.X, dataset_train.y
    # X_test, y_test = dataset_test.X, dataset_test.y

    model = RandomForestClassifier(n_estimators = 500, max_features = None, min_sample_split= 2,  max_depth = 10, seed = 42, mode = "gini" )

    model.fit(dataset_train)

    print("Model score:", model.score(dataset_test))


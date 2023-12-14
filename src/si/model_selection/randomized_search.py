from si.data.dataset import Dataset
from typing import Callable, Tuple, Dict, Any
import numpy as np
from cross_validate import k_fold_cross_validation


def  randomized_search_cv (model, dataset:Dataset, hyperparameter_grid: Dict[str, tuple], scoring: Callable = None, cv: int = 2, n_iter:int = None):
    """
    Performs a randomized search for hyperparameters using cross-validation.

    Parameters
    ----------
    - model: Model object for hyperparameter optimization.
    - dataset: Dataset for cross-validation.
    - hyperparameter_grid: Dictionary with hyperparameter names and possible values represented by numpy arrays.
    - scoring: Scoring function to evaluate model performance. Can be null if not provided.
    - cv: Number of folds for cross-validation.
    - n_iter: Number of hyperparameter combinations to test.

    Returns:
    ---------
    Dictionary with the results of the randomized search for hyperparameters, including lists of tested hyperparameters, corresponding scores, the best hyperparameter combination, and the best score.
    """    
    
    for hyperparameter in hyperparameter_grid:
        if not hasattr(model, hyperparameter):
            raise ValueError(f"Invalid hyperparameter: {hyperparameter}")
        
    hyperparameters_list = []
    scores_list = []

    for _ in range(n_iter):
    # Initialize an empty dictionary for the current hyperparameters
        current_hyperparameters = {}

        # Set each hyperparameter with a randomly chosen value
        for hyperparameter, values in hyperparameter_grid.items():   #Obtemos uma sequencia de (key, value). As keys são nomes de hiperparâmetros e os valores são listas de valores possíveis para esses hiperparâmetros.
            current_hyperparameters[hyperparameter] = np.random.choice(values)   # Seleciona um valor da lista correspondente de valores possíveis (values) e o associa ao hiperparâmetro (hyperparameter).

        # Set the model hyperparameters with the current combination
        for hyperparameter, value in current_hyperparameters.items():   
            setattr(model, hyperparameter, value)   # Define os hiperparâmetros do objeto model com os valores escolhidos aleatoriamente armazenados em current_hyperparameters

        scores = k_fold_cross_validation(model, dataset, scoring, cv)

        mean_score = np.mean(scores)
        hyperparameters_list.append(current_hyperparameters)
        scores_list.append(mean_score)

        # Check if the current combination is the best so far
        best_score = 0
        if mean_score > best_score:
            best_score = mean_score
            best_hyperparameters = current_hyperparameters.copy()  #Dicicionário em que ficam guardados os valores de hiperparâmetros que levaram à melhor pontuação obtida durante a procura aleatória.

    return {
        'hyperparameters': hyperparameters_list,
        'scores': scores_list,
        'best_hyperparameters': best_hyperparameters,
        'best_score': best_score}

if __name__ == '__main__':

    from si.io.csv_file import read_csv
    from si.models.logistic_regression import LogisticRegression

    path = "C:/Users/luis-/Documents/GitHub/Sistemas_inteligentes/datasets/breast-bin/breast-bin.csv"
    dataset = read_csv(path, sep= ',', label= True,  features = True)

    hyperparameter_grid = {
        'l2_penalty': np.linspace(1, 10, 10),
        'alpha': np.linspace(0.001, 0.0001, 100),
        'max_iter': np.linspace(1000, 2000, 200)
    }

    model = LogisticRegression()

    result = randomized_search_cv(model, dataset, hyperparameter_grid, scoring=None, cv=3, n_iter=10)

    # Print the results
    print("Hyperparameters:", result['hyperparameters'])
    print("Scores:", result['scores'])
    print("Best Hyperparameters:", result['best_hyperparameters'])
    print("Best Score:", result['best_score'])
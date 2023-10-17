from typing import Tuple, Union

import numpy as np
from scipy import stats

from si.data.dataset import Dataset


def f_classification(dataset: Dataset) -> Union[Tuple[np.ndarray, np.ndarray],
                                                Tuple[float, float]]:
    """
    Scoring function for classification problems. It computes one-way ANOVA F-value for the
    provided dataset. The F-value scores allows analyzing if the mean between two or more groups (factors)
    are significantly different. Samples are grouped by the labels of the dataset.

    Parameters
    ----------
    dataset: Dataset
        A labeled dataset

    Returns
    -------
    F: np.array, shape (n_features,)
        F scores
    p: np.array, shape (n_features,)
        p-values
    """
    classes = dataset.get_classes()
    groups = [dataset.X[dataset.y == c] for c in classes]   # Cria um array com as amostras de cada classe, se forem amostras da mesma classe, juntasse. Cria um boleano com os casos verdadeiros as amostras com a mesma classe que o c.
    F, p = stats.f_oneway(*groups)   
    return F, p

# Cada valor na matriz F corresponde a uma característica específica e representa a diferença na variabilidade entre os grupos para essa característica.

if __name__ == '__main__':
    data = Dataset(np.array([[1, 2, 1], [4, 5, 6], [7, 8, 7], [10, 11, 12]]), np.array([1, 0, 1, 0]))
    print(f_classification(data))

# Dados estão desta forma 

#  Feature_1   Feature_2    Feature_3   Label
#  1          2            1            1
#  4          5            6            0
#  7          8            7            1
#  10         11           12           0

#Como os dados ficam agrupados:

#Grupo 0

# Amostra 2: [4, 5, 6]
# Amostra 3: [10, 11, 12]

#Grupo 1

# Amostra 0: [1, 2, 1]
# Amostra 1: [7, 8, 7]

# O F para a feature 1 (coluna 1) -> Média dentro do grupo -> Grupo 0: [4, 10] (média = 7)
#                                                             Grupo 1: [1, 7] (média = 4)
# Os valores de F indicam quão diferentes são as médias entre os grupos para cada característica.                            
# F grande -> Se a variabilidade entre os grupos (0 e 1) for muito maior do que a variabilidade dentro dos grupos (variabilidade dos dados em relação à média), o valor de F será grande, o que indica uma grande dispersão entre as médias do grupo 0 e 1.
# F baixo -> significa que a variabilidade dentro de cada grupo (variabilidade dos dados em relação à média) é maior do que a variabilidade entre os grupos.
# A H0 para calcular o p é que as médias são iguais entre os grupos (não há diferenças significativas)
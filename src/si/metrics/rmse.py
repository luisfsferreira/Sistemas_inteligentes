import numpy as np

def rmsee (y_true: int, Y_pred: int) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE) between real values and predicted values.

    Parameters
    ----------
    y_true : int
        Real values.
    y_pred : int
        Predicted values.

    Returns
    -------
    float
        Value corresponding to the RMSE between y_true and y_pred.
    """
    return np.sqrt(np.sum((y_true-Y_pred)**2) / len(y_true))

# if __name__ == '__main__':
#     # Exemplo: 

#     y_true = np.array([1, 2, 3])
#     y_pred = np.array([2, 2, 4])

#     # Calcular as diferenças ao quadrado
#     diferencas_ao_quadrado = (y_true - y_pred)**2  # Resultado: array([1, 0, 1])

#     print(diferencas_ao_quadrado)

#     # Soma das diferenças ao quadrado
#     soma_diferencas_ao_quadrado = np.sum(diferencas_ao_quadrado)

#     print(soma_diferencas_ao_quadrado)
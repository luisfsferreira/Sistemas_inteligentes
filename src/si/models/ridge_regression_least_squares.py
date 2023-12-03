import numpy as np
from si.data.dataset import Dataset
from si.data.metrics.mse import mse

class RidgeRegressionLeastSquares:
    """
    Implementation of Ridge Regression using Least Squares.

    Parameters
    ----------
    l2_penalty: float
        L2 regularization penalty parameter.
    scale: bool
        Indicates whether the data should be normalized.

    Attributes
    ----------

    theta: np.ndarray
        Coefficient vector of the model.
    theta_zero: float
        Intercept of the model.
    """
    def __init__(self, l2_penalty:float, scale:bool):
        """
        Initializes an instance of RidgeRegressionLeastSquares.

        Parameters:
            - l2_penalty (float): L2 regularization penalty parameter (Ridge).
            - scale (bool): Indicates whether the data should be normalized.
        """
        self.l2_penalty = l2_penalty
        self.scale = scale

        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def fit (self, dataset: Dataset) -> "RidgeRegressionLeastSquates":
        """
        Trains the Ridge Regression model using Least Squares.

        Parameters:
            - dataset (Dataset): Training dataset.

        Returns:
            - RidgeRegressionLeastSquares: The instance of the class itself.
        """
        if self.scale:  #se for necessário normalizar
            # calcula a média e desvio padrão de cada característica (ou coluna) no conjunto de dados X
            self.mean = np.nanmean(dataset.X, axis=0)   # calcula a média, ignorando valores NaN. Eixo 0, ou seja, ao longo da coluna
            self.std = np.nanstd(dataset.X, axis=0)     # calcula o desvio padrão, ignorando valores NaN
            # scale the dataset
            X = (dataset.X - self.mean) / self.std       
        else: 
            X = dataset.X 
        
        m, n = dataset.shape  # (linhas,colunas) & (amostras, carateristicas)

        X = np.c_[np.ones(m), X]

        penalty_matrix = self.l2_penalty * np.eye (n + 1) # np.eye cria a matriz identidade x por x, em que x é o numero de colunas (aumentamos o numero de colunas em cima, por isso adicionamos 1).
        penalty_matrix[0,0] = 0

        lambda_1 = np.linalg.inv(penalty_matrix + X.T.dot(X))
        lambda_2 = lambda_1.dot(dataset.y)

        self.theta_zero = lambda_2[0]
        self.theta = lambda_2 [1:]

        return self
    
    def predict(self, dataset:Dataset) -> np.ndarray:
        """
        Performs predictions for the input dataset.

        Parameters:
            - dataset (Dataset): Dataset for making predictions.

        Returns:
            - np.ndarray: Vector of predictions.
        """
        if self.scale:  #in case its necessary to scale
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        m, n = dataset.shape  # (linhas,colunas) & (amostras, carateristicas)
        X = np.c_[np.ones(m), X]
    
        Y_predict = X.dot(np.r_(self.theta_zero, self.theta))

        return Y_predict
    
    def score (self, dataset:Dataset) -> float:
        """
        Calculates the Mean Squared Error (MSE) between predictions and actual labels.

        Parameters:
            - dataset (Dataset): Dataset for evaluation.

        Returns:
            - float: MSE value.
        """
        Y_predict = self.predict[dataset]
        mse_score = mse(dataset.y, Y_predict)

        return mse_score
    

if __name__ == '__main__':
    # Criar um conjunto de dados linear
    X = np.array([[0, 1], [3, 2], [4, 1]])
    y = np.dot(X, np.array([3, 1])) + 3  # O np.dot é usado para calcular o produto escalar entre a matriz de características X e o vetor de pesos. O np.array([1,2]) determina a contribuição/peso de cada caraterística para prever o y. O 3 é o intercepto e mesmo que todas as características em X sejam zero, o valor previsto de y ainda será pelo menos 3.
    dataset_ = Dataset(X=X, y=y)

    # Inicializar e treinar o modelo RidgeRegressionLeastSquares
    model_custom = RidgeRegressionLeastSquares(l2_penalty=100, scale=True)
    model_custom.fit(dataset_)
    
    # Imprimir coeficientes e intercepto do modelo personalizado
    print("Coeficientes do modelo customizado:", model_custom.theta)
    print("Intercepto do modelo customizado:", model_custom.theta_zero)

    # Calcular e imprimir a pontuação do modelo personalizado
    print("Pontuação do modelo customizado:", model_custom.score(dataset_))

    # -------------- Comparação com o modelo Ridge do scikit-learn -------------- #
    from sklearn.linear_model import Ridge

    # Inicializar e treinar o modelo Ridge do scikit-learn
    model_sklearn = Ridge(alpha=100.0)  # Usando a mesma penalização
    # Escalonar os dados
    X_scaled = (dataset_.X - np.nanmean(dataset_.X, axis=0)) / np.nanstd(dataset_.X, axis=0)
    model_sklearn.fit(X_scaled, dataset_.y)

    # Imprimir coeficientes e intercepto do modelo Ridge do scikit-learn
    print("\nCoeficientes do modelo Ridge do scikit-learn:", model_sklearn.coef_)  #theta
    print("Intercepto do modelo Ridge do scikit-learn:", model_sklearn.intercept_)   #theta zero

    # Calcular e imprimir o MSE entre os rótulos reais e as previsões do modelo Ridge do scikit-learn
    mse_sklearn = mse(dataset_.y, model_sklearn.predict(X_scaled))
    print("MSE do modelo Ridge do scikit-learn:", mse_sklearn)

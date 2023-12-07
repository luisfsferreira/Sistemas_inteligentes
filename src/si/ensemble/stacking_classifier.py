from si.data.dataset import Dataset
import numpy as np
from si.metrics.accuracy import accuracy

class StackingClassifier:
    """
    Initializes the StackingClassifier.

    Parameters
    ----------
    models : list
        Initial set of models to form the ensemble.
    final_model : object
        The model used to make the final predictions based on the ensemble.

    Attributes
    ----------
    models : list
        List to store the initial set of models.
    final_model : object
        The final model to make the ultimate predictions.
    """    
 
    def __init__(self, models: list, final_model):
        self.models = models
        self.final_model = final_model

    def fit (self, dataset: Dataset) -> "StackingClassifier":
        """
        Fit the models to the dataset.

        Parameters
        ----------
        dataset : Dataset
            The training dataset.

        Returns
        -------
        self 
        """

        for model in self.models:
            model.fit(dataset)
        
        predicts = []
        for model in self.models:
            predicts.append(model.predict(dataset))

        predictions_1 = np.array(predicts).T # Passar de uma lista para um array e fazer a transposta. Cada coluna representa a previsao de cada modelo para diferentes amostras
        
        # trains the final model
        self.final_model.fit(Dataset(predictions_1, dataset.y)) # as características são as previsões dos modelos base (predictions) e as variáveis alvo são os rótulos verdadeiros (dataset.y) do conjunto de dados original (dataset)
        return self                                             # durante o treino, usamos dataset.y porque estamos ensinando o modelo final com base nos rótulos verdadeiros.
                                                                # ajusta o modelo final com base nas previsões dos modelos base e nos rótulos verdadeiros.
                                                                # No final, o modelo está pronto para fazer previsões em novos dados.
    # Para entender os dados 
    # Temos este dataset X_train = np.array([[0.1, 0.2, 0.3],    amostra 1
    #                                        [0.4, 0.5, 0.6],     amostra 
    #                                        [0.7, 0.8, 0.9]])   amostra 3

    # Rótulos verdadeiros
    #                                        y_train = np.array([1, 0, 1])

    # Aqui é como se fosse a nossa variavel predicts em que cada linha é um modelo. Ao fazer a transposta, cada coluna é um modelo.

    # model1_predictions = np.array([0.9, 0.1, 0.8])  # Exemplo de previsões contínuas
    # model2_predictions = np.array([0.2, 0.8, 0.3])  # Exemplo de previsões contínuas
    #                              Cada valor presente no modelo representa a probabilidade da amostra [amostra1, amostra2, amostra3]
    
    # Os modelos base são responsáveis por gerar previsões individuais, enquanto o modelo final é treinado para combinar essas previsões de maneira eficaz, melhorando o desempenho geral do ensemble.

    def predict (self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the labels using the ensemble models.

        Parameters
        ----------
        dataset : Dataset
            The dataset for which predictions are to be made.

        Returns
        -------
        final_pred : np.ndarray
            Array of final predictions made by the ensemble.
        """
        predictions_2 = []
        for model in self.models:
            predictions_2.append(model.predict(dataset))
        
        # gets the final model previsions
        final_pred = self.final_model.predict(Dataset(dataset.X, np.array(predictions_2).T))  # forneco ao modelo final as características originais e as previsões dos modelos base. Aqui o predict faz previsões em um conjunto de dados de entrada (dataset.X) com base no modelo treinado.
         
        return final_pred                                                                     
    
    def score (self, dataset: Dataset) -> float:
        """
        Calculate the accuracy of the StackingClassifier on the provided dataset.

        Parameters
        ----------
        dataset : Dataset 
            The dataset for which accuracy is to be calculated.

        Returns
        -------
        score : float
            Accuracy of the StackingClassifier on the provided dataset.
        """

        y_pred = self.predict(dataset)           # O score  é responsável por calcular a precisão do modelo no conjunto de dados fornecido
        score = accuracy(dataset.y, y_pred)      # Comparamos os rotulos reais, com os rotulos que foram previstos.

        return score
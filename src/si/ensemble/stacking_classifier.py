from si.data.dataset import Dataset
import numpy as np
from si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance
from si.statistics.manhattan_distance import manhattan_distance

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
        """
        Initializes the StackingClassifier.
        """
        self.models = models
        self.final_model = final_model


    def fit(self, dataset: Dataset) -> "StackingClassifier":
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

        predictions = []
        for model in self.models:
            predictions.append(model.predict(dataset))

        # gets the final model prediction
        y_pred = self.final_model.predict(Dataset(np.array(predictions).T, dataset.y))

        return y_pred

    
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
    
if __name__ == "__main__":
    
    from si.io.csv_file import read_csv
    from si.model_selection.split import stratified_train_test_split
    from si.models.knn_classifier import KNNClassifier
    from si.models.logistic_regression import LogisticRegression
    from si.models.decision_tree_classifier import DecisionTreeClassifier
    
    path = "C:/Users/luis-/Documents/GitHub/Sistemas_inteligentes/datasets/breast-bin/breast-bin.csv"
    dataset = read_csv(path, sep= ',', label= True,  features = True)

    train, test = stratified_train_test_split(dataset, test_size= 0.3, random_state= 42)

    model_knn = KNNClassifier(k= 3)
    model_logistic = LogisticRegression (l2_penalty=1, alpha=0.001, max_iter=1000)
    model_decision_tree = DecisionTreeClassifier( min_sample_split=3, max_depth=3, mode='gini')

    model_knn_final = KNNClassifier(k=2)
    models = [model_knn, model_logistic, model_decision_tree]

    compare = StackingClassifier(models, model_knn_final)

    compare.fit(train)

    print(compare.score(test)) # Usasse as previsões dos modelos base como características para o modelo final. O valor de 1 significa que todas as previsões do modelo foram corretas no conjunto de teste
 






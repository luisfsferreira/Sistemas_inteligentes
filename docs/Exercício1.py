# 1

#1.1)

import numpy as np

from si.io.csv_file import read_csv

path= "C:/Users/luis-/Documents/GitHub/Sistemas_inteligentes/datasets/iris/iris.csv"
dataset =  read_csv(path, features=True, label=True)

print(dataset)

print()
#1.2) 

array = dataset.X[:,-2] # dataset.X Representa os dados do conjunto de recursos. Estes dados incluem as colunas sepal_length, sepal_width, petal_length e petal_width, que são as características ou atributos do conjunto de dados.

# print(dir(dataset)) # lista de todos os atributos e métodos disponíveis para o objeto dataset

#print(dataset.y) # Representa os rótulos ou classes do conjunto de dados, neste caso as classes representam as espécies de flores, como "Iris-setosa.". Este é frequentemente o que você está tentando prever ou classificar. 

print("Dimensão do array resultante:", array.shape) # Retorna as dimensões do conjunto de dados, ou seja, o número de observações (linhas) e características (colunas).

print()
#1.3)

ultimas_amostras = (dataset.X[-10:])
média = ultimas_amostras.mean(axis=0)
print("Média das últimas 10 amostras de cada coluna:", média)   # Calcular a média ao longo do eixo 0 (média de cada coluna);

print()

#1.4)
menores6 = (dataset.X <= 6).all(axis=1)    #Vamos obter um array boleano em que se todas (all) as colunas (axis = 1) na mesma linha, forem verdadeiras, ou seja, menores ou iguais a 6, retorna True, se não, retorna False

total = menores6.sum()  #soma os casos verdadeiros 

print("Total de amostras:", total)
print()

#1.5)

dataset_registo = dataset.y != 'Iris-setosa'  #Obtemos um boleano em que se amostra for igual a iris-setosa, vai ser falso.

dataset_sem_iris_setosa = dataset.X[dataset_registo, :]  #  todas as amostras onde dataset_registo é True serão selecionadas, se existir um false na linha, essa linha nao vai contar. Vamos considerar para todas as colunas

dimensão = dataset_sem_iris_setosa.shape[0]

print("Amostras obtidas:", dimensão)

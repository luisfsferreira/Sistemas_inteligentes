import numpy as np


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    It computes the euclidean distance of a point (x) to a set of points y.
        distance_x_y1 = sqrt((x1 - y11)^2 + (x2 - y12)^2 + ... + (xn - y1n)^2)
        distance_x_y2 = sqrt((x1 - y21)^2 + (x2 - y22)^2 + ... + (xn - y2n)^2)
        ...

    Parameters
    ----------
    x: np.ndarray
        Point.
    y: np.ndarray
        Set of points.

    Returns
    -------
    np.ndarray
        Euclidean distance for each point in y.
    """
    return np.sqrt(((x - y) ** 2).sum(axis=1))  #soma ao longo da coluna


if __name__ == '__main__':
    # test euclidean_distance
    x = np.array([1, 2, 3])                #x é um único ponto de referência com coordenadas conhecidas.
    y = np.array([[1, 2, 3], [4, 5, 6]])   #y é um conjunto de pontos de destino (pode haver um ou mais pontos) nos qual se pretende calcular as distâncias em relação ao ponto x
    our_distance = euclidean_distance(x, y)  #distâncias Euclidianas entre o ponto x e cada ponto em y

    # using sklearn
    # to test this snippet, you need to install sklearn (pip install -U scikit-learn)

    from sklearn.metrics.pairwise import euclidean_distances
    sklearn_distance = euclidean_distances(x.reshape(1, -1), y)  #distâncias Euclidianas entre um único ponto x e um conjunto de pontos y
                                            # O 1 indica que há apenas um ponto no conjunto x. O -  1 é usado para que o NumPy calcule automaticamente o tamanho da segunda dimensão com base no número total de elementos em x.
                                            #como x tem 3 elementos, podia ser (1,3), pois 1x3 = 3 (numero de elementos). Se x fosse [1,2,3,4,5,6], poderia ser (2,-1) ou (2,3), pois 2x3=6 (numero de elementos)
    assert np.allclose(our_distance, sklearn_distance)
    print(our_distance, sklearn_distance) # No our_distance -> a distância entre o ponto x e o primeiro ponto em y (que é o próprio ponto x) é zero.

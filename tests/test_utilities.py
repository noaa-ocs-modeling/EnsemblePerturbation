import numpy

from ensembleperturbation.utilities import encode_categorical_values


def test_encode_categorical_values():
    values_1 = ['apple', 'banana', 'apple', 'orange']
    values_2 = [-5, 20, -1, 0, 0]
    values_3 = numpy.array([[-100, 100], [0, 0], [100, 0],])

    encoded_1 = encode_categorical_values(values_1)
    encoded_2 = encode_categorical_values(values_2)
    encoded_3 = encode_categorical_values(values_3)

    assert numpy.all(encoded_1 == [0, 1, 0, 2])
    assert numpy.all(encoded_2 == [0, 3, 1, 2, 2])
    assert numpy.all(encoded_3 == [[0, 2], [1, 1], [2, 1]])

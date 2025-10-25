import numpy as np
from numpy.testing import assert_allclose
from typing import Callable

class MatrixMultiplicaitonTester:
    def __init__(self, function_to_test: Callable[[np.ndarray, np.ndarray], np.ndarray]):
       self.function = function_to_test

    def run_all_tests(self) -> None:
        self.test_one_by_one_matrices()
        self.test_two_by_two_matrices()
        self.test_three_by_three_matrices()
        self.test_random_matrices_small()
        self.test_random_matrices_large()
        print("Wszystkie testy przeszły!")

    def test_one_by_one_matrices(self) -> None:
        matrix_a = np.array([4])
        matrix_b = np.array([3])
        expected = np.array([12])
        assert_allclose(expected, self.function(matrix_a, matrix_b))

    def test_two_by_two_matrices(self) -> None:
        matrix_a = np.array([[1, 2], [3, 4]])
        matrix_b = np.array([[5, 6], [7, 8]])
        expected = np.array([[19, 22], [43, 50]])
        assert_allclose(expected, self.function(matrix_a, matrix_b))

    def test_three_by_three_matrices(self) -> None:
        matrix_a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        matrix_b = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
        expected = np.array([[30, 24, 18], [84, 69, 54], [138, 114, 90]])
        assert_allclose(expected, self.function(matrix_a, matrix_b))

    def test_random_matrices_small(self) -> None:
        np.random.seed(0)
        matrix_a = np.random.rand(10, 10)
        matrix_b = np.random.rand(10, 10)
        expected = np.dot(matrix_a, matrix_b)
        assert_allclose(expected, self.function(matrix_a, matrix_b))

    def test_random_matrices_large(self) -> None:
        np.random.seed(0)
        matrix_a = np.random.rand(100, 100)
        matrix_b = np.random.rand(100, 100)
        expected = np.dot(matrix_a, matrix_b)
        assert_allclose(expected, self.function(matrix_a, matrix_b))

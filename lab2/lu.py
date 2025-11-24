from copy import deepcopy

import numpy as np

from lab1.binet import BinetWrapper
from lab1.strassen import StrassenWrapper
from lab2.inverse import InverseWrapper


class LUWrapper:
    def __init__(self, matrix_multiplier: BinetWrapper | StrassenWrapper):
        self.matmul = matrix_multiplier
        self.inverse = InverseWrapper(deepcopy(matrix_multiplier))
        self.flops = 0
        self.memory_used = 0
        self.time_used = []

    def __call__(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.lu(matrix)

    def lu(self, matrix: np.ndarray, top_call: bool=True) -> tuple[np.ndarray, np.ndarray]:
        """
        Funkcja przeprowadzająca rozkład LU podanej macierzy.
        :param matrix: Macierz wejściowa.
        :param top_call: Informacja, czy jest to zawołanie rekurencyjne funkcji (false), czy pierwsze, z innego
        kawałka kodu (true).
        :return: Krotka macierzy L i U.
        """
        if matrix.shape[0] == 1 and matrix.shape[1] == 1:
            return np.ones_like(matrix), matrix

        a11, a12, a21, a22 = self.split(matrix)

        l11, u11 = self.lu(a11, False)

        u11_rev = self.inverse.inverse(u11)

        l21 = self.matmul(a21, u11_rev)

        l11_rev = self.inverse.inverse(l11)

        u12 = self.matmul(l11_rev, a12)

        l22 = a22 - self.matmul(a21, u11_rev, l11_rev, a12)
        self.memory_used += l22.nbytes
        self.flops += a22.shape[0] * a22.shape[1]

        ls, us = self.lu(l22, False)

        u22 = us
        l22 = ls

        if top_call:
            self.flops += self.inverse.flops + self.matmul.flops
            self.memory_used += self.inverse.memory_used + self.matmul.memory_used

        lu_tuple = (np.vstack((np.hstack((l11, np.zeros((l11.shape[0], l22.shape[1])))), np.hstack((l21, l22)))),
                np.vstack((np.hstack((u11, u12)), np.hstack((np.zeros((u22.shape[0], u11.shape[1])), u22)))))
        self.memory_used += sum(i.nbytes for i in lu_tuple)
        return lu_tuple

    def det(self, matrix: np.ndarray) -> float:
        """
        Funkcja obliczająca wyznacznik macierzy za pomocą rozkładu LU.
        :param matrix: Macierz wejściowa.
        :return: Wyznacznik macierzy.
        """
        _, u = self.lu(matrix)
        det = 1
        for num in [u[i][i] for i in range(u.shape[0])]:
            det *= num
            self.flops += 1
        self.flops -= 1
        return det

    def det_lu(self, u_matrix: np.ndarray) -> float:
        """
        Funkcja obliczająca wyznacznik macierzy trójkątnej górnej.
        :param u_matrix: Macierz trójkątna górna.
        :return: Wyznacznik macierzy.
        """
        det = 1
        for i, num in enumerate(u_matrix):
            det *= num[i]
            self.flops += 1
        self.flops -= 1
        return det

    def split(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Funkcja rozdzielająca macierz na 4 (w miarę możliwości równe) części.
        :param matrix: Macierz do rozdzielenia.
        :return: 4 macierze powstałe przez rozdział tej oryginalnej
        """
        horizontal_split_point = matrix.shape[1] // 2
        vertical_split_point = matrix.shape[0] // 2

        self.memory_used += matrix.nbytes  # Tworzymy 4 nowe macierze, ale w gruncie rzeczy one razem zajmują tyle pamięci co ta oryginalna
        return (matrix[:vertical_split_point, :horizontal_split_point],
                matrix[:vertical_split_point, horizontal_split_point:],
                matrix[vertical_split_point:, :horizontal_split_point],
                matrix[vertical_split_point:, horizontal_split_point:])

if __name__ == "__main__":
    # Tests
    strassen_wrapper = StrassenWrapper()
    lu_strassen_wrapper = LUWrapper(strassen_wrapper)
    binet_wrapper = BinetWrapper()
    lu_binet_wrapper = LUWrapper(binet_wrapper)

    matrix = np.array([[1, 2], [3, 4]])
    print(lu_strassen_wrapper.lu(matrix))
    assert lu_strassen_wrapper.det(matrix) == -2, "Det Strassen źle"
    _, u = lu_strassen_wrapper.lu(matrix)
    assert lu_strassen_wrapper.det_lu(u) == -2, "Det_LU Strassen źle"

    for i in [1, 2, 3, 4, 5, 8, 16, 20]:
        matrix = np.random.rand(i, i)
        L_strassen, U_strassen = lu_strassen_wrapper(matrix)
        L_binet, U_binet = lu_binet_wrapper(matrix)
        assert np.allclose(matrix, lu_strassen_wrapper.matmul(L_strassen, U_strassen)), f"LU Strassen źle dla {matrix}, {L_strassen}, {U_strassen}"
        assert np.allclose(matrix, lu_binet_wrapper.matmul(L_binet, U_binet)), f"Binet źle dla {matrix}, {L_binet}, {U_binet}"
    print("All tests passed.")
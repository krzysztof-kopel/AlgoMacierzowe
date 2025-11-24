import numpy as np

class BinetWrapper:
    def __init__(self):
        self.flops = 0
        self.memory_used = 0
        self.time_used = [] # Tablica, aby można było mierzyć czas wielokrotnie i brać medianę

    def __call__(self, *args: tuple[np.ndarray]) -> np.ndarray:
        result = args[0]
        for arg in args[1:]:
            result = self.binet(result, arg)
        return result

    def binet(self, matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
        """
        Rekurencyjna funkcja mnożąca dwie macerze metodą Bineta.
        :param matrix_a: Pierwsza macierz
        :param matrix_b: Druga macierz
        :return: Wynik mnożenia macierzy
        """
        if min(matrix_a.shape) == 1 or min(matrix_b.shape) == 1:
            result = np.zeros((matrix_a.shape[0], matrix_b.shape[1]))
            self.memory_used += result.nbytes
            for i in range(matrix_a.shape[0]):
                for j in range(matrix_b.shape[1]):
                    for k in range(matrix_a.shape[1]):
                        result[i, j] += matrix_a[i, k] * matrix_b[k, j]
                        self.flops += 2
                    self.flops -= 1
            return result


        a11, a12, a21, a22 = self.split(matrix_a)
        b11, b12, b21, b22 = self.split(matrix_b)

        prod_1 = self.binet(a11, b11)
        prod_2 = self.binet(a12, b21)
        c1 = prod_1 + prod_2
        self.flops += prod_1.shape[0] * prod_1.shape[1]

        prod_3 = self.binet(a11, b12)
        prod_4 = self.binet(a12, b22)
        c2 = prod_3 + prod_4
        self.flops += prod_3.shape[0] * prod_3.shape[1]

        prod_5 = self.binet(a21, b11)
        prod_6 = self.binet(a22, b21)
        c3 = prod_5 + prod_6
        self.flops += prod_5.shape[0] * prod_5.shape[1]

        prod_7 = self.binet(a21, b12)
        prod_8 = self.binet(a22, b22)
        c4 = prod_7 + prod_8
        self.flops += prod_7.shape[0] * prod_7.shape[1]

        self.memory_used += c1.nbytes + c2.nbytes + c3.nbytes + c4.nbytes
        return np.vstack((np.hstack((c1, c2)), np.hstack((c3, c4))))

    def split(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Funkcja rozdzielająca macierz na 4 (w miarę możliwości równe) części.
        :param matrix: Macierz do rozdzielenia.
        :return: 4 macierze powstałe przez rozdział tej oryginalnej
        """
        horizontal_split_point = matrix.shape[1] // 2
        vertical_split_point = matrix.shape[0] // 2

        self.memory_used += matrix.nbytes # Tworzymy 4 nowe macierze, ale w gruncie rzeczy one razem zajmują tyle pamięci co ta oryginalna
        return (matrix[:vertical_split_point, :horizontal_split_point],
                matrix[:vertical_split_point, horizontal_split_point:],
                matrix[vertical_split_point:, :horizontal_split_point],
                matrix[vertical_split_point:, horizontal_split_point:])


if __name__ == "__main__":
    from lab1.tests import MatrixMultiplicaitonTester

    binet_wrapper = BinetWrapper()
    tester = MatrixMultiplicaitonTester(binet_wrapper.binet)
    tester.run_all_tests()

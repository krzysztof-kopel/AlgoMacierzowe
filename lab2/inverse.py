import numpy as np

from lab1.binet import BinetWrapper
from lab1.strassen import StrassenWrapper

class InverseWrapper:
    def __init__(self, matrix_multiplier: BinetWrapper | StrassenWrapper):
        self.matmul = matrix_multiplier
        self.flops = 0
        self.memory_used = 0
        self.time_used = []

    def __call__(self, matrix: np.ndarray, top_call: bool = True) -> np.ndarray:
        return self.inverse(matrix, top_call=top_call)

    def inverse(self, matrix: np.ndarray, top_call: bool=True) -> np.ndarray:
        """
        Funkcja odwracająca macierz z wykorzystaniem sposobu podanego na wykładzie.
        :param matrix: Macierz do odwrócenia
        :param top_call: Informacja, czy jest to zawołanie rekurencyjne funkcji (false), czy pierwsze, z innego
        kawałka kodu (true).
        :return: Macierz odwrotna do podanej
        """
        if top_call:
            self.matmul.flops = 0
            self.matmul.memory_used = 0

        if tuple(matrix.shape) == (1, 1):
            self.flops += 1
            return np.array([[1 / matrix[0][0]]])

        a11, a12, a21, a22 = self.split(matrix)

        a11_rev = self.inverse(a11, top_call=False)

        s22 = a22 - self.matmul(a21, a11_rev, a12)
        self.flops += a22.shape[0] * a22.shape[1]
        self.memory_used += s22.nbytes

        s22_rev = self.inverse(s22, top_call=False)

        b11 = self.matmul(a11_rev, np.eye(a11_rev.shape[0], a11_rev.shape[1]) + self.matmul(a12, s22_rev, a21, a11_rev))
        self.flops += a11_rev.shape[0] * a11_rev.shape[1]
        self.memory_used += b11.nbytes

        b12 = -1 * self.matmul(a11_rev, a12, s22_rev)
        self.flops += b12.shape[0] * b12.shape[1]
        self.memory_used += b12.nbytes

        b21 = -1 * self.matmul(s22_rev, a21, a11_rev)
        self.flops += b21.shape[0] * b21.shape[1]
        self.memory_used += b21.nbytes

        if top_call:
            self.memory_used += self.matmul.memory_used
            self.flops += self.matmul.flops

        self.memory_used += b11.nbytes + b12.nbytes + b21.nbytes + s22_rev.nbytes
        return np.vstack((np.hstack((b11, b12)), np.hstack((b21, s22_rev))))


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
    # Testy
    strassen_wrapper = StrassenWrapper()
    inverse_strassen_wrapper = InverseWrapper(strassen_wrapper)

    binet_wrapper = BinetWrapper()
    inverse_binet_wrapper = InverseWrapper(binet_wrapper)

    matrix = np.array([[1, 2], [3, 4]])
    print(inverse_binet_wrapper.inverse(matrix))

    for size in [1, 2, 3, 4, 5, 8, 16, 20]:
        A = np.random.rand(size, size) * 10 + 1
        A_inv_strassen = inverse_strassen_wrapper.inverse(A)
        A_inv_binet = inverse_binet_wrapper.inverse(A)
        
        print("Binet: ", inverse_binet_wrapper.flops)
        print("Strassen: ", inverse_strassen_wrapper.flops)

        identity_strassen = strassen_wrapper(A, A_inv_strassen)
        identity_binet = binet_wrapper(A, A_inv_binet)
        
        assert np.allclose(identity_strassen, np.eye(size), atol=1e-5), f"Strassen źle dla {size}: {A}, {A_inv_strassen}, {identity_strassen}"
        assert np.allclose(identity_binet, np.eye(size), atol=1e-5), f"Binet źle dla {size}: {A}, {A_inv_binet}, {identity_binet}"
    print("Wszystkie testy przeszły pomyślnie.")
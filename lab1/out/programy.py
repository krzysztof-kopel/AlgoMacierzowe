import numpy as np

class BinetWrapper:
    def __init__(self):
        self.flops = 0
        self.memory_used = 0
        self.time_used = [] # Tablica, aby można było mierzyć czas wielokrotnie i brać medianę

    def binet(self, matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
        """
        Rekurencyjna funkcja mnożąca dwie macierze metodą Bineta.
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
            return result


        a11, a12, a21, a22 = self.split(matrix_a)
        b11, b12, b21, b22 = self.split(matrix_b)

        prod_1 = self.binet(a11, b11)
        prod_2 = self.binet(a12, b21)
        c1 = prod_1 + prod_2
        self.flops += prod_1.shape[0] ** 2

        prod_3 = self.binet(a11, b12)
        prod_4 = self.binet(a12, b22)
        c2 = prod_3 + prod_4
        self.flops += prod_3.shape[0] ** 2

        prod_5 = self.binet(a21, b11)
        prod_6 = self.binet(a22, b21)
        c3 = prod_5 + prod_6
        self.flops += prod_5.shape[0] ** 2

        prod_7 = self.binet(a21, b12)
        prod_8 = self.binet(a22, b22)
        c4 = prod_7 + prod_8
        self.flops += prod_7.shape[0] ** 2

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
        self.flops += 2

        self.memory_used += matrix.nbytes # Tworzymy 4 nowe macierze, ale w gruncie rzeczy one razem zajmują tyle pamięci co ta oryginalna
        return (matrix[:vertical_split_point, :horizontal_split_point],
                matrix[:vertical_split_point, horizontal_split_point:],
                matrix[vertical_split_point:, :horizontal_split_point],
                matrix[vertical_split_point:, horizontal_split_point:])



class StrassenWrapper:
    def __init__(self):
        self.flops = 0
        self.memory_used = 0
        self.time_used = []

    def pad_matrix(self, A, new_shape):
        """
        Funkcja do uzupełnia macierzy prostokątnej A do rozmiaru kwadratowego new_shape
        """
        m, n = A.shape
        m2, n2 = new_shape
        P = np.zeros((m2, n2), dtype=A.dtype)
        P[:m, :n] = A
        self.memory_used += P.nbytes
        return P

    def pad_matrix_even(self, A):
        """
        Funkcja do uzupełniania macierzy kwadratowej A o nieparzystym rozmiarze zerami tak aby rozmaiar był parzysty (jednen wiersz i kolumna zer)
        """
        n = A.shape[0]
        if n % 2 == 0:
            return A
        P = np.zeros((n + 1, n + 1), dtype=A.dtype)
        P[:n, :n] = A
        self.memory_used += P.nbytes
        return P

    def matmul(self, A, B):
        """
        Funkcja klasycznego mnożenia macierzy
        """
        assert A.shape[1] == B.shape[0], "Incorrect sizes"
        m, n, p = A.shape[0], A.shape[1], B.shape[1]
        C = np.zeros((m, p), dtype=A.dtype)
        self.memory_used += C.nbytes
        for i in range(m):
            for j in range(p):
                for k in range(n):
                    C[i, j] += A[i, k] * B[k, j]
                    self.flops += 2
        return C

    def _strassen_flexible(self, A, B):
        """
        Funkcja mnożenia dwóch macierzy kwadratowych metodą Strassena, dzięki odpowiedniemu paddingowi zerami (nie zachłannemu),
        Można używać jej dla każdych dwóch odpowiadających macierzy kwadratowych, a nie tylko do takich o rozmiarze 2^n, gdzie n to liczba naturalna.
        """
        n = A.shape[0]

        if n % 2 != 0:
            A = self.pad_matrix_even(A)
            B = self.pad_matrix_even(B)
            n += 1

        if n <= 4:
            return self.matmul(A, B)

        k = n // 2

        A11, A12, A21, A22 = A[:k, :k], A[:k, k:], A[k:, :k], A[k:, k:]
        B11, B12, B21, B22 = B[:k, :k], B[:k, k:], B[k:, :k], B[k:, k:]
        self.flops += 18 * (k ** 2)

        def felxible_multiply(X, Y):
            if X.shape == Y.shape and X.shape[0] == X.shape[1] and X.shape[0] > 4:
                return self.strassen(X, Y)
            else:
                return self.matmul(X, Y)

        P1 = felxible_multiply(A11 + A22, B11 + B22)
        P2 = felxible_multiply(A21 + A22, B11)
        P3 = felxible_multiply(A11, B12 - B22)
        P4 = felxible_multiply(A22, B21 - B11)
        P5 = felxible_multiply(A11 + A12, B22)
        P6 = felxible_multiply(A21 - A11, B11 + B12)
        P7 = felxible_multiply(A12 - A22, B21 + B22)

        C11 = P1 + P4 - P5 + P7
        C12 = P3 + P5
        C21 = P2 + P4
        C22 = P1 - P2 + P3 + P6

        self.flops += 8 * (k ** 2)

        C = np.vstack([
            np.hstack([C11, C12]),
            np.hstack([C21, C22])
        ])
        self.memory_used += C.nbytes
        return C[:n, :n]

    def strassen(self, A, B):
        """
        Funkcja mnożenia macierzy metodą Strassena, zoptymalizowana dla macierzy dowolnych rozmiarów.
        Przyjmuje macierze A i B o dowolnym rozmiarze, zwraca macierz C będącą wynikiem mnożenia A i B
        """

        if A.ndim == 1:
            A = A.reshape(1, 1)
        if B.ndim == 1:
            B = B.reshape(1, 1)
        n, m = A.shape
        m2, p = B.shape
        assert m == m2, "Incorrect sizes"

        q = max(m, n, p)

        A_pad = self.pad_matrix(A, (q, q))
        B_pad = self.pad_matrix(B, (q, q))

        C_pad = self._strassen_flexible(A_pad, B_pad)

        C = C_pad[:n, :p].astype(A.dtype)

        self.memory_used += C.nbytes

        return C

from math import log

class AIWrapper:
    def __init__(self):
        self.flops = 0
        self.memory_used = 0
        self.time_used = []

    def ai_matrix_multiply_strict(self, A, B):
        """
        Funkcja mnożenia macierzy o rozmiarach 4x5, 5x5,
        korzystająca ze sposobu opisanego w artykule AI killing Strassen, w czasopiśmie Nature.
        """
        assert A.shape == (4, 5) and B.shape == (5,
                                                 5), "Incorrect sizes for this multiplication method, should be 4x5 and 5x5"
        H1 = A[2, 1] * (-B[1, 0] - B[1, 4] - B[2, 0])
        H2 = (A[1, 1] + A[1, 4] - A[2, 4]) * (-B[1, 4] - B[4, 0])
        H3 = (-A[2, 0] - A[3, 0] + A[3, 1]) * (-B[0, 0] + B[1, 4])
        H4 = (A[0, 1] + A[0, 3] + A[2, 3]) * (-B[1, 4] - B[3, 0])
        H5 = (A[0, 4] + A[1, 1] + A[1, 4]) * (-B[1, 3] + B[4, 0])
        H6 = (-A[1, 1] - A[1, 4] - A[3, 4]) * (B[1, 2] + B[4, 0])
        H7 = (-A[0, 0] + A[3, 0] - A[3, 1]) * (B[0, 0] + B[1, 3])
        H8 = (A[2, 1] - A[2, 2] - A[3, 2]) * (-B[1, 2] + B[2, 0])
        H9 = (-A[0, 1] - A[0, 3] + A[3, 3]) * (B[1, 2] + B[3, 0])
        H10 = (A[1, 1] + A[1, 4]) * B[4, 0]
        H11 = (-A[1, 0] - A[3, 0] + A[3, 1]) * (-B[0, 0] + B[1, 1])
        H12 = (A[3, 0] - A[3, 1]) * B[0, 0]
        H13 = (A[0, 1] + A[0, 3] + A[1, 3]) * (B[1, 1] + B[3, 0])
        H14 = (A[0, 2] - A[2, 1] + A[2, 2]) * (B[1, 3] + B[2, 0])
        H15 = (-A[0, 1] - A[0, 3]) * B[3, 0]
        H16 = (-A[2, 1] + A[2, 2]) * B[2, 0]
        H17 = (A[0, 1] + A[0, 3] - A[1, 0] + A[1, 1] - A[1, 2] + A[1, 3] - A[2, 1] + A[2, 2] - A[3, 0] + A[3, 1]) * B[
            1, 1]
        H18 = A[1, 0] * (B[0, 0] + B[0, 1] + B[4, 1])
        H19 = -A[1, 2] * (B[2, 0] + B[2, 1] + B[4, 1])
        H20 = (-A[0, 4] + A[1, 0] + A[1, 2] - A[1, 4]) * (-B[0, 0] - B[0, 1] + B[0, 3] - B[4, 1])
        H21 = (A[1, 0] + A[1, 2] - A[1, 4]) * B[4, 1]
        H22 = (A[0, 2] - A[0, 3] - A[1, 3]) * (B[0, 0] + B[0, 1] - B[0, 3] - B[2, 0] - B[2, 1] + B[2, 3] + B[3, 3])
        H23 = A[0, 2] * (-B[2, 0] + B[2, 3] + B[3, 3])
        H24 = A[0, 4] * (-B[3, 3] - B[4, 0] + B[4, 3])
        H25 = -A[0, 0] * (B[0, 0] - B[0, 3])
        H26 = (-A[0, 2] + A[0, 3] + A[0, 4]) * B[3, 3]
        H27 = (A[0, 2] - A[2, 0] + A[2, 2]) * (B[0, 0] - B[0, 3] + B[0, 4] + B[2, 4])
        H28 = -A[2, 3] * (-B[2, 4] - B[3, 0] - B[3, 4])
        H29 = A[2, 0] * (B[0, 0] + B[0, 4] + B[2, 4])
        H30 = (A[2, 0] - A[2, 2] + A[2, 3]) * B[2, 4]
        H31 = (-A[0, 3] - A[0, 4] - A[2, 3]) * (-B[3, 3] - B[4, 0] + B[4, 3] - B[4, 4])
        H32 = (A[1, 0] + A[3, 0] + A[3, 3]) * (B[0, 2] - B[3, 0] - B[3, 1] - B[3, 2])
        H33 = A[3, 2] * (-B[2, 0] - B[2, 2])
        H34 = A[3, 3] * (-B[0, 2] + B[3, 0] + B[3, 2])
        H35 = -A[3, 4] * (B[0, 2] + B[4, 0] + B[4, 2])
        H36 = (A[1, 2] - A[1, 4] - A[3, 4]) * (B[2, 0] + B[2, 1] + B[2, 2] + B[4, 1])
        H37 = (-A[3, 0] - A[3, 3] + A[3, 4]) * B[0, 2]
        H38 = (-A[1, 2] - A[2, 0] + A[2, 2] - A[2, 3]) * (B[2, 4] + B[3, 0] + B[3, 1] + B[3, 4])
        H39 = (-A[2, 0] - A[3, 0] - A[3, 3] + A[3, 4]) * (B[0, 2] + B[4, 0] + B[4, 2] + B[4, 4])
        H40 = (-A[0, 2] + A[0, 3] + A[0, 4] - A[3, 3]) * (-B[2, 0] - B[2, 2] + B[2, 3] + B[3, 3])
        H41 = (-A[0, 0] + A[3, 0] - A[3, 4]) * (B[0, 2] + B[2, 0] + B[2, 2] - B[2, 3] + B[4, 0] + B[4, 2] - B[4, 3])
        H42 = (-A[1, 0] + A[1, 4] - A[2, 4]) * (-B[0, 0] - B[0, 1] - B[0, 4] + B[3, 0] + B[3, 1] + B[3, 4] - B[4, 1])
        H43 = A[1, 3] * (B[3, 0] + B[3, 1])
        H44 = (A[1, 2] + A[2, 1] - A[2, 2]) * (B[1, 1] - B[2, 0])
        H45 = (-A[2, 2] + A[2, 3] - A[3, 2]) * (B[2, 4] + B[3, 0] + B[3, 2] + B[3, 4] + B[4, 0] + B[4, 2] + B[4, 4])
        H46 = -A[2, 4] * (-B[4, 0] - B[4, 4])
        H47 = (A[1, 0] - A[1, 4] - A[2, 0] + A[2, 4]) * (B[0, 0] + B[0, 1] + B[0, 4] - B[3, 0] - B[3, 1] - B[3, 4])
        H48 = (-A[1, 2] + A[2, 2]) * (B[1, 1] + B[2, 1] + B[2, 3] + B[3, 0] + B[3, 1] + B[3, 4])
        H49 = (-A[0, 0] - A[0, 2] + A[0, 3] + A[0, 4] - A[1, 0] - A[1, 2] + A[1, 3] + A[1, 4]) * (
                    -B[0, 0] - B[0, 1] + B[0, 3])
        H50 = (-A[0, 3] - A[1, 3]) * (B[1, 1] - B[2, 0] - B[2, 1] + B[2, 3] - B[3, 1] + B[3, 3])
        H51 = A[1, 1] * (B[1, 0] + B[1, 1] - B[4, 0])
        H52 = A[3, 1] * (B[0, 0] + B[1, 0] + B[1, 2])
        H53 = -A[0, 1] * (-B[1, 0] + B[1, 3] + B[3, 0])
        H54 = (A[0, 1] + A[0, 3] - A[1, 1] - A[1, 4] - A[2, 1] + A[2, 2] - A[3, 1] + A[3, 2] - A[3, 3] - A[3, 4]) * B[
            1, 2]
        H55 = (A[0, 3] - A[3, 3]) * (-B[1, 2] + B[2, 0] + B[2, 2] - B[2, 3] + B[4, 2] - B[3, 3])
        H56 = (A[0, 0] - A[0, 4] - A[3, 0] + A[3, 4]) * (B[2, 0] + B[2, 2] - B[2, 3] + B[4, 0] + B[4, 2] - B[4, 3])
        H57 = (-A[2, 0] - A[3, 0]) * (-B[0, 2] - B[0, 4] - B[1, 4] - B[4, 0] - B[4, 2] - B[4, 4])
        H58 = (-A[0, 3] - A[0, 4] - A[2, 3] - A[2, 4]) * (-B[4, 0] + B[4, 3] - B[4, 4])
        H59 = (-A[2, 2] + A[2, 3] - A[3, 2] + A[3, 3]) * (B[3, 0] + B[3, 2] + B[3, 4] + B[4, 0] + B[4, 2] + B[4, 4])
        H60 = (A[1, 4] + A[3, 4]) * (B[1, 2] - B[2, 0] - B[2, 1] - B[2, 2] - B[4, 1] - B[4, 2])
        H61 = (A[0, 3] + A[2, 3]) * (
                    B[0, 0] - B[0, 3] + B[0, 4] - B[1, 4] - B[3, 3] + B[3, 4] - B[4, 0] + B[4, 3] - B[4, 4])
        H62 = (A[1, 0] + A[3, 0]) * (B[0, 1] + B[0, 2] + B[1, 1] - B[3, 0] - B[3, 1] - B[3, 2])
        H63 = (-A[2, 2] - A[3, 2]) * (-B[1, 2] - B[2, 2] - B[2, 4] - B[3, 0] - B[3, 2] - B[3, 4])
        H64 = (A[0, 0] - A[0, 2] - A[0, 3] + A[2, 0] - A[2, 2] - A[2, 3]) * (B[0, 0] - B[0, 3] + B[0, 4])
        H65 = (-A[0, 0] + A[3, 0]) * (-B[0, 2] + B[0, 3] + B[1, 3] - B[4, 0] + B[4, 3])
        H66 = (A[0, 0] - A[0, 1] + A[0, 2] - A[0, 4] - A[1, 1] - A[1, 4] - A[2, 1] + A[2, 2] - A[3, 0] + A[3, 1]) * B[
            1, 3]
        H67 = (A[1, 4] - A[2, 4]) * (
                    B[0, 0] + B[0, 1] + B[0, 4] - B[1, 4] - B[3, 0] - B[3, 1] - B[3, 4] + B[4, 1] + B[4, 4])
        H68 = (A[0, 0] + A[0, 2] - A[0, 3] - A[0, 4] - A[3, 0] - A[3, 2] + A[3, 3] + A[3, 4]) * (
                    -B[2, 0] - B[2, 2] + B[2, 3])
        H69 = (-A[0, 2] + A[0, 3] - A[1, 2] + A[1, 3]) * (-B[1, 3] - B[2, 0] - B[2, 1] + B[2, 3] - B[4, 1] + B[4, 3])
        H70 = (A[1, 2] - A[1, 4] + A[3, 2] - A[3, 4]) * (-B[2, 0] - B[2, 1] - B[2, 2])
        H71 = (-A[2, 0] + A[2, 2] - A[2, 3] + A[2, 4] - A[3, 0] + A[3, 2] - A[3, 3] + A[3, 4]) * (
                    -B[4, 0] - B[4, 2] - B[4, 4])
        H72 = (-A[1, 0] - A[1, 3] - A[3, 0] - A[3, 3]) * (B[3, 0] + B[3, 1] + B[3, 2])
        H73 = (A[0, 2] - A[0, 3] - A[0, 4] + A[1, 2] - A[1, 3] - A[1, 4]) * (
                    B[0, 0] + B[0, 1] - B[0, 3] + B[1, 3] + B[4, 1] - B[4, 3])
        H74 = (A[1, 0] - A[1, 2] + A[1, 3] - A[2, 0] + A[2, 2] - A[2, 3]) * (B[3, 0] + B[3, 1] + B[3, 4])
        H75 = -(A[0, 1] + A[0, 3] - A[1, 1] - A[1, 4] - A[2, 0] + A[2, 1] + A[2, 3] + A[2, 4] - A[3, 0] + A[3, 1]) * B[
            1, 4]
        H76 = (A[0, 2] + A[2, 2]) * (-B[0, 0] + B[0, 3] - B[0, 4] + B[1, 3] + B[2, 3] - B[2, 4])
        C = np.zeros((4, 5))
        C[0, 0] = int(-H10 + H12 + H14 - H15 - H16 + H53 + H5 - H66 - H7)
        C[1, 0] = int(H10 + H11 - H12 + H13 + H15 + H16 - H17 - H44 + H51)
        C[2, 0] = int(H10 - H12 + H15 + H16 - H1 + H2 + H3 - H4 + H75)
        C[3, 0] = int(-H10 + H12 - H15 - H16 + H52 + H54 - H6 - H8 + H9)
        C[0, 1] = int(H13 + H15 + H20 + H21 - H22 + H23 + H25 - H43 + H49 + H50)
        C[1, 1] = int(-H11 + H12 - H13 - H15 - H16 + H17 + H18 - H19 - H21 + H43 + H44)
        C[2, 1] = int(-H16 - H19 - H21 - H28 - H29 - H38 + H42 + H44 - H47 + H48)
        C[3, 1] = int(H11 - H12 - H18 + H21 - H32 + H33 - H34 - H36 + H62 - H70)
        C[0, 2] = int(H15 + H23 + H24 + H34 - H37 + H40 - H41 + H55 - H56 - H9)
        C[1, 2] = int(-H10 + H19 + H32 + H35 + H36 + H37 - H43 - H60 - H6 - H72)
        C[2, 2] = int(-H16 - H28 + H33 + H37 - H39 + H45 - H46 + H63 - H71 - H8)
        C[3, 2] = int(-H10 + H15 + H16 - H33 + H34 - H35 - H37 - H54 + H6 + H8 - H9)
        C[0, 3] = int(-H10 + H12 + H14 - H16 + H23 + H24 + H25 + H26 + H5 - H66 - H7)
        C[1, 3] = int(-H10 + H18 - H19 + H20 - H22 - H24 - H26 - H5 - H69 + H73)
        C[2, 3] = int(-H14 + H16 - H23 - H26 + H27 + H29 + H31 + H46 - H58 + H76)
        C[3, 3] = int(H12 + H25 + H26 - H33 - H35 - H40 + H41 + H65 - H68 - H7)
        C[0, 4] = int(H15 + H24 + H25 + H27 - H28 + H30 + H31 - H4 + H61 + H64)
        C[1, 4] = int(-H10 - H18 - H2 - H30 - H38 + H42 - H43 + H46 + H67 + H74)
        C[2, 4] = int(-H10 + H12 - H15 + H28 + H29 - H2 - H30 - H3 + H46 + H4 - H75)
        C[3, 4] = int(-H12 - H29 + H30 - H34 + H35 + H39 + H3 - H45 + H57 + H59)
        self.flops += 539
        return C

    def _ai_matrix_multiply(self, A, B):
        """
        Rozszerzenie funkcji ai_matrix_multiply_strict,
        wykorzystujące ją do mnożenia między sobą dwóch macierzy
        o rozmiarach będących potęgami rozmiarów 4x5, 5x5.
        """
        assert (
                A.shape[1] == 5 ** round(log(A.shape[1], 5)) and
                A.shape[0] == 4 ** round(log(A.shape[0], 4)) and
                B.shape[0] == 5 ** round(log(B.shape[0], 5)) and
                B.shape[1] == 5 ** round(log(B.shape[1], 5)) and
                A.shape[1] == B.shape[0]
        ), "Incorrect matrix sizes, must be 4x5, 5x5, or their powers"

        def ai_matrix_multiply_rec(A, B):
            if A.shape == (4, 5) and B.shape == (5, 5):
                return self.ai_matrix_multiply_strict(A, B)
            else:
                n, m = A.shape
                k4 = n // 4
                k5 = m // 5
                A_blocks = [[A[i * k4:(i + 1) * k4, j * k5:(j + 1) * k5] for j in range(5)] for i in range(4)]
                B_blocks = [[B[i * k5:(i + 1) * k5, j * k5:(j + 1) * k5] for j in range(5)] for i in range(5)]
                C_blocks = [[np.zeros((4, 5)) for _ in range(5)] for _ in range(4)]
                for i in range(4):
                    for j in range(5):
                        for k in range(5):
                            C_blocks[i][j] = ai_matrix_multiply_rec(A_blocks[i][k], B_blocks[k][j])
                C = np.block(C_blocks)
                return C

        return ai_matrix_multiply_rec(A, B)

    def ai_matrix_multiply(self, A, B):
        """
        Opakowanie funkcji _ai_matrix_multiply,
        liczące czas trwania, flops, oraz zużytą pamięć.
        """
        self.memory_used += A.nbytes + B.nbytes
        C = self._ai_matrix_multiply(A, B)
        self.memory_used += C.nbytes
        return C

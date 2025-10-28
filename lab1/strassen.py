import numpy as np
from tests import MatrixMultiplicaitonTester

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

        if C.shape[0] == 1 or C.shape[1] == 1:
            return C.reshape(-1)
        else:
            return C


if __name__ == "__main__":
    strassen_wrapper = StrassenWrapper()
    tester = MatrixMultiplicaitonTester(strassen_wrapper.strassen)
    tester.run_all_tests()

    print(f"FLOPS: {strassen_wrapper.flops}")
    print(f"Memory used: {strassen_wrapper.memory_used / 1024:.2f} KB")
    if strassen_wrapper.time_used:
        print(f"Median time: {np.median(strassen_wrapper.time_used):.6f} s")

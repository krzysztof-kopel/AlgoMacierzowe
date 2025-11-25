import numpy as np

from binet import BinetWrapper
from strassen import StrassenWrapper
from inverse import InverseWrapper
from lu import LUWrapper

from copy import deepcopy

class GaussWrapper:
    def __init__(self, matrix_multiplier: BinetWrapper | StrassenWrapper):
        self.matmul = matrix_multiplier
        self.LU = LUWrapper(matrix_multiplier=deepcopy(matrix_multiplier))
        self.inverse = InverseWrapper(matrix_multiplier=deepcopy(matrix_multiplier))
        self.flops = 0
        self.memory_used = 0
        self.time_used = []

    def gaussElimination(self, matrix: np.ndarray, topCall: bool = True) -> np.ndarray:
        """
        Rekurencyjna blokowa eliminacja Gaussa: dzieli macierz na bloki,
        wykonuje eliminację na bloku A11, oblicza dopełnienie Schura
        i rekurencyjnie przetwarza blok dolny prawy. Zwracas
        macierz po jednym pełnym kroku eliminacji blokowej.
        """
        A = matrix[:, :-1]
        b = matrix[:, -1]

        if A.shape == (1,1):
            return matrix

        A11, A12, A21, A22 = self.split(A)
        self.memory_used += matrix.nbytes
        n1 = A11.shape[0]

        b1 = b[:n1].reshape(-1, 1)
        b2 = b[n1:].reshape(-1, 1)

        L11, U11 = self.LU(A11)
        L11_inv = self.inverse(L11)
        U11_inv = self.inverse(U11)

        A11p = U11
        A12p = self.matmul(L11_inv, A12) 
        A21p = np.zeros_like(A21)

        S = A22 - self.matmul(A21, U11_inv, A12p)
        self.memory_used += S.nbytes
        self.flops += A22.shape[0] * A22.shape[1]

        b1p = self.matmul(L11_inv, b1)
        
        b2p = b2 - self.matmul(A21, U11_inv, b1p)
        self.flops += b2.shape[0] * b2.shape[1]

        bottom = self.gaussElimination(np.column_stack((S, b2p)), topCall=False)

        A22p = bottom[:, :-1]
        b2pp = bottom[:, -1].reshape(-1, 1)

        A_top = np.hstack((A11p, A12p))
        A_bottom = np.hstack((A21p, A22p))
        A_new = np.vstack((A_top, A_bottom))

        b_new = np.concatenate((b1p, b2pp))

        self.memory_used += A_top.nbytes
        self.memory_used += A_bottom.nbytes
        self.memory_used += A_new.nbytes
        self.memory_used += b_new.nbytes
        if topCall:
            self.flops += self.inverse.flops + self.matmul.flops + self.LU.flops
            self.memory_used += self.inverse.memory_used + self.matmul.memory_used + self.LU.memory_used

        return np.column_stack((A_new, b_new))

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
    strassen_wrapper = StrassenWrapper()
    gauss_strassen_wrapper = GaussWrapper(strassen_wrapper)

    binet_wrapper = BinetWrapper()
    gauss_binet_wrapper = GaussWrapper(binet_wrapper)

    for size in [4]:
        A = np.random.rand(size, size) * 10 + 1
        x_true = np.random.rand(size)
        b = A @ x_true

        augmented = np.column_stack((A, b))

        # Gauss + Strassen 
        Ab_strassen = gauss_strassen_wrapper.gaussElimination(augmented.copy())
        U_strassen = Ab_strassen[:, :-1]
        b_strassen = Ab_strassen[:, -1]

        print(gauss_strassen_wrapper.flops)

        x_strassen = np.linalg.solve(U_strassen, b_strassen)
        assert np.allclose(x_strassen, x_true, atol=1e-5), \
            f"Gauss(Strassen) źle dla size={size}"

        assert np.allclose(
            U_strassen[np.tril_indices(size, -1)], 0, atol=1e-5
        ), f"U(Strassen) nie jest górnotrójkątna dla size={size}"

        # Gauss + Binet
        Ab_binet = gauss_binet_wrapper.gaussElimination(augmented.copy())
        U_binet = Ab_binet[:, :-1]
        b_binet = Ab_binet[:, -1]

        print(gauss_binet_wrapper.flops)

        x_binet = np.linalg.solve(U_binet, b_binet)
        assert np.allclose(x_binet, x_true, atol=1e-5), \
            f"Gauss(Binet) źle dla size={size}"

        assert np.allclose(
            U_binet[np.tril_indices(size, -1)], 0, atol=1e-5
        ), f"U(Binet) nie jest górnotrójkątna dla size={size}"

    print("GaussWrapper: wszystkie testy przeszły pomyślnie.")
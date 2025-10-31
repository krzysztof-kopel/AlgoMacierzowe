import numpy as np
import time
from tests import MatrixMultiplicaitonTester
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
        assert A.shape==(4,5) and B.shape==(5,5), "Incorrect sizes for this multiplication method, should be 4x5 and 5x5"
        H1  = A[2,1] * (-B[1,0] - B[1,4] - B[2,0])
        H2  = (A[1,1] + A[1,4] - A[2,4]) * (-B[1,4] - B[4,0])
        H3  = (-A[2,0] - A[3,0] + A[3,1]) * (-B[0,0] + B[1,4])
        H4  = (A[0,1] + A[0,3] + A[2,3]) * (-B[1,4] - B[3,0])
        H5  = (A[0,4] + A[1,1] + A[1,4]) * (-B[1,3] + B[4,0])
        H6  = (-A[1,1] - A[1,4] - A[3,4]) * (B[1,2] + B[4,0])
        H7  = (-A[0,0] + A[3,0] - A[3,1]) * (B[0,0] + B[1,3])
        H8  = (A[2,1] - A[2,2] - A[3,2]) * (-B[1,2] + B[2,0])
        H9  = (-A[0,1] - A[0,3] + A[3,3]) * (B[1,2] + B[3,0])
        H10 = (A[1,1] + A[1,4]) * B[4,0]
        H11 = (-A[1,0] - A[3,0] + A[3,1]) * (-B[0,0] + B[1,1])
        H12 = (A[3,0] - A[3,1]) * B[0,0]
        H13 = (A[0,1] + A[0,3] + A[1,3]) * (B[1,1] + B[3,0])
        H14 = (A[0,2] - A[2,1] + A[2,2]) * (B[1,3] + B[2,0])
        H15 = (-A[0,1] - A[0,3]) * B[3,0]
        H16 = (-A[2,1] + A[2,2]) * B[2,0]
        H17 = (A[0,1] + A[0,3] - A[1,0] + A[1,1] - A[1,2] + A[1,3] - A[2,1] + A[2,2] - A[3,0] + A[3,1]) * B[1,1]
        H18 = A[1,0] * (B[0,0] + B[0,1] + B[4,1])
        H19 = -A[1,2] * (B[2,0] + B[2,1] + B[4,1])
        H20 = (-A[0,4] + A[1,0] + A[1,2] - A[1,4]) * (-B[0,0] - B[0,1] + B[0,3] - B[4,1])
        H21 = (A[1,0] + A[1,2] - A[1,4]) * B[4,1] 
        H22 = (A[0,2] - A[0,3] - A[1,3]) * (B[0,0] + B[0,1] - B[0,3] - B[2,0] - B[2,1] + B[2,3] + B[3,3])
        H23 = A[0,2] * (-B[2,0] + B[2,3] + B[3,3])
        H24 = A[0,4] * (-B[3,3] - B[4,0] + B[4,3])
        H25 = -A[0,0] * (B[0,0] - B[0,3])
        H26 = (-A[0,2] + A[0,3] + A[0,4]) * B[3,3]
        H27 = (A[0,2] - A[2,0] + A[2,2]) * (B[0,0] - B[0,3] + B[0,4] + B[2,4])
        H28 = -A[2,3] * (-B[2,4] - B[3,0] - B[3,4])
        H29 = A[2,0] * (B[0,0] + B[0,4] + B[2,4])
        H30 = (A[2,0] - A[2,2] + A[2,3]) * B[2,4]
        H31 = (-A[0,3] - A[0,4] - A[2,3]) * (-B[3,3] - B[4,0] + B[4,3] - B[4,4])
        H32 = (A[1,0] + A[3,0] + A[3,3]) * (B[0,2] - B[3,0] - B[3,1] - B[3,2])
        H33 = A[3,2] * (-B[2,0] - B[2,2])
        H34 = A[3,3] * (-B[0,2] + B[3,0] + B[3,2])
        H35 = -A[3,4] * (B[0,2] + B[4,0] + B[4,2])
        H36 = (A[1,2] - A[1,4] - A[3,4]) * (B[2,0] + B[2,1] + B[2,2] + B[4,1])
        H37 = (-A[3,0] - A[3,3] + A[3,4]) * B[0,2]
        H38 = (-A[1,2] - A[2,0] + A[2,2] - A[2,3]) * (B[2,4] + B[3,0] + B[3,1] + B[3,4])
        H39 = (-A[2,0] - A[3,0] - A[3,3] + A[3,4]) * (B[0,2] + B[4,0] + B[4,2] + B[4,4])
        H40 = (-A[0,2] + A[0,3] + A[0,4] - A[3,3]) * (-B[2,0] - B[2,2] + B[2,3] + B[3,3])
        H41 = (-A[0,0] + A[3,0] - A[3,4]) * (B[0,2] + B[2,0] + B[2,2] - B[2,3] + B[4,0] + B[4,2] - B[4,3])
        H42 = (-A[1,0] + A[1,4] - A[2,4]) * (-B[0,0] - B[0,1] - B[0,4] + B[3,0] + B[3,1] + B[3,4] - B[4,1])
        H43 = A[1,3] * (B[3,0] + B[3,1])
        H44 = (A[1,2] + A[2,1] - A[2,2]) * (B[1,1] - B[2,0])
        H45 = (-A[2,2] + A[2,3] - A[3,2]) * (B[2,4] + B[3,0] + B[3,2] + B[3,4] + B[4,0] + B[4,2] + B[4,4])
        H46 = -A[2,4] * (-B[4,0] - B[4,4])
        H47 = (A[1,0] - A[1,4] - A[2,0] + A[2,4]) * (B[0,0] + B[0,1] + B[0,4] - B[3,0] - B[3,1] - B[3,4])
        H48 = (-A[1,2] + A[2,2]) * (B[1,1] + B[2,1] + B[2,3] + B[3,0] + B[3,1] + B[3,4])
        H49 = (-A[0,0] - A[0,2] + A[0,3] + A[0,4] - A[1,0] - A[1,2] + A[1,3] + A[1,4]) * (-B[0,0] - B[0,1] + B[0,3])
        H50 = (-A[0,3] - A[1,3]) * (B[1,1] - B[2,0] - B[2,1] + B[2,3] - B[3,1] + B[3,3])
        H51 = A[1,1] * (B[1,0] + B[1,1] - B[4,0])
        H52 = A[3,1] * (B[0,0] + B[1,0] + B[1,2])
        H53 = -A[0,1] * (-B[1,0] + B[1,3] + B[3,0])
        H54 = (A[0,1] + A[0,3] - A[1,1] - A[1,4] - A[2,1] + A[2,2] - A[3,1] + A[3,2] - A[3,3] - A[3,4]) * B[1,2]
        H55 = (A[0,3] - A[3,3]) * (-B[1,2] + B[2,0] + B[2,2] - B[2,3] + B[4,2] - B[3,3])
        H56 = (A[0,0] - A[0,4] - A[3,0] + A[3,4]) * (B[2,0] + B[2,2] - B[2,3] + B[4,0] + B[4,2] - B[4,3])
        H57 = (-A[2,0] - A[3,0]) * (-B[0,2] - B[0,4] - B[1,4] - B[4,0] - B[4,2] - B[4,4])
        H58 = (-A[0,3] - A[0,4] - A[2,3] - A[2,4]) * (-B[4,0] + B[4,3] - B[4,4])
        H59 = (-A[2,2] + A[2,3] - A[3,2] + A[3,3]) * (B[3,0] + B[3,2] + B[3,4] + B[4,0] + B[4,2] + B[4,4])
        H60 = (A[1,4] + A[3,4]) * (B[1,2] - B[2,0] - B[2,1] - B[2,2] - B[4,1] - B[4,2])
        H61 = (A[0,3] + A[2,3]) * (B[0,0] - B[0,3] + B[0,4] - B[1,4] - B[3,3] + B[3,4] -B[4,0] + B[4,3] - B[4,4])
        H62 = (A[1,0] + A[3,0]) * (B[0,1] + B[0,2] + B[1,1] - B[3,0] - B[3,1] - B[3,2])
        H63 = (-A[2,2] - A[3,2]) * (-B[1,2] - B[2,2] - B[2,4] - B[3,0] - B[3,2] - B[3,4])
        H64 = (A[0,0] - A[0,2] - A[0,3] + A[2,0] - A[2,2] - A[2,3]) * (B[0,0] - B[0,3] + B[0,4])
        H65 = (-A[0,0] + A[3,0]) * (-B[0,2] + B[0,3] + B[1,3] - B[4,0] + B[4,3])
        H66 = (A[0,0] - A[0,1] + A[0,2] - A[0,4] - A[1,1] - A[1,4] - A[2,1] + A[2,2] - A[3,0] + A[3,1]) * B[1,3]
        H67 = (A[1,4] - A[2,4]) * (B[0,0] + B[0,1] + B[0,4] - B[1,4] - B[3,0] - B[3,1] - B[3,4] + B[4,1] + B[4,4])
        H68 = (A[0,0] + A[0,2] - A[0,3] - A[0,4] - A[3,0] - A[3,2] + A[3,3] + A[3,4]) * (-B[2,0] - B[2,2] + B[2,3])
        H69 = (-A[0,2] + A[0,3] - A[1,2] + A[1,3]) * (-B[1,3] - B[2,0] - B[2,1] + B[2,3] - B[4,1] + B[4,3])
        H70 = (A[1,2] - A[1,4] + A[3,2] - A[3,4]) * (-B[2,0] - B[2,1] - B[2,2])
        H71 = (-A[2,0] + A[2,2] - A[2,3] + A[2,4] - A[3,0] + A[3,2] - A[3,3] + A[3,4]) * (-B[4,0] - B[4,2] - B[4,4])
        H72 = (-A[1,0] - A[1,3] - A[3,0] - A[3,3]) * (B[3,0] + B[3,1] + B[3,2]) 
        H73 = (A[0,2] - A[0,3] - A[0,4] + A[1,2] - A[1,3] - A[1,4]) * (B[0,0] + B[0,1] - B[0,3] + B[1,3] + B[4,1] - B[4,3])
        H74 = (A[1,0] - A[1,2] + A[1,3] - A[2,0] + A[2,2] - A[2,3]) * (B[3,0] + B[3,1] + B[3,4])
        H75 = -(A[0,1] + A[0,3] - A[1,1] - A[1,4] - A[2,0] + A[2,1] + A[2,3] + A[2,4] - A[3,0] + A[3,1]) * B[1,4]
        H76 = (A[0,2] + A[2,2]) * (-B[0,0] + B[0,3] - B[0,4] + B[1,3] + B[2,3] - B[2,4])
        C = np.zeros((4,5))
        C[0,0] = int(-H10 + H12 + H14 - H15 - H16 + H53 + H5 - H66 - H7)
        C[1,0] = int(H10 + H11 - H12 + H13 + H15 + H16 - H17 - H44 + H51)
        C[2,0] = int(H10 - H12 + H15 + H16 - H1 + H2 + H3 - H4 + H75)
        C[3,0] = int(-H10 + H12 - H15 - H16 + H52 + H54 - H6 - H8 + H9)
        C[0,1] = int(H13 + H15 + H20 + H21 - H22 + H23 + H25 - H43 + H49 + H50)
        C[1,1] = int(-H11 + H12 - H13 - H15 - H16 + H17 + H18 - H19 - H21 + H43 + H44)
        C[2,1] = int(-H16 - H19 - H21 - H28 - H29 - H38 + H42 + H44 - H47 + H48)
        C[3,1] = int(H11 - H12 - H18 + H21 - H32 + H33 - H34 - H36 + H62 - H70)
        C[0,2] = int(H15 + H23 + H24 + H34 - H37 + H40 - H41 + H55 - H56 - H9)
        C[1,2] = int(-H10 + H19 + H32 + H35 + H36 + H37 - H43 - H60 - H6 - H72)
        C[2,2] = int(-H16 - H28 + H33 + H37 - H39 + H45 - H46 + H63 - H71 - H8)
        C[3,2] = int(-H10 + H15 + H16 - H33 + H34 - H35 - H37 - H54 + H6 + H8 - H9)
        C[0,3] = int(-H10 + H12 + H14 - H16 + H23 + H24 + H25 + H26 + H5 - H66 - H7)
        C[1,3] = int(-H10 + H18 - H19 + H20 - H22 - H24 - H26 - H5 - H69 + H73)
        C[2,3] = int(-H14 + H16 - H23 - H26 + H27 + H29 + H31 + H46 - H58 + H76)
        C[3,3] = int(H12 + H25 + H26 - H33 - H35 - H40 + H41 + H65 - H68 - H7)
        C[0,4] = int(H15 + H24 + H25 + H27 - H28 + H30 + H31 - H4 + H61 + H64)
        C[1,4] = int(-H10 - H18 - H2 - H30 - H38 + H42 - H43 + H46 + H67 + H74)
        C[2,4] = int(-H10 + H12 - H15 + H28 + H29 - H2 - H30 - H3 + H46 + H4 - H75)
        C[3,4] = int(-H12 - H29 + H30 - H34 + H35 + H39 + H3 - H45 + H57 + H59)
        return C

    def _ai_matrix_multiply(self, A, B):
        """
        Rozszerzenie funkcji ai_matrix_multiply_strict,
        wykorzytsujące ją do mnożenia między sobą dwóch macierzy
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
            if A.shape==(4,5) and B.shape==(5,5):
                return self.ai_matrix_multiply_strict(A, B)
            else:
                n, m = A.shape
                k4 = n // 4
                k5 = m // 5
                A_blocks = [[A[i*k4:(i+1)*k4, j*k5:(j+1)*k5] for j in range(5)] for i in range(4)]
                B_blocks = [[B[i*k5:(i+1)*k5, j*k5:(j+1)*k5] for j in range(5)] for i in range(5)]
                C_blocks = [[np.zeros((4,5)) for _ in range(5)] for _ in range(4)]
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
        liczące czas trwaniam flops, oraz zużytą pamięć.
        """
        start_time = time.time()
        self.memory_used += A.nbytes + B.nbytes
        self.flops += 76 * 10 + 4 * 5 * 9
        C = self._ai_matrix_multiply(A, B)
        self.memory_used += C.nbytes
        end_time = time.time()
        return C

if __name__ == "__main__":
    ai_wrapper = AIWrapper()

    # jakieś testy
    np.set_printoptions(precision=3, suppress=True)
    rng = np.random.default_rng(42)
    A = rng.integers(-5, 6, size=(4, 5), dtype=int)
    B = rng.integers(-5, 6, size=(5, 5), dtype=int)
    print(A)
    print(B)
    print(A@B)
    print(ai_wrapper.ai_matrix_multiply(A, B))

    
    print(f"FLOPS: {ai_wrapper.flops}")
    print(f"Memory used: {ai_wrapper.memory_used / 1024:.2f} KB")
    if ai_wrapper.time_used:
        print(f"Median time: {np.median(ai_wrapper.time_used):.6f} s")

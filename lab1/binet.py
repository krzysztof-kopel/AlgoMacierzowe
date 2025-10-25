import numpy as np
from lab1.tests import MatrixMultiplicaitonTester


# To nie jest oczywiście implementacja Bineta, tylko przykład, jak możemy to pisać, żeby później móc łatwo testować.
def simple_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Tylko przykład do testów."""
    return np.dot(a, b)

if __name__ == "__main__":
    tester = MatrixMultiplicaitonTester(simple_multiply)
    tester.run_all_tests()

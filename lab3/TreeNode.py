import numpy as np
import scipy.sparse.linalg as spl


class SVDComponents:
    def __init__(self, singular_values: np.ndarray, U: np.ndarray, V: np.ndarray):
        self.singular_values = singular_values
        self.U = U
        self.V = V


class SVD:
    @staticmethod
    def svd_partial_decomposition(matrix: np.ndarray, rank: int):
        svd_result = spl.svds(matrix, k=rank)
        return SVDComponents(svd_result[1], svd_result[0], svd_result[2])


class TreeNode:
    def __init__(self, rank: int, coordinates: tuple[int, int, int, int],
                 matrix: np.ndarray):
        self.rank = rank
        self.coordinates = coordinates  # zmieniłem size na coordinates jak coś, imo bardziej sugerująca nazwa
        self.matrix = matrix
        self.svd: SVDComponents | None = None
        self.children: list['TreeNode'] = []
        self.is_leaf = True

    def append_child(self, child: 'TreeNode'):
        self.children.append(child)
        self.is_leaf = False

    def compress_matrix(self, epsilon: float | None = None, singular_values_number: int | None = None):
        matrix = self.matrix
        m, n = matrix.shape

        if np.count_nonzero(matrix) == 0:
            self.svd = SVDComponents(
                np.array([]),
                np.empty((m, 0)),
                np.empty((0, n))
            )
            return

        if min(m, n) <= 1 or self.rank >= min(m, n):
            return

        # dla danej macierzy robimy czesciowe svd, dla rank i epsilon
        svd_components = SVD.svd_partial_decomposition(matrix, self.rank)

        singular_values = svd_components.singular_values

        # jesli uzyskana wartość na rank jest niższa niż epsilon to jest okay,
        # używamy danego partial svd jako kompresji dla danej części
        if epsilon is not None:
            rank_to_use = self.rank
            if singular_values[-1] < epsilon:
                while rank_to_use > 0 and singular_values[rank_to_use - 1] < epsilon:
                    rank_to_use -= 1

                self.svd = SVDComponents(
                    singular_values[:rank_to_use],
                    svd_components.U[:, :rank_to_use],
                    svd_components.V[:rank_to_use, :]
                )
                return
        elif singular_values_number is not None:
            self.svd = SVDComponents(
                singular_values[:singular_values_number],
                svd_components.U[:, :singular_values_number],
                svd_components.V[:singular_values_number, :]
            )
        else:
            raise ValueError("Either epsilon or singular_values_number must be provided.")

        # jeśli nie to dzielimy daną macierz na 4 podmacierze i wrzucamy rekurencyjnie
        mid_row = m // 2
        mid_col = n // 2
        r0, c0, r1, c1 = self.coordinates

        blocks = [
            (matrix[:mid_row, :mid_col],
             (r0, c0, r0 + mid_row, c0 + mid_col)),

            (matrix[:mid_row, mid_col:],
             (r0, c0 + mid_col, r0 + mid_row, c1)),

            (matrix[mid_row:, :mid_col],
             (r0 + mid_row, c0, r1, c0 + mid_col)),

            (matrix[mid_row:, mid_col:],
             (r0 + mid_row, c0 + mid_col, r1, c1)),
        ]

        for submatrix, coords in blocks:
            child = TreeNode(self.rank, coords, submatrix)
            child.compress_matrix(epsilon)
            self.append_child(child)


# proste testy, jutro dorobie porządne

M = np.random.randn(1280, 1280)
rank = 8
eps = 1e-8

root = TreeNode(
    rank=rank,
    coordinates=(0, 0, M.shape[0], M.shape[1]),
    matrix=M
)

root.compress_matrix(epsilon=eps)


# na razie taka zvibecodowana funkcja do testowania

def reconstruct_matrix(node: TreeNode, shape):
    result = np.zeros(shape)

    def fill(node: TreeNode):
        r0, c0, r1, c1 = node.coordinates

        if node.is_leaf:
            if node.svd is not None and node.svd.singular_values.size > 0:
                U = node.svd.U
                S = np.diag(node.svd.singular_values)
                V = node.svd.V
                result[r0:r1, c0:c1] = U @ S @ V
            else:
                result[r0:r1, c0:c1] = node.matrix
        else:
            for child in node.children:
                fill(child)

    fill(node)
    return result


M_hat = reconstruct_matrix(root, M.shape)

rel_error = np.linalg.norm(M - M_hat) / np.linalg.norm(M)
print("Relative reconstruction error:", rel_error)

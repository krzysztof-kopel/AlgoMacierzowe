import numpy as np
import scipy.sparse.linalg as spl
import matplotlib.pyplot as plt

from skimage import data

class SVDComponents:
    def __init__(self, singular_values: np.ndarray, U: np.ndarray, V: np.ndarray):
        self.singular_values = singular_values
        self.U = U
        self.V = V


class SVD:
    @staticmethod
    def svd_full_decomposition(matrix: np.ndarray):
        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
        return SVDComponents(s, U, Vt)

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

    def compress_matrix(self, epsilon: float = 1e-10):
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
        rank_to_use = self.rank

        # jesli uzyskana wartość na rank jest niższa niż epsilon to jest okay,
        # używamy danego partial svd jako kompresji dla danej części
        if singular_values[-1] < epsilon:
            while rank_to_use > 0 and singular_values[rank_to_use - 1] < epsilon:
                rank_to_use -= 1

            self.svd = SVDComponents(
                singular_values[:rank_to_use],
                svd_components.U[:, :rank_to_use],
                svd_components.V[:rank_to_use, :]
            )
            return

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

    def create_structure_image(self, shape: tuple[int, int] = None):
        if shape is None:
            shape = self.matrix.shape

        img = np.zeros(shape)

        def draw(node):
            if not node.children:
                r0, c0, r1, c1 = node.coordinates

                img[r0:r1, c0:c1] = 1.0

                # tutaj są tylko góna i lewa krawędź bo operujemy na pojedynczych pixelach
                img[r0:r0+1, c0:c1] = 0.0
                img[r0:r1, c0:c0+1] = 0.0
            else:
                for ch in node.children:
                    draw(ch)

        draw(self)
        return img



M = data.astronaut()[:, :, 2]
rank = 4
eps = 1e-2

root = TreeNode(
    rank=rank,
    coordinates=(0, 0, M.shape[0], M.shape[1]),
    matrix=M
)

root.compress_matrix(epsilon=eps)

img = root.create_structure_image()
plt.imshow(img, cmap="gray", interpolation="nearest")
plt.savefig("structure.png", dpi=800)
plt.close()




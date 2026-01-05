import numpy as np

class SVDComponents:
    def __init__(self, singular_values: np.ndarray, U: np.ndarray, V: np.ndarray):
        self.singular_values = singular_values
        self.U = U
        self.V = V

class TreeNode:
    def __init__(self, rank: int, size: tuple[int, int, int, int],
                 matrix: np.ndarray):
        # Przez size rozumiem tu bardziej "które to są współrzędne w oryginalnej macierzy",
        # w sumie coś podobnego do tego, co było na wykłdzie.
        self.rank = rank
        self.size = size
        self.matrix = matrix
        self.svd: SVDComponents | None = None
        self.children: list['TreeNode'] = []
        self.is_leaf = True

    def append_child(self, child: 'TreeNode'):
        self.children.append(child)
        self.is_leaf = False

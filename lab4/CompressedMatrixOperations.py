from platform import node
import numpy as np
from lab3.TreeNode import TreeNode
from lab3.TreeNode import SVDComponents

class CompressedMatrixOperations:
    @staticmethod
    def split(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        horizontal_split_point = matrix.shape[1] // 2
        vertical_split_point = matrix.shape[0] // 2

        return (matrix[:vertical_split_point, :horizontal_split_point],
                matrix[:vertical_split_point, horizontal_split_point:],
                matrix[vertical_split_point:, :horizontal_split_point],
                matrix[vertical_split_point:, horizontal_split_point:])

    @staticmethod
    def matrix_vector_mult(treeNode: TreeNode, vector: np.ndarray) -> np.ndarray:
        if not treeNode.children:
            if treeNode.rank == 0:
                return np.zeros(treeNode.matrix.shape[0], dtype=vector.dtype)
            return treeNode.svd.U @ (treeNode.svd.V @ vector)

        rows = treeNode.matrix.shape[0]
        X1 = vector[:rows // 2]
        X2 = vector[rows // 2:]

        Y1 = CompressedMatrixOperations.matrix_vector_mult(treeNode.children[0], X1)
        Y2 = CompressedMatrixOperations.matrix_vector_mult(treeNode.children[1], X2)
        Y3 = CompressedMatrixOperations.matrix_vector_mult(treeNode.children[2], X1)
        Y4 = CompressedMatrixOperations.matrix_vector_mult(treeNode.children[3], X2)

        return np.vstack((Y1 + Y2, Y3 + Y4))

    @staticmethod
    def rSVDofCompressed(U: np.ndarray, V: np.ndarray, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        M = U @ V.T
        U_full, S, Vt = np.linalg.svd(M, full_matrices=False)

        r = np.sum(S > eps)
 
        return U_full[:, :r], S[:r], Vt[:r, :]

    @staticmethod
    def matrix_matrix_add(treeNodeA: TreeNode, treeNodeB: TreeNode) -> TreeNode:
        def low_rank_leaf(U, V, coords):
            res_node = TreeNode(
                rank=U.shape[1],
                coordinates=coords,
                matrix=None
            )
            res_node.svd = SVDComponents(
                singular_values=np.ones(U.shape[1]),
                U=U,
                V=V.T
            )
            return res_node

        if not treeNodeA.children and not treeNodeB.children and treeNodeA.rank == 0 and treeNodeB.rank == 0:
            res_node = TreeNode(
                rank=0,
                coordinates=treeNodeA.coordinates,
                matrix=None
            )
            res_node.svd = SVDComponents(
                np.array([]),
                np.empty((treeNodeA.matrix.shape[0], 0)),
                np.empty((0, treeNodeA.matrix.shape[1]))
            )
            return res_node
        if not treeNodeA.children and not treeNodeB.children and treeNodeA.rank != 0 and treeNodeB.rank != 0:
            singularvalues_sqrt_A = np.sqrt(treeNodeA.svd.singular_values)
            singularvalues_sqrt_B = np.sqrt(treeNodeB.svd.singular_values)
            
            Ua = treeNodeA.svd.U * singularvalues_sqrt_A
            Va = treeNodeA.svd.V * singularvalues_sqrt_A
            Ub = treeNodeB.svd.U * singularvalues_sqrt_B
            Vb = treeNodeB.svd.V * singularvalues_sqrt_B

            U = np.hstack([Ua, Ub])
            V = np.hstack([Va, Vb])

            U_new, S_new, V_new = CompressedMatrixOperations.rSVDofCompressed(U, V)

            res_node = TreeNode(
                rank=len(S_new),
                coordinates=treeNodeA.coordinates,
                matrix=None
            )
            res_node.svd = SVDComponents(S_new, U_new, V_new)
            return res_node

        if treeNodeA.children and treeNodeB.children:
            A1, A2, A3, A4 = treeNodeA.children
            B1, B2, B3, B4 = treeNodeB.children

            C1 = CompressedMatrixOperations.matrix_matrix_add(A1, B1)
            C2 = CompressedMatrixOperations.matrix_matrix_add(A2, B2)
            C3 = CompressedMatrixOperations.matrix_matrix_add(A3, B3)
            C4 = CompressedMatrixOperations.matrix_matrix_add(A4, B4)

            res_node = TreeNode(
                rank=0,
                coordinates=treeNodeA.coordinates,
                matrix=None
            )
            res_node.append_child(C1)
            res_node.append_child(C2)
            res_node.append_child(C3)
            res_node.append_child(C4)

            return res_node
        
        if not treeNodeA.children and treeNodeB.children:
            singularvalues_sqrt = np.sqrt(treeNodeA.svd.singular_values)
            U1 = treeNodeA.svd.U * singularvalues_sqrt
            V1 = treeNodeA.svd.V * singularvalues_sqrt

            rows = U1.shape[0]
            U11, U12 = U1[:(rows//2), :], U1[(rows//2):, :]

            cols = V1.shape[1]
            V11, V12 = V1[:, :cols//2], V1[:, cols//2:]

            B1, B2, B3, B4 = treeNodeB.children

            A1 = low_rank_leaf(U11, V11, B1.coordinates)
            A2 = low_rank_leaf(U11, V12, B2.coordinates)
            A3 = low_rank_leaf(U12, V11, B3.coordinates)
            A4 = low_rank_leaf(U12, V12, B4.coordinates)

            C1 = CompressedMatrixOperations.matrix_matrix_add(A1, B1)
            C2 = CompressedMatrixOperations.matrix_matrix_add(A2, B2)
            C3 = CompressedMatrixOperations.matrix_matrix_add(A3, B3)
            C4 = CompressedMatrixOperations.matrix_matrix_add(A4, B4)

            res_node = TreeNode(rank=0, coordinates=treeNodeB.coordinates, matrix=None)
            res_node.append_child(C1)
            res_node.append_child(C2)
            res_node.append_child(C3)
            res_node.append_child(C4)

            return res_node

        if treeNodeA.children and not treeNodeB.children:
            A1, A2, A3, A4 = treeNodeA.children
            singulrvalues_sqrt = np.sqrt(treeNodeB.svd.singular_values)
            U1 = treeNodeB.svd.U * singulrvalues_sqrt
            V1 = treeNodeB.svd.V * singulrvalues_sqrt

            rows = U1.shape[0]
            U11, U12 = U1[:(rows//2), :], U1[(rows//2):, :]

            rows = V1.shape[0]
            V11, V12 = V1[:rows//2, :], V1[rows//2:, :]

            B1 = low_rank_leaf(U11, V11, A1.coordinates)
            B2 = low_rank_leaf(U11, V12, A2.coordinates)
            B3 = low_rank_leaf(U12, V11, A3.coordinates)
            B4 = low_rank_leaf(U12, V12, A4.coordinates)

            C1 = CompressedMatrixOperations.matrix_matrix_add(A1, B1)
            C2 = CompressedMatrixOperations.matrix_matrix_add(A2, B2)
            C3 = CompressedMatrixOperations.matrix_matrix_add(A3, B3)
            C4 = CompressedMatrixOperations.matrix_matrix_add(A4, B4)

            res_node = TreeNode(rank=0, coordinates=treeNodeA.coordinates, matrix=None)
            res_node.append_child(C1)
            res_node.append_child(C2)
            res_node.append_child(C3)
            res_node.append_child(C4)   

            return res_node

        if not treeNodeA.children and not treeNodeB.children and treeNodeA.matrix is not None and treeNodeB.matrix is not None and treeNodeA.matrix.shape == (1, 1) and treeNodeB.matrix.shape == (1, 1):
            val = treeNodeA.matrix[0, 0] + treeNodeB.matrix[0, 0]

            res_node = TreeNode(
                rank=0,
                coordinates=treeNodeA.coordinates,
                matrix=np.array([[val]])
            )
            res_node.svd = SVDComponents(
                np.array([]),
                np.empty((1, 0)),
                np.empty((0, 1))
            )

            return res_node

    @staticmethod
    def matrix_matrix_mult(treeNodeA: TreeNode, treeNodeB: TreeNode) -> TreeNode:
        def low_rank_leaf(U, V, coords):
            node = TreeNode(
                rank=U.shape[1],
                coordinates=coords,
                matrix=None
            )
            node.svd = SVDComponents(
                singular_values=np.ones(U.shape[1]),
                U=U,
                V=V
            )
            return node

        if not treeNodeA.children and not treeNodeB.children and treeNodeA.rank == 0 and treeNodeB.rank == 0:
            res_node = TreeNode(
                rank=0,
                coordinates=treeNodeA.coordinates,
                matrix=None
            )
            res_node.svd = SVDComponents(
                singular_values=np.array([]),
                U=np.empty((treeNodeA.svd.U.shape[0], 0)),
                V=np.empty((0, treeNodeB.svd.V.shape[1]))
            )
            return res_node
        
        if not treeNodeA.children and not treeNodeB.children and treeNodeA.rank != 0 and treeNodeB.rank != 0:
            M = treeNodeA.svd.V @ treeNodeB.svd.U 

            U_new = treeNodeA.svd.U @ M             
            V_new = treeNodeB.svd.V              

            res = TreeNode(
                rank=U_new.shape[1],
                coordinates=treeNodeA.coordinates,
                matrix=None
            )
            res.svd = SVDComponents(
                singular_values=np.ones(U_new.shape[1]),
                U=U_new,
                V=V_new
            )
            return res
        
        if treeNodeA.children and treeNodeB.children:
            A1, A2, A3, A4 = treeNodeA.children
            B1, B2, B3, B4 = treeNodeB.children

            C1 = CompressedMatrixOperations.matrix_matrix_add(CompressedMatrixOperations.matrix_matrix_mult(A1, B1), CompressedMatrixOperations.matrix_matrix_mult(A2, B3))
            C2 = CompressedMatrixOperations.matrix_matrix_add(CompressedMatrixOperations.matrix_matrix_mult(A1, B2), CompressedMatrixOperations.matrix_matrix_mult(A2, B4))
            C3 = CompressedMatrixOperations.matrix_matrix_add(CompressedMatrixOperations.matrix_matrix_mult(A3, B1), CompressedMatrixOperations.matrix_matrix_mult(A4, B3))
            C4 = CompressedMatrixOperations.matrix_matrix_add(CompressedMatrixOperations.matrix_matrix_mult(A3, B2), CompressedMatrixOperations.matrix_matrix_mult(A4, B4))
            
            res_node = TreeNode(
                rank=0,
                coordinates=treeNodeA.coordinates,
                matrix=None
            )   
            res_node.append_child(C1)
            res_node.append_child(C2)
            res_node.append_child(C3)   
            res_node.append_child(C4)

            return res_node
        
        if not treeNodeA.children and treeNodeB.children:
            B1, B2, B3, B4 = treeNodeB.children

            singularvalues_sqrt = np.sqrt(treeNodeA.svd.singular_values)
            U1 = treeNodeA.svd.U * singularvalues_sqrt
            V1 = treeNodeA.svd.V * singularvalues_sqrt

            rows = U1.shape[0]
            U11 = U1[:(rows//2), :]
            U12 = U1[(rows//2):, :]

            cols = V1.shape[1]
            V11 = V1[:, :(cols//2)]
            V12 = V1[:, (cols//2):]

            A11 = low_rank_leaf(U11, V11, B1.coordinates)
            A12 = low_rank_leaf(U11, V12, B2.coordinates)
            A13 = low_rank_leaf(U12, V11, B3.coordinates)
            A14 = low_rank_leaf(U12, V12, B4.coordinates)

            C1 = CompressedMatrixOperations.matrix_matrix_add(CompressedMatrixOperations.matrix_matrix_mult(A11, B1), CompressedMatrixOperations.matrix_matrix_mult(A12, B3))
            C2 = CompressedMatrixOperations.matrix_matrix_add(CompressedMatrixOperations.matrix_matrix_mult(A11, B2), CompressedMatrixOperations.matrix_matrix_mult(A12, B4))
            C3 = CompressedMatrixOperations.matrix_matrix_add(CompressedMatrixOperations.matrix_matrix_mult(A13, B1), CompressedMatrixOperations.matrix_matrix_mult(A14, B3))
            C4 = CompressedMatrixOperations.matrix_matrix_add(CompressedMatrixOperations.matrix_matrix_mult(A13, B2), CompressedMatrixOperations.matrix_matrix_mult(A14, B4))

            res_node = TreeNode(
                rank=0,
                coordinates=treeNodeB.coordinates,
                matrix=None
            )
            res_node.append_child(C1)   
            res_node.append_child(C2)
            res_node.append_child(C3)   
            res_node.append_child(C4)

            return res_node
        
        if treeNodeA.children and not treeNodeB.children:
            A1, A2, A3, A4 = treeNodeA.children

            U1 = treeNodeB.svd.U
            V1 = treeNodeB.svd.V

            rows = U1.shape[0]
            U11 = U1[:(rows//2), :]
            U12 = U1[(rows//2):, :]

            cols = V1.shape[1]
            V11 = V1[:, :(cols//2)]
            V12 = V1[:, (cols//2):]

            B11 = low_rank_leaf(U11, V11, A1.coordinates)
            B12 = low_rank_leaf(U11, V12, A2.coordinates)
            B13 = low_rank_leaf(U12, V11, A3.coordinates)
            B14 = low_rank_leaf(U12, V12, A4.coordinates)

            C1 = CompressedMatrixOperations.matrix_matrix_add(CompressedMatrixOperations.matrix_matrix_mult(A1, B11), CompressedMatrixOperations.matrix_matrix_mult(A3, B12))
            C2 = CompressedMatrixOperations.matrix_matrix_add(CompressedMatrixOperations.matrix_matrix_mult(A2, B11), CompressedMatrixOperations.matrix_matrix_mult(A4, B12))
            C3 = CompressedMatrixOperations.matrix_matrix_add(CompressedMatrixOperations.matrix_matrix_mult(A1, B13), CompressedMatrixOperations.matrix_matrix_mult(A3, B14))
            C4 = CompressedMatrixOperations.matrix_matrix_add(CompressedMatrixOperations.matrix_matrix_mult(A2, B13), CompressedMatrixOperations.matrix_matrix_mult(A4, B14))

            res_node = TreeNode(
                rank=0,
                coordinates=treeNodeA.coordinates,
                matrix=None
            )
            res_node.append_child(C1)
            res_node.append_child(C2)
            res_node.append_child(C3)
            res_node.append_child(C4)

            return res_node

        if (not treeNodeA.children and not treeNodeB.children 
            and treeNodeA.matrix is not None 
            and treeNodeB.matrix is not None 
            and treeNodeA.matrix.shape == (1, 1) 
            and treeNodeB.matrix.shape == (1, 1)):
            val = treeNodeA.matrix[0, 0] * treeNodeB.matrix[0, 0]

            res = TreeNode(
                rank=0,
                coordinates=treeNodeA.coordinates,
                matrix=np.array([[val]])
            )
            res.svd = SVDComponents(
                singular_values=np.array([]),
                U=np.empty((1, 0)),
                V=np.empty((0, 1))
            )
            return res
        

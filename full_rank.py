import numpy as np
from scipy.linalg import lu

def my_full_rank_bysvd(A):
    """
    对矩阵 A 进行满秩分解 A = FG
    :param A: 原矩阵
    :return: 满秩分解的矩阵 F 和 G
    """
    # 矩阵秩
    rank = np.linalg.matrix_rank(A)
    
    # 奇异值分解 (SVD)
    U, Sigma, Vh = np.linalg.svd(A, full_matrices=False)
    
    # 构造 F 和 G
    F = U[:, :rank] @ np.diag(Sigma[:rank])  # 取前 rank 个奇异值及对应向量
    G = Vh[:rank, :]                         # 取前 rank 个右奇异向量
    
    return F, G

def my_full_rank_byplu(A):
    # 使用 SciPy 的 LU 分解
    P, L, U = lu(A)  # P: 置换矩阵, L: 下三角矩阵, U: 上三角矩阵

    # 确定矩阵的秩 r
    r = np.linalg.matrix_rank(A)

    # 提取 L 和 U 中的秩相关部分
    B = L[:, :r]  # L 的前 r 列
    C = U[:r, :]  # U 的前 r 行

    return P, B, C

# 定义矩阵 A
A = np.array([
    [-1,  0,  1,  2],
    [ 1,  2, -1,  1],
    [ 2,  2, -2, -1]
])

print("原矩阵 A:")
print(A)

# 满秩分解
P, B, C = my_full_rank_byplu(A)

# 验证分解结果
print("\n-------通过PLU分解构造满秩分解---------")
print("\n通过范数验证 A = P * B * C:")
print(np.linalg.norm(A - np.dot(P, np.dot(B, C)), 1))  # 验证是否还原

F, G = my_full_rank_bysvd(A)
print("\n-------通过SVD分解构造满秩分解---------")
print("\n分解得到的矩阵 F:")
print(F)
print("\n分解得到的矩阵 G:")
print(G)
print("\n通过范数验证 A = F * G:")
print(np.linalg.norm(A - np.dot(F, G), 1))  # 验证是否还原
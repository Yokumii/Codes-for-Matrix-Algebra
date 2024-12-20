import numpy as np
from scipy.linalg import lu

def classical_gram_schmidt(A):
    """
    经典 Gram-Schmidt 正交化算法
    输入:
        A: 输入矩阵 (m x n)，m >= n
    输出:
        Q: 正交矩阵 (m x n)
        R: 上三角矩阵 (n x n)
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for k in range(n):
        Q[:, k] = A[:, k]
        for j in range(k):
            R[j, k] = np.dot(Q[:, j], A[:, k])
            Q[:, k] = Q[:, k] - R[j, k] * Q[:, j]
        R[k, k] = np.linalg.norm(Q[:, k])
        Q[:, k] = Q[:, k] / R[k, k]
    
    return Q, R

def modified_gram_schmidt(A):
    """
    修正 Gram-Schmidt 正交化算法
    输入:
        A: 输入矩阵 (m x n)，m >= n
    输出:
        Q: 正交矩阵 (m x n)
        R: 上三角矩阵 (n x n)
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    A_copy = A.copy()

    for k in range(n):
        R[k, k] = np.linalg.norm(A_copy[:, k])
        Q[:, k] = A_copy[:, k] / R[k, k]
        for j in range(k + 1, n):
            R[k, j] = np.dot(Q[:, k], A_copy[:, j])
            A_copy[:, j] = A_copy[:, j] - R[k, j] * Q[:, k]
    
    return Q, R

# 测试函数
def test():
    # 测试用病态矩阵
    np.random.seed(0)
    A = np.array([[1.0, 1.0, 1.0],
                  [1.0, 1.0 + 1e-10, 1.0 + 2e-10],
                  [1.0, 1.0 + 2e-10, 1.0 + 3e-10]])
    print("测试病态矩阵 A:\n", A)

    # 经典 Gram-Schmidt
    Q_cgs, R_cgs = classical_gram_schmidt(A)
    # print("\n经典 Gram-Schmidt 结果:")
    # print("Q_cgs:\n", Q_cgs)
    # print("R_cgs:\n", R_cgs)
    # print("Q_cgs.T @ Q_cgs (正交性检查):\n", np.round(Q_cgs.T @ Q_cgs, 8))

    print("\n----数值稳定性比较：Q.T @ Q - I的范数------")
    delta_cgs = Q_cgs.T @ Q_cgs - np.eye(A.shape[1])
    norm_cgs = np.linalg.norm(delta_cgs, ord='fro')
    print("\n经典 Gram-Schmidt 算法的正交性范数:", norm_cgs)

    # 修正 Gram-Schmidt
    Q_mgs, R_mgs = modified_gram_schmidt(A)
    # print("\n修正 Gram-Schmidt 结果:")
    # print("Q_mgs:\n", Q_mgs)
    # print("R_mgs:\n", R_mgs)
    # print("Q_mgs.T @ Q_mgs (正交性检查):\n", np.round(Q_mgs.T @ Q_mgs, 8))
    delta_mgs = Q_mgs.T @ Q_mgs - np.eye(A.shape[1])
    norm_mgs = np.linalg.norm(delta_mgs, ord='fro')
    print("\n修正 Gram-Schmidt 算法的正交性范数:", norm_mgs)

    # NumPy 库函数 QR 分解
    Q_lib, R_lib = np.linalg.qr(A)
    # print("\nNumPy QR 分解结果:")
    # print("Q_lib:\n", Q_lib)
    # print("R_lib:\n", R_lib)
    # print("Q_lib.T @ Q_lib (正交性检查):\n", np.round(Q_lib.T @ Q_lib, 8))
    delta_lib = Q_lib.T @ Q_lib - np.eye(A.shape[1])
    norm_lib = np.linalg.norm(delta_lib, ord='fro')
    print("\nNumPy QR 分解的正交性范数:", norm_lib)

    # LU分解判断 R 是否为上三角矩阵
    P, L, U_cgs = lu(R_cgs)
    L_norm_cgs = np.linalg.norm(L - np.eye(L.shape[0]), ord='fro')
    U_norm_cgs = np.linalg.norm(U_cgs - R_cgs, ord='fro')

    P, L, U_mgs = lu(R_mgs)
    L_norm_mgs = np.linalg.norm(L - np.eye(L.shape[0]), ord='fro')
    U_norm_mgs = np.linalg.norm(U_mgs - R_mgs, ord='fro')

    P, L, U_lib = lu(R_lib)
    L_norm_lib = np.linalg.norm(L - np.eye(L.shape[0]), ord='fro')
    U_norm_lib = np.linalg.norm(U_lib - R_lib, ord='fro')

    print("\n--------R 上三角矩阵性质检查----------")
    print("\n经典 Gram-Schmidt算法:")
    print("\nL - I 的范数:", L_norm_cgs)
    print("\nU - R 的范数:", U_norm_cgs)
    print("\n修正 Gram-Schmidt算法:")
    print("\nL - I 的范数:", L_norm_mgs)
    print("\nU - R 的范数:", U_norm_mgs)
    print("\nNumPy QR 分解:")
    print("\nL - I 的范数:", L_norm_lib)
    print("\nU - R 的范数:", U_norm_lib)

    # 比较三种方法的误差
    print("\n---------误差比较-----------")
    print("经典 Gram-Schmidt vs 原矩阵 A (||A - Q @ R||_F):", np.linalg.norm(A - Q_cgs @ R_cgs, ord='fro'))
    print("修正 Gram-Schmidt vs 原矩阵 A (||A - Q @ R||_F):", np.linalg.norm(A - Q_mgs @ R_mgs, ord='fro'))
    print("NumPy QR 分解 vs 原矩阵 A (||A - Q @ R||_F):", np.linalg.norm(A - Q_lib @ R_lib, ord='fro'))

if __name__ == "__main__":
    test()
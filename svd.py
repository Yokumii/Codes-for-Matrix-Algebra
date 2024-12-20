import numpy as np
from scipy import linalg as la

def my_svd(A):
    """
    实现 SVD 分解（Gram-Schmidt法进行基扩充）
    参数:
        A: 输入矩阵，形状为 (m, n)
    返回:
        U: 左奇异矩阵，形状为 (m, m)
        Sigma: 奇异值对角矩阵，形状为 (m, n)
        V: 右奇异矩阵，形状为 (n, n)
    """
    m, n = A.shape
    
    # Step 1: 求 A^T A 的特征值和特征向量
    AtA = A.T @ A  # A^T * A
    eigvals, V = np.linalg.eigh(AtA)  # 求特征值和特征向量（对称矩阵用 eigh）

    # Step 2: 按特征值从大到小排序
    sorted_indices = np.argsort(eigvals)[::-1]  # 特征值从大到小排序
    eigvals = eigvals[sorted_indices]
    V = V[:, sorted_indices]

    # Step 3: 构造奇异值矩阵 Σ
    r = np.sum(eigvals > 1e-10)  # 非零特征值的个数（秩）
    singular_values = np.sqrt(eigvals[:r])  # 非零奇异值
    Sigma = np.zeros((m, n))
    np.fill_diagonal(Sigma, singular_values)  # 填充奇异值到对角线

    # Step 4: 计算左奇异向量 U
    U = np.zeros((m, m))
    for i in range(r):
        ui = A @ V[:, i] / singular_values[i]
        U[:, i] = ui

    # Step 5: 扩充 U 到标准正交基
    # 使用 Gram-Schmidt 方法对 U 的剩余列进行扩展
    for i in range(r, m):
        basis_vector = np.random.rand(m)  # 随机生成向量
        for j in range(i):
            basis_vector -= np.dot(U[:, j], basis_vector) * U[:, j]
        basis_vector /= np.linalg.norm(basis_vector)
        U[:, i] = basis_vector

    return U, Sigma, V

def my_svd_2(A):
    """
    实现 SVD 分解 A = U Σ V^H（基于列主元满秩分解和QR分解补全酉矩阵）。
    参数:
        A: 输入矩阵，形状为 (m, n)
    返回:
        U: 左奇异矩阵，形状为 (m, m)
        Σ: 奇异值对角矩阵，形状为 (m, n)
        V: 右奇异矩阵，形状为 (n, n)
        SingularValues: 奇异值的数组，长度为 min(m, n)
    """
    def extend_to_unitary(A1, m):
        """
        将矩阵 A1 扩展为酉矩阵 A。
        参数:
            A1: 输入矩阵，形状为 (m, n)
            m: 扩展后矩阵的行数和列数（酉矩阵是方阵）
        返回:
            A: 扩展后的酉矩阵，形状为 (m, m)
        """
        n = A1.shape[1]
        A0 = np.zeros((m, m), dtype=np.complex128)
        A0[:, :n] = A1[:, :]  # 保留 A1 的列
        P, L, U = la.lu(A0)  # LU 分解

        # 处理上三角矩阵 U
        B = np.eye(m)
        B[:, :n] = 0
        U = U + B
        AA = np.dot(P, np.dot(L, U))  # 重构矩阵

        Q, R = la.qr(AA)  # QR 分解
        A = Q
        A[:, :n] = A1[:, :]  # 恢复 A1
        return A

    # Step 1: 构造 A^H A，并计算其特征值和特征向量
    B = np.dot(A.conj().T, A)  # A^H * A
    Lambda, V0 = np.linalg.eig(B)  # 特征值和特征向量

    # 确保特征值非负，避免计算误差导致负值
    Lambda = np.abs(Lambda)

    # Step 2: 对特征向量进行正交化
    V, _ = la.qr(V0)  # V 为正交矩阵

    # Step 3: 计算奇异值
    rank = np.linalg.matrix_rank(A)  # 矩阵的秩
    D = np.zeros_like(A, dtype=np.complex128)  # 初始化奇异值矩阵
    sigma = np.zeros((rank, rank), dtype=np.complex128)  # 非零奇异值矩阵
    sigma_inv = np.zeros((rank, rank), dtype=np.complex128)  # 逆奇异值矩阵
    SingularValues = []

    for i in range(rank):
        D[i, i] = np.sqrt(Lambda[i])  # 奇异值
        sigma[i, i] = D[i, i]
        sigma_inv[i, i] = 1 / sigma[i, i]  # 计算奇异值的倒数
        SingularValues.append(D[i, i])

    # Step 4: 计算左奇异矩阵的前 rank 列
    V1 = V[:, :rank]  # 提取前 rank 列的右奇异向量
    U1 = np.dot(np.dot(A, V1), sigma_inv)  # 计算前 rank 列的左奇异向量

    # Step 5: 补全 U，使其为酉矩阵
    m, n = A.shape
    U = extend_to_unitary(U1, m)

    # Step 6: 返回结果
    Σ = np.zeros_like(A, dtype=np.complex128)
    np.fill_diagonal(Σ, np.array(SingularValues))
    SingularValues = np.array(SingularValues)
    return U, Σ, V, SingularValues

# 测试代码
if __name__ == "__main__":
    # 例题4.14
    print("------------例题4.14-------------\n")
    # 测试矩阵
    A = np.array([[1, 0, 1],
                  [0, 1, 1],
                  [0, 0, 0]])

    U, Sigma, V = my_svd(A)
    U_2, Sigma_2, V_2, SingularValues = my_svd_2(A)

    # 使用库函数进行 SVD 分解
    U_lib, Sigma_lib, Vh_lib = la.svd(A)
    Sigma_lib_full = np.zeros_like(A, dtype=np.float64)
    np.fill_diagonal(Sigma_lib_full, Sigma_lib)

    print("Matrix A:")
    print(A)

    print("\n基于库函数求解的奇异值:")
    print(Sigma_lib)

    print("\n自定义函数求解的奇异值:")
    print(SingularValues)
    
    # 验证 SVD 分解是否正确
    print("\nScipy求解的范数验证结果:")
    print(np.linalg.norm(A - U_lib @ Sigma_lib @ Vh_lib.T))

    print("\n基于Gram-Schmidt补全酉矩阵求解的范数验证结果:")
    print(np.linalg.norm(A - U @ Sigma @ V.T))

    print("\n基于列主元满秩分解和QR分解补全酉矩阵求解的范数验证结果:")
    print(np.linalg.norm(A - np.dot(U, np.dot(Sigma_2, V.conj().T))))

    # 习题4.4.2
    print("\n------------习题4.4.2-------------\n")
    # 测试矩阵
    A = np.array([[1, 0],
                  [0, 1],
                  [1, 1]])

    U, Sigma, V = my_svd(A)
    U_2, Sigma_2, V_2, SingularValues = my_svd_2(A)

    # 使用库函数进行 SVD 分解
    U_lib, Sigma_lib, Vh_lib = la.svd(A)
    Sigma_lib_full = np.zeros_like(A, dtype=np.float64)
    np.fill_diagonal(Sigma_lib_full, Sigma_lib)

    print("Matrix A:")
    print(A)
    print("\n基于库函数(SciPy.linalg.svd)求解的奇异值:")
    print(Sigma_lib)
    print("\n自定义函数求解的奇异值:")
    print(SingularValues)
    
    # 验证 SVD 分解是否正确
    print("\nScipy求解的范数验证结果:")
    print(np.linalg.norm(A - U_lib @ Sigma_lib_full @ Vh_lib.T))

    print("\n基于Gram-Schmidt补全酉矩阵求解的范数验证结果:")
    print(np.linalg.norm(A - U @ Sigma @ V.T))

    print("\n基于列主元满秩分解和QR分解补全酉矩阵求解的范数验证结果:")
    print(np.linalg.norm(A - np.dot(U, np.dot(Sigma_2, V.conj().T))))
import numpy as np
from scipy.linalg import lu

def moore_penrose_pseudoinverse_bysvd(A):
    """
    使用奇异值分解（SVD）计算矩阵 A 的 Moore-Penrose 广义逆.
    
    参数:
        A (ndarray): 输入的矩阵 (m x n)
    
    返回:
        A_plus (ndarray): 矩阵 A 的广义逆 (n x m)
    """
    # 1. 奇异值分解 (SVD): A = U Σ V^H
    U, sigma, Vh = np.linalg.svd(A, full_matrices=True)
    
    # 2. 构造 Σ^+ (广义逆奇异值矩阵)
    # 只取非零奇异值的倒数，构造对角矩阵
    sigma_plus = np.zeros((Vh.shape[0], U.shape[0]))
    for i in range(len(sigma)):
        if sigma[i] > 1e-12:  # 避免对接近 0 的奇异值求倒数
            sigma_plus[i, i] = 1 / sigma[i]
    
    # 3. 计算 A^+ = V Σ^+ U^H
    A_plus = Vh.T @ sigma_plus @ U.T
    
    return A_plus

def moore_penrose_pseudoinverse_byfr(A):
    """
    利用满秩分解法计算矩阵的 Moore-Penrose 广义逆
    :param A: 输入矩阵 A (m x n)
    :return: A 的广义逆 A^+
    """
    m, n = A.shape
    
    # 用LU分解构造满秩分解
    P, L, U = lu(A)  # LU 分解
    r = np.linalg.matrix_rank(A)  # 矩阵秩
    
    # 提取 L 和 U 中的秩相关部分
    B = L[:, :r]  # L 的前 r 列
    C = U[:r, :]  # U 的前 r 行

    F = np.dot(P , B)
    G = C

    # 求满秩矩阵的广义逆
    F_plus = np.linalg.inv(F.T @ F) @ F.T
    G_plus = G.T @ np.linalg.inv(G @ G.T)

    # 构造 A 的广义逆
    A_plus = G_plus @ F_plus

    return A_plus

# 验证 Penrose 条件
def verify_penrose_conditions(A, A_plus):
    """
    验证 Penrose 条件是否成立
    :param A: 矩阵 A
    :param A_plus: A 的广义逆
    :return: 各个 Penrose 条件是否成立
    """
    # 1. A * A+ * A = A
    cond1 = np.allclose(A @ A_plus @ A, A)
    # 2. A+ * A * A+ = A+
    cond2 = np.allclose(A_plus @ A @ A_plus, A_plus)
    # 3. (A * A+)^H = A * A+
    cond3 = np.allclose((A @ A_plus).T.conj(), A @ A_plus)
    # 4. (A+ * A)^H = A+ * A
    cond4 = np.allclose((A_plus @ A).T.conj(), A_plus @ A)
    
    return cond1, cond2, cond3, cond4

# 测试
if __name__ == "__main__":
    # 定义一个矩阵 A
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    # 打印结果
    print("矩阵 A:")
    print(A)
    print("\n------使用满秩分解计算矩阵广义逆------")
    # 计算广义逆
    A_plus = moore_penrose_pseudoinverse_byfr(A)
    print("\n矩阵 A 的广义逆 A^+:")
    print(A_plus)
    # 验证 Penrose 条件
    cond1, cond2, cond3, cond4 = verify_penrose_conditions(A, A_plus)
    print("\nPenrose 条件验证结果：")
    print(f"1. A * A+ * A = A: {cond1}")
    print(f"2. A+ * A * A+ = A+: {cond2}")
    print(f"3. (A * A+)^H = A * A+: {cond3}")
    print(f"4. (A+ * A)^H = A+ * A: {cond4}")
    
    print("\n------使用SVD分解计算矩阵广义逆------")
    # 计算 A 的广义逆
    A_plus = moore_penrose_pseudoinverse_bysvd(A)
    print("A 的广义逆 A^+：")
    print(A_plus)
    
    # 验证 Penrose 条件
    cond1, cond2, cond3, cond4 = verify_penrose_conditions(A, A_plus)
    print("\nPenrose 条件验证结果：")
    print(f"1. A * A+ * A = A: {cond1}")
    print(f"2. A+ * A * A+ = A+: {cond2}")
    print(f"3. (A * A+)^H = A * A+: {cond3}")
    print(f"4. (A+ * A)^H = A+ * A: {cond4}")
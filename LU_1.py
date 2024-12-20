import numpy as np
import time

def lu_decomposition(A):
    """
    对方阵 A 进行 LU 分解，将其分解为下三角矩阵 L 和上三角矩阵 U。
    输入：
        A: 方阵（numpy 数组）
    输出：
        L: 下三角矩阵（对角线元素为 1）
        U: 上三角矩阵
    """
    n = A.shape[0]
    L = np.eye(n)  # 初始化 L 为单位矩阵
    U = A.copy()   # 创建矩阵 A 的副本，用于存储 U

    for k in range(n - 1):
        if U[k, k] == 0:
            raise ValueError("矩阵是奇异的，无法进行 LU 分解。")
        tau = U[k + 1:n, k] / U[k, k]
        L[k + 1:n, k] = tau
        U[k + 1:n, :] -= np.outer(tau, U[k, :])

    return L, U

def solve_lu(L, U, b):
    """
    利用 LU 分解求解线性方程组 Ax = b。
    输入：
        L: 下三角矩阵
        U: 上三角矩阵
        b: 右侧列向量
    输出：
        x: 解向量
    """
    # 前向替代法，求解 Lz = b
    n = L.shape[0]
    z = np.zeros(n)
    for i in range(n):
        z[i] = b[i] - np.dot(L[i, :i], z[:i])

    # 回代法，求解 Ux = z
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (z[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x

# 比较性能和误差
def compare_lu_methods(n):
    """
    比较自实现 LU 分解与库函数在性能和误差上的差异
    """
    # 随机生成矩阵和向量
    np.random.seed(42)
    A = np.random.rand(n, n) * 10
    b = np.random.rand(n) * 10

    # 自实现 LU 分解
    start_time = time.time()
    L, U = lu_decomposition(A)
    x_custom = solve_lu(L, U, b)
    custom_time = time.time() - start_time

    # 使用库函数
    start_time = time.time()
    x_numpy = np.linalg.solve(A, b)
    numpy_time = time.time() - start_time

    # 计算误差
    custom_error = np.linalg.norm(np.dot(A, x_custom) - b)
    numpy_error = np.linalg.norm(np.dot(A, x_numpy) - b)

    # 输出结果
    print(f"矩阵维度: {n}×{n}")
    print(f"自实现 LU 分解时间: {custom_time:.6f} 秒")
    print(f"库函数计算时间: {numpy_time:.6f} 秒")
    print(f"自实现 LU 分解误差: {custom_error:.6e}")
    print(f"库函数计算误差: {numpy_error:.6e}")

# 比较不同矩阵维度下的性能和误差
for n in [10, 100, 2000]:
    compare_lu_methods(n)

def test():
    # 随机生成矩阵 A 和向量 b
    np.random.seed(42)  # 固定随机种子以复现结果
    n = 4  # 矩阵维度
    A = np.random.rand(n, n) * 10  # 随机生成 n×n 的矩阵
    b = np.random.rand(n) * 10     # 随机生成长度为 n 的向量

    # LU 分解
    L, U = lu_decomposition(A)

    # 求解线性方程组
    x = solve_lu(L, U, b)

    # 验证结果
    print("随机生成的矩阵 A:")
    print(A)
    print("\n随机生成的向量 b:")
    print(b)
    print("\n下三角矩阵 L:")
    print(L)
    print("\n上三角矩阵 U:")
    print(U)
    print("\n求解得到的解 x:")
    print(x)
    print("\n验证 Ax 是否接近 b:")
    print(np.dot(A, x))
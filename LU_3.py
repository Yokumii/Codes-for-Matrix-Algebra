import numpy as np
from numba import jit
import scipy.linalg
import time

# 创建具有指定条件数的矩阵
def create_matrix_with_condition_number(n, cond_number):
    # 确保矩阵大小
    if cond_number < 1:
        raise ValueError("Condition number must be greater than or equal to 1")

    # 创建一个正交矩阵 Q
    Q = np.random.randn(n, n)
    Q, _ = np.linalg.qr(Q)  # QR分解得到正交矩阵

    # 创建一个对角矩阵 D，其中包含控制条件数的奇异值
    singular_values = np.linspace(1, cond_number, n)  # 奇异值从1到cond_number线性递增

    D = np.diag(singular_values)

    # 计算最终的矩阵 A
    A = np.dot(np.dot(Q, D), Q.T)
    
    return A

# 测试矩阵生成
def generate_test_data(N):
    np.random.seed(40)  # 固定随机种子，保证可重复性
    A = create_matrix_with_condition_number(N, 10)
    x_true = np.random.rand(N)
    b = A @ x_true
    return A, b, x_true

# 高斯-若尔当法
@jit(nopython=True)
def gauss_jordan(A, b):
    n = len(b)
    Aug = np.hstack((A, b.reshape(-1, 1)))  # 增广矩阵
    for i in range(n):
        # 选主元
        max_row = i + np.argmax(np.abs(Aug[i:, i]))
        if max_row != i:  # 交换行
            Aug[i], Aug[max_row] = Aug[max_row].copy(), Aug[i].copy()
        # 归一化主元行
        Aug[i] /= Aug[i, i]
        # 消元
        for j in range(n):
            if j != i:
                Aug[j] -= Aug[j, i] * Aug[i]
    return Aug[:, -1]

# 高斯消元法
@jit(nopython=True)
def gauss_elimination(A, b):
    n = len(b)
    A = A.copy()
    b = b.copy()
    for i in range(n):
        # 选主元
        max_row = i + np.argmax(np.abs(A[i:, i]))
        if max_row != i:  # 交换行
            A[i], A[max_row] = A[max_row].copy(), A[i].copy()
            b[i], b[max_row] = b[max_row], b[i]
        # 消元
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]
    # 回代
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
    return x

# 自定义LU分解
@jit(nopython=True)
def my_lu(A):
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    for i in range(n - 1):
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            U[j, i:] -= factor * U[i, i:]
            L[j, i] = factor
    return L, U

@jit(nopython=True)
def my_plu(A):
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)
    
    for i in range(n):
        # 选主元
        max_row = i + np.argmax(np.abs(U[i:, i]))
        
        if max_row != i:  # 交换行
            # 交换 U 的行
            temp = U[i].copy()
            U[i] = U[max_row]
            U[max_row] = temp
            
            # 交换 P 的行
            temp = P[i].copy()
            P[i] = P[max_row]
            P[max_row] = temp
            
            if i > 0:  # 交换 L 的行（仅交换左上三角部分）
                temp = L[i, :i].copy()
                L[i, :i] = L[max_row, :i]
                L[max_row, :i] = temp

        # 计算 L 和 U
        for j in range(i + 1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] -= L[j, i] * U[i, i:]
            U[j, i] = 0.0  # 保证下三角部分是 0

    return P, L, U

# 测试方法
def test_methods(N=5000):
    A, b, x_true = generate_test_data(N)

    # 输出矩阵条件数
    cond_A = np.linalg.cond(A)
    print("A的条件数为:", cond_A)

    print("A = ")
    print(A)
    print("\nx = ")
    print(x_true)
    print("\nComputed b (Ax) = ")
    print(b)

    # 高斯-若尔当法
    start = time.time()
    x_gj = gauss_jordan(A, b)
    time_gj = time.time() - start

    # 高斯消元法
    start = time.time()
    x_ge = gauss_elimination(A, b)
    time_ge = time.time() - start

    # SciPy LU 分解
    start = time.time()
    P, L, U = scipy.linalg.lu(A)
    x_scipy_lu = np.dot(np.linalg.inv(U), np.dot(np.linalg.inv(L), P @ b))
    time_scipy_lu = time.time() - start

    # 自定义LU分解
    start = time.time()
    L, U = my_lu(A)
    x_my_lu = np.dot(np.linalg.inv(U), np.dot(np.linalg.inv(L), b))
    time_my_lu = time.time() - start

    # 自定义PLU分解
    start = time.time()
    P, L, U = my_plu(A)
    x_my_plu = np.dot(np.linalg.inv(U), np.dot(np.linalg.inv(L), P @ b))
    time_my_plu = time.time() - start

    # 输出结果
    print("高斯-若尔当法时间：", time_gj, "误差：", np.linalg.norm(x_true - x_gj))
    print("高斯消元法时间：", time_ge, "误差：", np.linalg.norm(x_true - x_ge))
    print("SciPy LU分解时间：", time_scipy_lu, "误差：", np.linalg.norm(x_true - x_scipy_lu))
    print("自定义LU分解时间：", time_my_lu, "误差：", np.linalg.norm(x_true - x_my_lu))
    print("自定义PLU分解时间：", time_my_plu, "误差：", np.linalg.norm(x_true - x_my_plu))

# # 运行测试
# test_methods(5000)

import matplotlib.pyplot as plt

# 测试方法性能
def test_performance(dimensions):
    methods = {
        "Gauss-Jordan": gauss_jordan,
        "Gauss Elimination": gauss_elimination,
        "SciPy LU": lambda A, b: scipy.linalg.lu_solve(scipy.linalg.lu_factor(A), b),
        "Custom LU": lambda A, b: np.linalg.solve(*my_lu(A), b),
        "Custom PLU": lambda A, b: np.linalg.solve(*my_plu(A), b),
    }
    
    timings = {key: [] for key in methods.keys()}
    
    for N in dimensions:
        A, b, _ = generate_test_data(N)
        
        for method_name, method in methods.items():
            start = time.time()
            try:
                method(A, b)
            except Exception:
                pass
            timings[method_name].append(time.time() - start)
    
    return timings

# 绘制性能曲线
def plot_performance(dimensions, timings):
    for method_name, times in timings.items():
        plt.plot(dimensions, times, label=method_name)
    plt.xlabel("Matrix Dimension")
    plt.ylabel("Execution Time (s)")
    plt.title("Performance Comparison of Linear Solvers")
    plt.legend()
    plt.grid(True)
    plt.show()

# 执行性能测试
dimensions = range(100, 2001, 2)  # 从100到2000，每隔200取一个值
timings = test_performance(dimensions)
plot_performance(dimensions, timings)
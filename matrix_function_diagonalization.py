import numpy as np
from scipy.linalg import expm, eig
from sympy import cos, diag

def matrix_function_diagonalization(A, func):
    """
    用对角化法计算矩阵函数。
    
    参数：
    - A: 原矩阵
    - func: 对特征值应用的函数（如 np.exp、lambda x: np.exp(t * x) 等）
    
    返回：
    - f_A: 矩阵函数的值
    """
    # 求特征值和特征向量
    eigvals, eigvecs = eig(A)
    # 检查是否为对角化矩阵
    if np.linalg.matrix_rank(eigvecs) < A.shape[0]:
        raise ValueError("矩阵不可对角化，请考虑使用 Jordan 标准形法。")
    
    # 对特征值应用函数
    diag_func = np.diag([func(val) for val in eigvals])
    # 恢复原矩阵函数
    f_A = eigvecs @ diag_func @ np.linalg.inv(eigvecs)
    return np.real(f_A)  # 去掉可能的浮点数误差

# 定义矩阵 A
A = np.array([
    [4, 6, 0],
    [-3, -5, 0],
    [-3, -6, 1]
], dtype=float)

# 计算 e^A
e_A = matrix_function_diagonalization(A, np.exp)
print("e^A =\n", e_A)

# 定义 t
t = 2.0  # 由于符号求解不便，这里设 t = 2
e_tA = matrix_function_diagonalization(A, lambda x: np.exp(t * x))
print(f"e^({t}A) =\n", e_tA)

# 计算 cos(A)
cos_A = matrix_function_diagonalization(A, np.cos)
print("cos(A) =\n", cos_A)
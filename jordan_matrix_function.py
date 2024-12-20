import numpy as np

def compute_matrix_function(A, func):
    """
    计算给定矩阵的函数值，基于其 Jordan 标准形。
    
    参数:
    - A: 输入矩阵 (numpy.ndarray)
    - func: 需要计算的函数，作用于矩阵对角块 (例如: np.sin, np.cos)
    
    返回:
    - 矩阵函数值 (numpy.ndarray)
    """

    J = A.copy()
    
    # 创建与 J 同大小的零矩阵用于结果存储
    func_J = np.zeros_like(J, dtype=float)
    
    # 填充矩阵函数
    for i in range(len(J)):
        for j in range(len(J)):
            if i == j:  # 对角元素，直接作用于 func
                func_J[i, j] = func(J[i, j])
            elif j == i + 1 and J[i, j] == 1:  # 上对角线 Jordan 块
                if func == np.sin:  # 针对 sin 的特殊处理
                    func_J[i, j] = np.cos(J[i, i])
                elif func == np.cos:  # 针对 cos 的特殊处理
                    func_J[i, j] = -np.sin(J[i, i])
                # 其他矩阵函数（待扩展）
    return func_J

if __name__ == "__main__":

    A = np.array([
        [np.pi, 0, 0, 0],
        [0, -np.pi, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    
    # 计算 sin(A)
    sin_A = compute_matrix_function(A, np.sin)
    print("sin(A) =\n", sin_A)
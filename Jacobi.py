import numpy as np
from numpy.linalg import eig, norm

# Jacobi 迭代法
def jacobi_method(A, b, x0, tol=1e-6, max_iter=100):
    n = len(b)
    x = x0.copy()
    D = np.diag(np.diag(A))  # 提取对角矩阵 D
    R = A - D               # 剩余部分 R = A - D
    B = -np.linalg.inv(D).dot(R)  # 迭代矩阵 B
    c = np.linalg.inv(D).dot(b)   # 常向量 c

    # 检查谱半径
    eig_values, _ = eig(B)
    spectral_radius = max(abs(eig_values))
    print(f"迭代矩阵 B 的谱半径: {spectral_radius:.6f}")
    
    if spectral_radius >= 1:
        print("警告：谱半径大于或等于 1，迭代法可能不收敛！")
    else:
        print("谱半径小于 1，迭代法可能收敛。")
    
    # 迭代过程
    for k in range(max_iter):
        x_new = B.dot(x) + c
        if norm(x_new - x) < tol:  # 判断收敛条件
            print(f"迭代法收敛于第 {k+1} 次迭代")
            return x_new, k+1
        x = x_new

    print("迭代法未能在最大迭代次数内收敛")
    return x, max_iter

# 测试
if __name__ == "__main__":
    A1 = np.array([[2, -1], 
                   [-100, 7]])  # 非对角占优
    b1 = np.array([1, 2])
    x0_1 = np.zeros(len(b1))
    jacobi_method(A1, b1, x0_1)
    print("\n")
    
    A2 = np.array([[4, 1], 
                   [1, 3]])  # 对角占优
    b2 = np.array([1, 2])
    x0_2 = np.zeros(len(b2))
    jacobi_method(A2, b2, x0_2)
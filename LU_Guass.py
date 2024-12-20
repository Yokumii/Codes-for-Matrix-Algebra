# 比较的矩阵维度范围
dimensions = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
m = 10  # 固定多组 b 的数量

gauss_times = []  # 高斯消元法总时间
lu_decompose_times = []  # LU 分解阶段时间
lu_solve_times = []  # LU 求解阶段时间
lu_total_times = []  # LU 总时间

for n in dimensions:
    # 随机生成矩阵和右端向量
    A = np.random.rand(n, n)
    b_list = [np.random.rand(n) for _ in range(m)]
    
    # 记录高斯消元法的时间
    start = time.time()
    for b in b_list:
        gauss_elimination(A, b)  # 调用上面的高斯消元函数
    gauss_times.append(time.time() - start)
    
    # 记录 LU 分解时间
    start = time.time()
    P, L, U = scipy.linalg.lu(A)  # 分解阶段
    lu_decompose_times.append(time.time() - start)
    
    # 记录 LU 求解阶段时间
    start = time.time()
    for b in b_list:
        y = np.linalg.solve(L, P @ b)  # 解 Ly = Pb
        x_lu = np.linalg.solve(U, y)  # 解 Ux = y
    lu_solve_times.append(time.time() - start)
    
    # 计算 LU 总时间
    lu_total_times.append(lu_decompose_times[-1] + lu_solve_times[-1])

# 绘图
plt.figure(figsize=(10, 6))

# 高斯消元法
plt.plot(dimensions, gauss_times, label="Gaussian Elimination", marker="o")

# LU 分解
plt.plot(dimensions, lu_decompose_times, label="LU Decomposition (分解阶段)", marker="s")
plt.plot(dimensions, lu_solve_times, label="LU Solve (求解阶段)", marker="x")
plt.plot(dimensions, lu_total_times, label="LU 总时间", marker="^")

# 图形样式
plt.title("运行效率: Gaussian Elimination vs LU Decomposition", fontsize=14)
plt.xlabel("矩阵维度(n)", fontsize=12)
plt.ylabel("运行时间(s)", fontsize=12)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
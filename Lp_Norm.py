# 计算Lp范数，默认p = 2，计算Euclidean范数
def cal_LpNorm(x, p = 2):
    if p == np.Inf: #无穷范数
        return np.max(np.abs(x))
    return np.sum(np.abs(x)**p)**(1 / p)
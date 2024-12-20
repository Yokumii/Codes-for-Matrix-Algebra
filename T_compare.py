import numpy as np
import copy
from sympy import Matrix
   
class linear_space(object):
    
    def __init__(self, basis = [], number_field = complex):
        self.basis = basis #基
        self.number_field = number_field #数域

    def dim(self):
        return(len(self.basis)) #维数
    

class inner_product_space(linear_space): #内积空间
    
    def __init__(self, basis = [], number_field = complex, inner_product= 'None'):
        linear_space.__init__(self, basis, number_field)
        self.inner_product = inner_product
        self.gram_schmidt()
        
    def gram_schmidt(self): #施密特单位正交化
        temp_vectors = copy.copy(self.basis)
        Len = self.dim()
        result = []
        for k in range(Len):
            # 当前处理的向量
            current_vector = temp_vectors[k]
            # 当前向量归一化
            current_vector = current_vector / np.sqrt(self.inner_product(current_vector, current_vector))
            # 从其他向量中移除当前向量的投影
            for j in range(k+1, Len):
                temp_vectors[j] = temp_vectors[j] - (self.inner_product(current_vector, temp_vectors[j])) * current_vector
            # 添加正交化后的向量到结果中
            result.append(current_vector)
        self.basis = result
   
    
class element(object):
    
    def __init__(self, linear_space, info = 'coordinate', infomation = []):
        self.linear_space = linear_space #线性空间
        if info == 'coordinate':
            self.list2coordinate(infomation) #坐标
        if info == 'value':
            self.value2coordinate(infomation) #值
            
    def list2coordinate(self, coordinate):
        self.coordinate = []
        for line in coordinate:
            self.coordinate.append(self.linear_space.number_field(line))
        self.coordinate = np.array(self.coordinate)
 
    def value2coordinate(self, value): #计算坐标
        Len = self.linear_space.dim()
        self.coordinate = []
        for k in range(0, Len):
            self.coordinate.append(self.linear_space.inner_product(value, self.linear_space.basis[k]))
        self.coordinate = np.array(self.coordinate)
    
    def value(self): #计算数值
        v = self.linear_space.basis[0] * self.coordinate[0]
        Len = self.linear_space.dim()
        for k in range(1, Len):
            v = v + self.linear_space.basis[k] * self.coordinate[k]
        return v



class linear_transformation(object):
    
    def __init__(self, linear_space, transformation):
        self.linear_space = linear_space #线性空间
        self.trans = transformation #变换
        self.trans2matrix() #线性变换在标准正交基下的矩阵
        
    def trans2matrix(self): #线性变换在标准正交基下的矩阵
        te = []
        for line in self.linear_space.basis:
            te.append(self.trans(line))
        Len = self.linear_space.dim()
        self.matrix = np.zeros([Len,Len])
        for j in range(0, Len):
            for k in range(0, Len):
                self.matrix[k,j] = self.linear_space.inner_product(te[j], self.linear_space.basis[k])
        
        self.matrix = Matrix(self.matrix)
        P, J = self.matrix.jordan_form() #self.matrix = PJ(P^-1)
        invP = P.inv()
        self.P = P
        self.J = J
        self.invP = invP
        # print((self.matrix - P*J*invP).norm()) #self.matrix = PJ(P^-1)        
        
        
    def linear_trans(self, input_ele): #利用矩阵相乘来计算坐标
        output_ele = copy.copy(input_ele)
        # output_ele.coordinate = np.dot(self.matrix, input_ele.coordinate) #直接乘矩阵
        output_ele.coordinate = np.dot(self.P*self.J*self.invP, input_ele.coordinate) #利用jordan标准型
        return output_ele

    def dot(self, arr): #矩阵相乘
        y = np.dot(self.matrix, arr) 
        return y

    def dotJ(self, arr): #矩阵相乘
        y = np.dot(self.J, arr) 
        return y
    
    def apply_function(self, func): #将线性变换的函数作用于基本线性变换中，并且修改矩阵
        # self.matrix = func(self.dot)(np.eye(self.linear_space.dim())) #直接计算，满矩阵相乘
        
        self.J = func(self.dotJ)(np.eye(self.linear_space.dim())) #基于f(T) 计算f(J)，更新为J
        self.matrix = self.P*self.J*self.invP #更新矩阵
        self.trans = func(self.trans)
        
        
#########################################################################

def basis_coordinate(x, basis):
    '''
    通过坐标变换求元素x在其他基(X1,X2,X3)下的坐标beta
    首先有x = (e1,e2,e3) alpha #(e1,e2,e3)为线性空间V的标准正交基，alpha 为x在该基下的坐标
    先求过渡矩阵C：(X1,X2,X3) = (e1,e2,e3)C
    有 (e1,e2,e3) = (X1,X2,X3) C^-1, 带入上上式
    得 x = (X1,X2,X3) C^-1 *alpha
    得 beta = C^-1 *alpha
    满足 x = (X1,X2,X3) C^-1 * beta
    
    仅示意，实际使用要具体情况具体再写，不要直接调用本函数，否则每次都会重新计算过渡矩阵和逆，造成冗余。
    应该提前把过渡矩阵和逆计算好再保存，然后每次求坐标的时候进行调用即可
    '''
    alpha = x.coordinate #线性变换在标准正交基(e1,e2,e3)下的矩阵A已经提前算好了
    ls = x.linear_space #线性空间
    C = []
    Len = len(basis) #等于T.linear_space.dim()
    for j in range(0, Len):
        xj = element(ls, 'value', basis[j]) 
        C.append(xj.coordinate) #求Xj 在标准正交基下的坐标
    C = np.array(C).transpose() #把坐标合并成矩阵，过渡矩阵，满足(X1,X2,X3) = (e1,e2,e3)C, 注意转置
    invC = np.linalg.inv(C) #求C的逆C^-1
    beta = np.dot(invC, alpha) #B = C^-1 *A*C
    return beta


def trans_basis_matrix(T, basis):
    '''
    通过坐标变换求线性变换在其他基(X1,X2,X3)下的矩阵
    由 T(e1,e2,e3) = (e1,e2,e3)A #(e1,e2,e3)为线性空间V的标准正交基
    先求过渡矩阵C：(X1,X2,X3) = (e1,e2,e3)C
    有 (e1,e2,e3) = (X1,X2,X3) C^-1, 带入上上式
    得T(X1,X2,X3) C^-1 = (X1,X2,X3) C^-1 *A
    即T(X1,X2,X3)  = (X1,X2,X3) C^-1 *A*C
    得 B = C^-1 *A*C
    满足 T(X1,X2,X3) = (X1,X2,X3)B
    '''
    A = T.matrix #线性变换在标准正交基(e1,e2,e3)下的矩阵A已经提前算好了
    C = []
    Len = len(basis) #等于T.linear_space.dim()
    for j in range(0, Len):
        xj = element(ls, 'value', basis[j]) 
        C.append(xj.coordinate) #求Xj 在标准正交基下的坐标
    C = np.array(C).transpose() #把坐标合并成矩阵，过渡矩阵，满足(X1,X2,X3) = (e1,e2,e3)C, 注意转置
    invC = np.linalg.inv(C) #求C的逆C^-1
    B = np.dot(invC, np.dot(A,C)) #B = C^-1 *A*C
    return B

#########################################################################

def mapping(x): #基本的线性变换
    #换一个线性变换，带乘法，增加计算复杂度，方便测试。注意构造线性变换y=Tx的时候，保障y也属于该线性空间
    y = np.dot(np.array([[2,0],[5,4]]) , x) + np.dot(x,np.array([[2,-2],[2,-2]]))
    return y

def function(T): #线性变换的函数
    def inner_function(x):
        
        #人为增加计算量，模拟复杂问题
        for k in range(1000):
            y = T(x) 
            y = y*0  # 作零变换防止数值溢出
            
            
        basis = [x] #快速算法, 求多项式基（仿秦九韶算法）
        N = 3
        for n in range(0,N):
            basis.append(T(basis[-1]))
        
        number_field = float #实数域
        ls = linear_space(basis, number_field) #线性变换的幂张成一个线性空间
        alpha = [0, 0, 0, 1]
        a = element(ls, 'coordinate', alpha)
        y = a.value()
        return y
    return inner_function

def inner_product(x, y): #定义内积
    return np.sum(x*y)


import time
import matplotlib.pyplot as plt

def compare_methods(max_degree=10, num_tests=100):
    """
    比较直接计算方法和基于Jordan标准型的方法
    """
    degrees = []
    direct_times = []
    jordan_times = []
    
    # 设置基空间和线性变换
    basis = [np.array([[-1, 1], [0, 0]]), np.array([[0, 0], [1, 0]]), np.array([[0, 0], [0, 1]])]
    number_field = float
    ls = inner_product_space(basis, number_field, inner_product)  # 内积空间
    lt = linear_transformation(ls, mapping)  # 基线性变换

    # 测试多项式次数逐渐增加
    for degree in range(1, max_degree + 1):
        # 更新多项式函数
        def function_decorator(T):
            def inner_function(x):
                # 基于多项式次数的变化
                basis = [x]
                for _ in range(degree):  # 多项式次数逐渐增加
                    basis.append(T(basis[-1]))
                number_field = float
                ls = linear_space(basis, number_field)
                alpha = [1] * (degree + 1)
                a = element(ls, 'coordinate', alpha)
                return a.value()
            return inner_function

        # 更新 function
        function = function_decorator

        # 直接计算的时间
        start_time = time.time()
        for _ in range(num_tests):
            x = np.random.rand(2, 2)
            b = function(mapping)(x)  # 直接计算
        direct_times.append(time.time() - start_time)

        # 基于 Jordan 标准型计算的时间
        start_time = time.time()
        flt = copy.deepcopy(lt)
        flt.apply_function(function)  # 修改变换
        for _ in range(num_tests):
            x = np.random.rand(2, 2)
            a = element(ls, 'value', x)
            c = flt.linear_trans(a)
        jordan_times.append(time.time() - start_time)

        degrees.append(degree)

    return degrees, direct_times, jordan_times


def plot_comparison(degrees, direct_times, jordan_times):
    """
    绘制两种方法的计算时间对比
    """
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, direct_times, label="Direct Calculation", marker='o')
    plt.plot(degrees, jordan_times, label="Jordan Standard Form", marker='x')
    plt.xlabel("Degree of Polynomial")
    plt.ylabel("Time (seconds)")
    plt.title("Comparison of Direct vs Jordan Standard Form Calculations")
    plt.legend()
    plt.grid(True)
    plt.show()


# 测试与绘图
if __name__ == '__main__':
    max_degree = 500
    num_tests = 100
    degrees, direct_times, jordan_times = compare_methods(max_degree=max_degree, num_tests=num_tests)
    plot_comparison(degrees, direct_times, jordan_times)



    
    
    
    
    
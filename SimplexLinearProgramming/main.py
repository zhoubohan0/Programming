import numpy as np
import os
import csv
import pickle

M = 1e6


def inputtestdata(num):
    # M = 1e6

    # data1:
    # A = [[1, 1, 1, 1, 0, 0, 0],
    #      [-2, 1, -1, 0, -1, 1, 0],
    #      [0, 3, 1, 0, 0, 0, 1]]
    # b = [4, 1, 9]
    # c = [-3, 0, 1, 0, 0, -M, -M]
    # base = [3, 5, 6]
    # result1: X*= [0.  2.5 1.5 0.  0.  0.  0.];z_max = 1.5(sole)

    # data2:
    # A = [[2,2,1,0,0],[4,0,0,1,0],[0,5,0,0,1]]
    # b = [12,16,15]
    # c = [3,3,0,0,0]
    # base = [2,3,4,]
    # result2: X*= [4. 2. 0. 0. 5.]            ;z_max=18.0(infinite)

    # data3:
    # A = [[2, 2, 1, 0, 0, 0], [1, 2, 0, 1, 0, 0], [4, 0, 0, 0, 1, 0], [0, 4, 0, 0, 0, 1]]
    # b = [12, 8, 16, 12]
    # c = [2, 3, 0, 0, 0, 0]
    # base = [2, 3, 4, 5]
    # result3: X*=[4. 2. 0. 0. 0. 4.]          ;z_max=14.0(sole)

    # data4:
    # A = [[4, 0, 1]]
    # b = [16]
    # c = [2, 3, 0, ]
    # base = [2]
    # result4:(unbounded solution)

    # data5:
    # A = [[2, 2, 1, 0, 0], [1, 2, 0, -1, 1], ]
    # b = [6, 7]
    # c = [2, 3, 0, 0, -M]
    # base = [2, 4]
    # result5:(no solution)
    with open(f'./testdata/test{num}.pkl', 'rb') as f:
        d = pickle.load(f)
        return d['A'], d['b'], d['c'], d['base']


def outputmatrix(writer, m, n, A, c_b, x, base, dec=2):
    for i in range(m):
        outstr = [round(each, dec) for each in A[i]]
        writer.writerow([round(c_b[i], dec), round(base[i], dec), round(x[i], dec), '|', *outstr])
        print(c_b[i], '\t', base[i], '\t', x[i], '\t', *outstr)
    writer.writerow(['---------'] * (n + 4))
    print('-----------------------------------------------------')


def outputlist(writer, n, sigma, dec=2):
    for _ in sigma:
        print(_, end='\t')
    print('\n-----------------------------------------------------')
    outstr = [round(each, dec) for each in sigma]
    writer.writerow(['', 'sigma_j', '', '|', *outstr])
    writer.writerow(['---------'] * (n + 4))


def outputresult(writer, x_opt, z_opt,solutioncondition, dec=2):
    print(f'X*={x_opt}\nz_max={z_opt}\n{solutioncondition} solution')
    outstr = [round(each, dec) for each in x_opt]
    writer.writerow(['x_optimal=', *outstr])
    writer.writerow(['z_max=', round(z_opt, dec)])
    writer.writerow([solutioncondition+' solution'])

def checkall(l, mode='+0'):
    if mode == '+0':
        r = np.array([1 if i >= 0 else 0 for i in l])
    if mode == '-0':
        r = np.array([1 if i <= 0 else 0 for i in l])
    if mode == '+':
        r = np.array([1 if i > 0 else 0 for i in l])
    if mode == '-':
        r = np.array([1 if i < 0 else 0 for i in l])
    return len(r) == sum(r), r


def SimplexLP(A, b, c, base, outputpath):
    # 基本设置
    m = len(base)
    n = len(c)
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)
    base = np.array(base, dtype=np.int64)
    c_b = c[base]
    x = b  # 基解的非零部分，初始为b
    print(f'变量从X1到X{n}')
    # 写入表头以及初始数据
    writer = csv.writer(open(outputpath, 'w', newline=''))
    writer.writerow(['', '', '', '|', *c])
    writer.writerow(['C_Base', 'Base', 'b', '|', *[f'x_{i}' for i in range(1, n + 1)]], )
    writer.writerow(['---------'] * (n + 4))
    outputmatrix(writer, m, n, A, c_b, x, base)
    while True:
        # 计算检验数，判断结束
        c_b = c[base]
        sigma = [c[j] - np.dot(c_b, A[:, j]) for j in range(n)]
        outputlist(writer, n, sigma)
        judge, _ = checkall(sigma, '-0')
        j = np.argmax(sigma)
        _, positive = checkall(A[:, j], '+')
        theta = [x[i] / A[i, j] if positive[i] else M for i in range(m)]
        i = np.argmin(theta)
        if judge:  # 1.判断当前顶点是否是最优解
            solutioncondition = 'sole'
            for col in range(n):  # 2.判断是否无穷解（最优解在直线上或平面上取到)
                notpositive,_ = checkall(A[:, col], '-0')  # 不能全负代表存在theta
                if sigma[col] == 0 and col not in base and not notpositive:  # 所有sigma非正，对某非基变量sigma=0且能找到theta
                    solutioncondition = 'infinite'
                    break
            x_opt = np.zeros_like(c)
            x_opt[base] = x
            z_opt = np.dot(c_b, x)
            if z_opt<0:
                solutioncondition = 'no'
            outputresult(writer, x_opt, z_opt,solutioncondition)
            return x_opt, z_opt,solutioncondition
        else:  # 3.判断是否无界解
            for col in range(n):
                ne,_ = checkall(A[:,j],'-0')
                if sigma[col]>0 and ne:
                    solutioncondition = 'unbounded'
                    x_opt = np.ones_like(c)*M
                    z_opt = M
                    outputresult(writer, x_opt, z_opt,solutioncondition)
                    return x_opt, z_opt, solutioncondition
        # A矩阵线性变换
        B = np.hstack((A, x[:, np.newaxis]))  # 增广矩阵
        center = A[i, j]
        # 先将第i行“归一化”，再处理其他行
        B[i] = B[i] / center
        for row in range(m):
            if row != i:
                B[row] = B[row] - B[i] * B[row, j]
        # 更新
        x = B[:, -1]
        A = B[:, :-1]
        base[i] = j  # i号出基，j号进基
        c_b = c[base]
        print(f'{i}号出基，{j}号进基')
        outputmatrix(writer, m, n, A, c_b, x, base)


if __name__ == '__main__':
    # 数据输入（备注：需整理成线性规划标准型,没有显示单位阵的要构造人工变量）
    # A:技术系数,一定是二维列表
    # b:限额系数,一定是一维列表
    # c:价值系数,一定是一维列表
    # base:初始基，数字对应分量从x_0到x_{m-1},一定是一维列表
    A,b,c,base = inputtestdata(num=1)

    A = [[4,3,8,1,0,0],[4,1,12,0,1,0],[4,-1,3,0,0,1]]
    b = [12,8,8]
    c = [2,1,2,0,0,0]
    base = [3,4,5]
    # 单纯形法求解线性规划
    x_opt, z_opt,solutioncondition = SimplexLP(A, b, c, base, outputpath='./output/SimplexTable.csv')

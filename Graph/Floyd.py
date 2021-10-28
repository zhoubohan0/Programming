import numpy as np
def floyd(A, printprocess=False):
    if not type(A) == type(np.zeros(1)):
        A = np.array(A)
    assert A.shape[0] == A.shape[1]
    n = len(A)
    K = int(np.log2(n - 1))+1  # 最大迭代次数
    D = A.copy()
    for k in range(K):
        D_ = [min(D[i] + D[j]) for i in range(n) for j in range(n)]
        D_ = np.array(D_).reshape(D.shape)
        if (D_ == D).sum() == n*n:  # 收敛
            print(f'第{k+1}次迭代后收敛')
            return D
        else:  # 继续迭代
            D = D_
        if printprocess:
            print(f'第{k + 1}次迭代结果:')
            print(D)
            print('----------------------------------------')

if __name__ == '__main__':
    # 两个不可达的节点之间距离∞，节点到自身距离为0
    inf = int(1e8)
    A = [
        [0, 2, inf, 6],
        [2, 0, 3, 2],
        [inf, 3, 0, 2],
        [6, 2, 2, 0]
    ]
    print(floyd(A, True))

import numpy as np


class AssignmentProblem:
    def __init__(self, A):
        A = np.array(A)
        m, n = A.shape
        assert m == n  # 效率矩阵应是方阵
        self.A = A
        self.X = np.zeros_like(self.A)  # 解,可以作为全局变量因为无论怎么更新都不会使其变得更差
        self.n = n
        self.z_min = 0

    def sole0(self, x, delete):  # delete=1表示要删除的部分,判断向量中是否仅含一个0，返回其索引
        for i in range(len(x)):
            if delete[i]:
                x[i] = -1
        tmp = np.where(x == 0)[0]
        if len(tmp) == 1:
            return True, tmp[0]
        else:
            return False, -1

    def OR(self, vec_x, vec_y):
        return np.array([x | y for x in vec_x.astype(np.bool_) for y in vec_y.astype(np.bool_)]).reshape(self.n, self.n)

    def output(self):
        print('最优解：')
        for i in range(self.n):
            for j in range(self.n):
                print(self.X[i,j],end='\t')
            print()
        print(f'最优值： {self.z_min}')

    def HungarianMethod(self):  # 输入效率矩阵
        M = self.A.copy()
        # 每行减去最小元素
        M = (M.T - M.min(axis=1)).T
        # 每列减去最小元素
        M = M - M.min(axis=0)
        keep = False
        while True:  # 一直调整直到产生最优解
            if not keep:
                X = np.zeros_like(self.A)
                del_row = np.zeros(self.n)
                del_col = np.zeros(self.n)
            # 行中有唯一0，删除列
            for row in range(self.n):
                if del_row[row] == 0:
                    yes, col = self.sole0(M[row].copy(), del_col)
                    if yes:
                        X[row, col] = 1
                        del_col[col] = 1
            # 列中有唯一0，删除行
            for col in range(self.n):
                if del_col[col] == 0:
                    yes, row = self.sole0(M[:, col].copy(), del_row)
                    if yes:
                        X[row, col] = 1
                        del_row[row] = 1
            # 检验是否符合最优解
            if X.sum(axis=(0, 1)) == self.n:  # 正好有n个0，找到最优解
                self.X = X
                self.z_min = np.sum(self.A * self.X, axis=(0, 1))
                break  # 唯一出口！
            # '''这一块是处理回路的部分，待修正
            # elif (self.X-X).sum()==0:  # 完全没变化说明出现回路
            #     mg = self.OR(del_row, del_col)
            #     M_ = M.copy()
            #     M_[mg == True]=-1
            #     # 随机采样一个点划去行
            #     del_row[np.unique(np.where(M_==0)[0])]=1
            #     keep = True'''
            else:
                mg = self.OR(del_row, del_col)
                k = M[mg == False].min()
                if k == 0:  # 如果(0)数量<n但是仍然有0没有被划掉就继续
                    keep = True
                else:
                # 定义行势和列势
                    u = np.zeros(self.n)
                    v = np.zeros(self.n)
                    for i in range(self.n):
                        u[i] = 0 if del_row[i] == 1 else k
                    for j in range(self.n):
                        v[j] = -k if del_col[j] == 1 else 0
                    # 根据定理更新效率矩阵使其出现更多0
                    for i in range(self.n):
                        for j in range(self.n):
                            M[i, j] -= (u[i] + v[j])
                    keep = False
def loadData(num=1):
    if num==1:
        A = [
            [3, 8, 2, 10, 3],
            [8, 7, 2, 9, 7],
            [6, 4, 2, 7, 5],
            [8, 4, 2, 3, 5],
            [9, 10, 6, 9, 10],
        ]
    if num == 2:
        A = [
            [3, 8, 2, 10],
            [9, 7, 5, 3],
            [1, 5, 4, 3],
            [4, 5, 7, 9],
        ]
    if num == 3:
        A = [
            [12, 7, 9, 9, 9],
            [8, 9, 7, 7, 7],
            [7, 11, 12, 12, 9],
            [14, 14, 14, 7, 10],
            [4, 10, 10, 7, 9]
        ]
    return A
if __name__ == '__main__':
    A = loadData()
    solver = AssignmentProblem(A)
    solver.HungarianMethod()
    solver.output()

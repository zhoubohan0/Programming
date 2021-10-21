# 产销平衡问题
import numpy as np
def testdata(num=2):
    if num ==1:
        a = [7,4,9]     # 产地A_1~A_m,产量a_1~a_m;
        b = [3,6,5,6]   # 销地B_1~B_n,销量b_1~b_n;
        c = [           # 从A_i运往B_j成本c_{ij};
            [3,11,3,10],
            [1,9,2,8],
            [7,4,10,5],
        ]
    if num ==2:
        a = [7,25,26]
        b = [10,10,20,15,3]
        c = [
            [8,4,1,2,0],
            [6,9,4,7,0],
            [5,3,4,3,0],
            ]
    return a,b,c
def FinfBasicFeasibleSolution(a,b,c):# 伏格尔法
    '''[学习]:np.delete(array,obj,axis)
    array:需要处理的矩阵
    obj:需要处理的位置，比如要删除的第一行或者第一行和第二行
    axis:如果输入为None：array会先按行展开，然后按照obj，删除第obj-1(从0开始)位置的数，返回一个行矩阵。
         如果输入为0：按行删除
         如果输入为1：按列删除'''

    # 初始化
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    result = np.zeros_like(c)
    m,n = len(a),len(b)
    inf = 1e6
    # |最小-次小|

    def getdelta(target,judge):
        # 先清理掉没用的元素
        x = target.copy()
        remove = np.where(judge == 0)[0]
        x = np.delete(x,remove)
        if len(x)==0:
            return
        if len(x)==1:
            return x[0]
        sorted_x = sorted(x)
        return abs(sorted_x[0]-sorted_x[1])

    while sum(a)+sum(b)!=0:
        # 每行|最小运价-次小运价|
        row_delta = np.array([getdelta(c[i],b)if a[i] else -1 for i in range(m)])
        # 每列|最小运价-次小运价|
        col_delta = np.array([getdelta(c[:,j],a)if b[j] else -1 for j in range(n)])
        row_max,row_max_index = row_delta.max(),row_delta.argmax()
        col_max,col_max_index = col_delta.max(),col_delta.argmax()
        if col_max>row_max:
            tmp = c[:,col_max_index]
            tmp[a==0]=inf
            col_min,col_min_index = tmp.min(),tmp.argmin()
            i, j = col_min_index,col_max_index
        else:
            tmp = c[row_max_index]
            tmp[b == 0] = inf
            row_min, row_min_index = tmp.min(),tmp.argmin()
            i, j = row_max_index, row_min_index
        res = min(a[i],b[j])
        result[i, j] = res
        a[i]-=res
        b[j]-=res
    return result
def output(result):
    for row in result:
        for item in row:
            if item == 0:
                print('-',end='\t')
            else:
                print(item,end='\t')
        print()
if __name__ == '__main__':
    a,b,c = testdata()  # 此处输入数据
    result = FinfBasicFeasibleSolution(a,b,c)
    output(result)
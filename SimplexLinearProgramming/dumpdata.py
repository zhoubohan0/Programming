import os
import pickle
M=1e6
# data waiting for pickle
A = [[1, 1, 1, 1, 0, 0, 0],
     [-2, 1, -1, 0, -1, 1, 0],
     [0, 3, 1, 0, 0, 0, 1]]
b = [4, 1, 9]
c = [-3, 0, 1, 0, 0, -M, -M]
base = [3, 5, 6]

if __name__ == '__main__':
    testdir = r"./testdata"
    filename = os.path.join(testdir,f'test{len(os.listdir(testdir))}.pkl')
    with open(filename,'wb')as f:
        pickle.dump(dict(zip(['A','b','c','base'],[A,b,c,base])),f)
    with open(filename,'rb')as f:
        d=pickle.load(f)
        print(d)
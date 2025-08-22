'''
コード4.2 ■ BFGS 公式(H 公式) のコード
'''
import numpy as np
def BFGS_H(H, s, y):
    sy, Hy = s@y, H@y
    return H - (np.outer(Hy,s) + np.outer(s,Hy))/sy + (1+Hy@y/sy)*np.outer(s,s)/sy
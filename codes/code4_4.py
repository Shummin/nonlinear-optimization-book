'''
コード4.4 ■ BFGS 公式(B 公式) のコード
'''
import numpy as np
def BFGS_B(B, s, y):
    sy, Bs = s@y, B@s
    if sy > 0:
        sBs = Bs@s
        B = B - np.outer(Bs,Bs)/sBs + np.outer(y,y)/sy
    return B
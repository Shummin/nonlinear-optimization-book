'''
コード7.3 ■ Powell の修正BFGS 公式のコード
'''
import numpy as np
def BFGS_Powell(B, s, y, omega=0.2):
    sy, Bs = s@y, B@s
    sBs = s@Bs
    psi = 1 if sy >= omega*sBs else (1-omega)*sBs/(sBs-sy)
    z = psi * y + (1 - psi) * Bs
    return B -  np.outer(Bs, Bs) / sBs + np.outer(z, z) / (s@z)
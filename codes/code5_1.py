'''
コード5.1 ■ two loop(アルゴリズム5.1) のコード
'''
import numpy as np
def two_loop(q, S, Y, rho, H0, t):
    a = np.zeros(t)
    for i in range(t): # Step 1
        a[i] = rho[i] * np.dot(S[:, i], q)
        q -= a[i] * Y[:, i]
    r = H0 * q  #Step 2
    for i in range(t-1, -1, -1): # Step 3
        b = rho[i] * np.dot(Y[:, i], r)
        r += S[:, i] * (a[i] - b)
    return r # Step 4
'''
コード4.5 ■ 信頼領域法(アルゴリズム4.1) のコード
'''
import numpy as np
from code4_4 import * # BFGS公式（B公式）のコードを読み込み
def dogleg(obj_f, nab_f, x_k, max_iter=1000, eps=1.e-8, Delta_k=1, xi1=0.25, xi2=0.75, eta1=0.5, eta2=2):
    nab_f_k, B_k = nab_f(x_k), np.identity(len(x_k))
    nab_f_k_old = nab_f_k
    nab_f_k_norm = np.linalg.norm(nab_f_k)
    for k in range(max_iter):
        if nab_f_k_norm <= eps: # 終了判定
            break
        # ドッグレッグ法により部分問題の近似解を求める
        s_N = np.linalg.solve(B_k, -nab_f_k) # （準）ニュートン方向の計算 s_k^N
        s_N_norm = np.linalg.norm(s_N)
        if s_N_norm <= Delta_k:
            s_k = s_N
        else:
            Bnabf = B_k@nab_f_k
            s_C = - (nab_f_k_norm**2 /(nab_f_k@Bnabf)) * nab_f_k # s_k^C
            s_C_norm = np.linalg.norm(s_C)
            if s_C_norm >= Delta_k:
                s_k = - (Delta_k / nab_f_k_norm) *nab_f_k
            else:
                sNsC = s_N@s_C
                sNsC_norm2 = s_N_norm**2 - 2*sNsC +s_C_norm**2
                tau = (sNsC-s_C_norm**2)**2-sNsC_norm2*(s_C_norm**2-Delta_k**2)
                tau = (s_C_norm**2-sNsC+np.sqrt(tau))/sNsC_norm2
                s_k = s_C + tau*(s_N-s_C) 

        rho = (obj_f(x_k) - obj_f(x_k+s_k))/(-nab_f_k@s_k - s_k@(B_k@s_k)/2)
        if rho >= xi1:
            x_k = x_k + s_k # 点列を更新
            nab_f_k = nab_f(x_k)
            nab_f_k_norm = np.linalg.norm(nab_f_k)
            y_k = nab_f_k - nab_f_k_old
            nab_f_k_old = nab_f_k
            B_k = BFGS_B(B_k,s_k,y_k) # 行列を更新
            if rho > xi2:
                Delta_k = eta2*Delta_k # 信頼領域を拡大
        else:
            Delta_k = eta1*Delta_k # 信頼領域を縮小
    print('TrustRegion, iter:', k+1, 'f_val:', obj_f(x_k))
    return x_k
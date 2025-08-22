'''
コード3.3 ■ 非線形共役勾配法のコード（点列のログを残すバージョン）
'''
import numpy as np
from code2_2 import * # Wolfe条件のコードを読み込み
def CG_HS(obj_f, nab_f, x_k, max_iter=1000, eps=1.e-8):
    nab_f_k = nab_f(x_k)
    d_k = - nab_f_k
    seq_x = [x_k]
    for k in range(max_iter):
        alpha = line_Wolfe(obj_f, nab_f, x_k, d_k) # 直線探索
        x_k = x_k + alpha * d_k
        seq_x.append(x_k)
        nab_f_k_old = nab_f_k 
        nab_f_k = nab_f(x_k)
        if np.linalg.norm(nab_f_k) <= eps: #終了判定
            break
        beta = nab_f_k@(nab_f_k-nab_f_k_old)/(d_k@(nab_f_k-nab_f_k_old)) # HS公式を使用
        d_k = - nab_f_k + beta*d_k
    print('CG(HS), iter:', k+1, 'f(x):', obj_f(x_k))
    return np.array(seq_x)
'''
コード5.2 ■ 記憶制限付きBFGS 法(アルゴリズム5.2) のコード
'''
import numpy as np
from code5_1 import * # two loopのコードを読み込み
def LBFGS_H(obj_f, nab_f, x_k, max_iter=1000, eps = 1.e-8, memory=10):
    k,n  = 0, len(x_k)
    S_k = Y_k = np.zeros((n, memory))
    rho_k, m_k = np.zeros(memory), 0
    nab_f_k = nab_f(x_k)
    d_k = -nab_f_k # 初期探索方向は最急降下方向
    for k in range(max_iter):
        if np.linalg.norm(nab_f_k) <= eps: #終了判定
            break
        alpha = line_Wolfe(obj_f, nab_f, x_k, d_k) # 直線探索（Wolfe条件）
        x_k_old = x_k
        x_k = x_k + alpha * d_k # 点列の更新
        nab_f_k_old = nab_f_k
        nab_f_k = nab_f(x_k)
        s_k, y_k= x_k - x_k_old, nab_f_k - nab_f_k_old # s_k, y_kを計算
        # 記憶した行列（ベクトル）S, Y（rho）を更新
        S_k, Y_k, rho_k = np.roll(S_k, 1, axis=1), np.roll(Y_k, 1, axis=1), np.roll(rho_k, 1) 
        S_k[:, 0], Y_k[:, 0], rho_k[0] = s_k, y_k, 1 / (s_k@y_k) 
        H_k0 = S_k[:, 0]@Y_k[:, 0] / Y_k[:, 0]@ Y_k[:, 0] #初期行列を計算
        t_k = min(memory, k+1) 
        d_k = two_loop(-nab_f_k, S_k, Y_k, rho_k, H_k0, t_k) # 探索方向を計算
    print('L-BFGS, iter:', k+1, 'f(x):', obj_f(x_k))
    return x_k
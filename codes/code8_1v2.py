'''
コード8.1 ■ 定数ステップ幅を用いた近接勾配法(アルゴリズム8.1) のコード（点列のログを残すバージョン）
'''
import numpy as np
def ProximalGradient_const(obj_f, nab_f, phi, prox, C, x_k, alpha, max_iter=10000, eps = 1.e-6):
    seq_x = [x_k]
    for k in range(max_iter):
        T_k = prox(x_k - alpha*nab_f(x_k), alpha*C) # 近接勾配法の反復式
        G_k = (x_k - T_k)/alpha # 点列の変化 G を計算
        x_k =  T_k # 点列を更新
        seq_x.append(x_k)
        if np.linalg.norm(alpha*G_k) <= eps: # 終了判定
            break
    return np.array(seq_x)
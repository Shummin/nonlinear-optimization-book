'''
コード8.1 ■ 定数ステップ幅を用いた近接勾配法(アルゴリズム8.1) のコード
'''
import numpy as np
def ProximalGradient_const(obj_f, nab_f, phi, prox, C, x_k, alpha, max_iter=10000, eps = 1.e-6):
    for k in range(max_iter):
        T_k = prox(x_k - alpha*nab_f(x_k), alpha*C) # 近接勾配法の反復式
        G_k = (x_k - T_k)/alpha # 点列の変化 G を計算
        x_k =  T_k # 点列を更新
        if np.linalg.norm(alpha*G_k) <= eps: # 終了判定
            break
    print(f"PGM with const.:反復回数{k+1:d}, 最適値{obj_f(x_k) + phi(x_k):.5e}")
    return x_k
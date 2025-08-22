'''
コード8.3 ■ FISTA(アルゴリズム8.4) のコード
'''
import numpy as np
def FISTA(obj_f, nab_f, phi, prox, C, x_k, alpha, max_iter=10000, eps = 1.e-6):
    y_k = x_k  # 初期点
    tau_k = 1  # 初期の加速パラメータ
    for k in range(max_iter):
        x_old, nab_f_k = x_k, nab_f(y_k) # 前の点の保存と勾配の計算
        x_k = prox(y_k - alpha*nab_f_k, alpha*C) # FISTAの反復式
        tau_new = (1 + np.sqrt(1 + 4 * tau_k**2)) / 2 # 加速パラメータの更新
        y_k = x_k + ((tau_k - 1) / tau_new) * (x_k - x_old) # ykの更新
        if np.linalg.norm(x_k - x_old) <= eps: # 終了判定
            break
        tau_k = tau_new # 加速パラメータを更新
    print(f"FISTA: 反復回数{k+1:d}, 最適値{obj_f(x_k)+phi(x_k):.5e}")
    return x_k
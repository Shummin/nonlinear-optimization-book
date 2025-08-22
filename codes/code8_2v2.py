'''
コード8.2 ■ 近接勾配法(バックトラッキング) のコード（点列のログを残すバージョン）
'''
import numpy as np
def ProximalGradient_backtrack(obj_f, nab_f, phi, prox, C, x_k, alpha=1, tau=0.5, sig=0.1, max_iter=10000, eps = 1.e-6):
    seq_x = [x_k]
    F_k = obj_f(x_k)+phi(x_k)
    for k in range(max_iter):
        nab_f_k = nab_f(x_k) # fの勾配を計算
        T_k = prox(x_k - alpha*nab_f_k, alpha*C)  # 近接勾配法の反復式
        G_k = (x_k - T_k)/alpha # 点列の変化 G を計算
        F_old = F_k # 前の反復の目的関数値を保存
        F_k = obj_f(T_k)+phi(T_k) # 目的関数値を計算 
        while F_k > F_old - sig*alpha*np.linalg.norm(G_k)**2: # 直線探索
            alpha = alpha*tau # ステップ幅を縮小
            T_k = prox(x_k - alpha*nab_f_k, alpha*C) 
            G_k = (x_k - T_k)/alpha
            F_k = obj_f(T_k)+phi(T_k)
        x_k = T_k # 点列を更新
        seq_x.append(x_k)
        if np.linalg.norm(alpha*G_k) <= eps: # 終了判定
            break
    return np.array(seq_x)
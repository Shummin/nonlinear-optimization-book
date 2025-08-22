'''
コード8.6 ■ ニュートン型近接勾配法(アルゴリズム8.5) のコード
'''
import numpy as np
from code8_2 import * # 近接勾配法(バックトラッキング) ProximalGradient_backtrack のコードを読み込み
def Proximal_Newton(obj_f, nab_f, phi, prox, C, x_k, Hess, tau=0.5, sig=0.1, max_iter=1000, eps = 1.e-6):
    B_k, nab_f_k = Hess(x_k), nab_f(x_k) # ヘッセ行列と勾配を計算
    F_k = obj_f(x_k)+phi(x_k) # 目的関数値を計算
    for k in range(max_iter):
        QP = lambda x: obj_f(x_k) + nab_f_k.T@(x-x_k) + (B_k@(x-x_k)).T@(x-x_k)/2 # 部分問題の目的関数を定義
        nab_QP = lambda x: nab_f_k + B_k@(x-x_k) # 部分問題の勾配を定義
        x_k_plus = ProximalGradient_backtrack(QP, nab_QP, phi, prox, C, x_k) # 部分問題を解く(重み付き近接写像の計算)
        d_k = x_k_plus - x_k # 探索方向を計算
        alpha, F_old = 1, F_k
        Delta_k = nab_f_k.T@d_k + phi(x_k_plus) - phi(x_k)
        F_k = obj_f(x_k_plus)+phi(x_k_plus)
        alpha=1
        while F_k > F_old + alpha*sig*Delta_k: # 直線探索
            alpha = alpha*tau
            F_k = obj_f(x_k + alpha*d_k)+phi(x_k + alpha*d_k)
        x_k = x_k + alpha*d_k # 点列を更新
        print(f"Proximal_Newton:反復回数{k+1:d}, 最適値{F_k:.5e}") # 終了判定
        if np.linalg.norm(d_k) <= eps:
            break
        B_k, nab_f_k = Hess(x_k), nab_f(x_k) # ヘッセ行列と勾配法を計算
    return x_k
'''
コード2.1 ■ バックトラッキング法(アルゴリズム2.2) のコード
'''
import numpy as np
def line_Armijo(obj_f, nab_f, x_k, d_k, alpha=1, tau=0.5, sig1=0.0001):
    f_old, f_new = obj_f(x_k), obj_f(x_k+alpha*d_k) 
    nab_fTd = nab_f(x_k)@d_k # 方向微係数を計算　
    while( f_new > f_old+sig1*alpha*nab_fTd ): # Armijo条件のチェック 
        alpha = tau*alpha # ステップ幅を更新
        f_new = obj_f(x_k+alpha*d_k) # 関数値の計算
    return alpha 
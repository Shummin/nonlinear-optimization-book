'''
コード4.1 ■ ニュートン法のコード（点列のログを残す+直線探索バージョン）
'''
import numpy as np
from code2_1 import * # Armijo条件のコードを読み込み
def Newton(obj_f, nab_f, nab2_f, x_k, max_iter=1000, eps=1.e-8):
    seq_x = [x_k]
    for k in range(max_iter):
        d_k = np.linalg.solve(nab2_f(x_k), -nab_f(x_k)) # 探索方向を計算
        alpha = line_Armijo(obj_f, nab_f, x_k, d_k) # 直線探索
        x_k = x_k + alpha*d_k # 点列を更新
        seq_x.append(x_k)
        if np.linalg.norm(nab_f(x_k)) <= eps:
            break
    print('Newton, iter:', k+1, 'f(x):', obj_f(x_k))
    return np.array(seq_x)
'''
コード4.1 ■ ニュートン法のコード
'''
import numpy as np
def Newton(obj_f, nab_f, nab2_f, x_k, max_iter=1000, eps=1.e-8):
    for k in range(max_iter):
        d_k = np.linalg.solve(nab2_f(x_k), -nab_f(x_k)) # 探索方向を計算
        x_k = x_k + d_k # 点列を更新（直線探索なし）
        if np.linalg.norm(nab_f(x_k)) <= eps: # 終了判定
            break
    print('Newton, iter:', k+1, 'f(x):', obj_f(x_k))
    return x_k
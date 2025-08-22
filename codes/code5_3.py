'''
コード5.3 ■ Gauss-Newton 法のコード
'''
import numpy as np
def gauss_newton(res_r, jac_r, x_k, eps=1e-6, max_iter=1000):
    for k in range(max_iter):
        r_k = res_r(x_k) # 残差ベクトルを計算
        J_k = jac_r(x_k) # ヤコビ行列を計算
        B_k, nab_f_k = J_k.T @ J_k, J_k.T @ r_k # J^TJと勾配を計算
        d_k = np.linalg.solve(B_k, -nab_f_k) # 線形方程式を解いて探索方向を計算
        x_k = x_k + d_k
        if  np.linalg.norm(nab_f_k) < eps: # 終了判定
            break
    print('GN, iter:', k+1, '||r(x_k)||:', np.linalg.norm(r_k))
    return x_k
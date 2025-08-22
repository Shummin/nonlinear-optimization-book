'''
コード5.4 ■ Huschens 法(アルゴリズム5.3) のコード
'''
import numpy as np
def Huschens(res_r, jac_r, x_k, eps=1e-6, max_iter=1000):
    n = len(x_k)
    A_k = np.zeros((n,n,))
    J_k_old = J_k =  jac_r(x_k)
    B_k, r_k = J_k.T @ J_k, res_r(x_k)
    r_k_norm = np.linalg.norm(r_k)
    nab_f_k = J_k.T @ r_k
    for k in range(max_iter):
        d_k = np.linalg.solve(B_k, -nab_f_k) # 線形方程式を解いて探索方向を計算
        x_k_old, J_k_old = x_k, J_k  # 一つ前の点とヤコビ行列を保存
        x_k = x_k + d_k # 点列の更新（Step 3）
        r_k, J_k = res_r(x_k), jac_r(x_k)
        r_k_old_norm = r_k_norm # 一つ前の残差ベクトルのノルムを保存
        r_k_norm = np.linalg.norm(r_k) # 残差ベクトルのノルムを計算
        JJ_k, nab_f_k= J_k.T @ J_k, J_k.T @ r_k # J^TJと勾配を計算
        s_k = x_k - x_k_old # s_kを計算
        z_k = JJ_k@s_k + (J_k - J_k_old).T@r_k*(r_k_norm/r_k_old_norm) # z_kを計算
        B_k = JJ_k + r_k_norm*A_k # 近似行列を計算
        Bs = B_k@s_k
        A_k = A_k - np.outer(Bs, Bs) / np.dot(s_k,Bs) \
            + np.outer(z_k, z_k) / np.dot(s_k, z_k)/r_k_norm # A_kを更新
        if np.linalg.norm(nab_f_k) < eps: # 終了判定
            break
    print('Huschens, iter:', k+1, '||r(x_k)||:', np.linalg.norm(r_k))
    return x_k
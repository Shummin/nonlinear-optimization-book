'''
コード7.5 ■ 主双対内点法(アルゴリズム7.8) のコード
'''
import numpy as np
from code7_3 import * # 修正BFGSのコードを読み込み
def PDIP(obj_f, eq_h, nab_f, nab_h, x_0, max_out_iter=20, max_in_iter=20, nu=0.1, beta=0.5,  tau = 0.25, xi = 0.01, omega = 0.2, eps=1e-6):
    x_k, n, l =x_0, len(x_0), len(eq_h(x_0))
    y_k, z_k = np.ones(l), np.ones(n)
    M, M_L, M_U = 1, 2, 2 # 各種パラメータ を定義
    def merit_func(x, nu0, rho0): # メリット関数を定義
        P = obj_f(x) - nu0 * np.sum(np.log(x)) + rho0 * np.sum(np.abs(eq_h(x)))
        return P
    def merit_func_ap(x, dx, nu0, rho0): # メリット関数の一次近似を定義
        f, h, nf, nh = obj_f(x), eq_h(x),  nab_f(x), nab_h(x)
        P_l = f + np.dot(nf, dx) - nu0 * np.sum(np.log(x)+ dx / x) \
                    + rho0 * np.sum(np.abs(h + nh.T @ dx))
        return P_l
    for t in range(max_out_iter): # 外部反復
        h_k, nf_k, nh_k = eq_h(x_k), nab_f(x_k), nab_h(x_k)
        nL_k = nf_k + nh_k @ y_k - z_k
        r_k = np.concatenate([nL_k, h_k, x_k*z_k - nu])
        B_k = np.eye(n)
        for k in range(max_in_iter):
            r_k_norm = np.linalg.norm(r_k) # 残差ノルムを計算
            if r_k_norm <= M * nu: # 内部反復の終了判定
                break
            J_k = np.block([
                [B_k, nh_k, -np.eye(n)],
                [nh_k.T, np.zeros((l, l)), np.zeros((l, n))],
                [np.diag(z_k), np.zeros((n, l)), np.diag(x_k)]
            ]) # 係数行列を定義
            dw_k = np.linalg.solve(J_k, -r_k) # 線形方程式を解いて
            dx_k, dy_k, dz_k = np.split(dw_k, [n, n + l]) # n行目とn+m行目で分割
            # 以下でステップ幅を計算
            rho = max(np.abs(y_k + dy_k)) + 1
            P_k, P_l = merit_func(x_k, nu, rho), merit_func_ap(x_k, dx_k, nu, rho)
            delta_P = P_l - P_k
            alpha_x = min(1, 0.99*min(-x_k[dx_k < 0] / dx_k[dx_k < 0])) if np.any(dx_k < 0) else 1
            for _ in range(10):
                x_new = x_k + alpha_x * dx_k # xを更新
                P_new = merit_func(x_new, nu, rho)
                if P_new <= P_k + xi * alpha_x * delta_P and np.all(x_new > 0):
                    break
                alpha_x *= beta    
            c_Lk = np.minimum(nu/M_L, x_new*z_k)
            c_Uk = np.maximum(M_U*nu, x_new*z_k)
            alpha_zi = np.ones(n)
            for j in range(n):
                if dz_k[j] > 0:
                    alpha_zi[j] = (c_Uk[j]/x_new[j]- z_k[j])/dz_k[j]
                else:
                    alpha_zi[j] = (c_Lk[j]/x_new[j]- z_k[j])/dz_k[j]
            alpha_z = min(min(alpha_zi),1)
            y_new, z_new = y_k + dy_k, z_k + alpha_z*dz_k # y,zを更新(alpha_y = 1)
            h_k, nf_k, nh_k =  eq_h(x_new), nab_f(x_new), nab_h(x_new)
            nL_k_new = nf_k + nh_k @ y_new - z_new
            s_k, v_k  = x_new - x_k, nL_k_new - nL_k
            nL_k = nL_k_new
            x_k, y_k, z_k = x_new, y_new, z_new
            r_k = np.concatenate([nL_k, h_k, x_k*z_k - nu]) # x,y,z の成分を結合して残差ベクトルを計算
            B_k = BFGS_Powell(B_k, s_k, v_k, omega = omega) # ラグランジュ関数のヘッセ行列の近似行列を更新
        if nu < eps or np.linalg.norm(dw_k) < eps: # 外部反復の終了判定
            break
        nu *= tau # nuを縮小
    print(f"目的関数値: {obj_f(x_k):.2f}\nKKT条件のノルム: {np.linalg.norm(r_k):.2e}\n")
    return x_k, y_k, z_k
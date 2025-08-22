'''
コード8.8 ■ 一般化Lasso に特化したADMM(アルゴリズム8.6) のコード
'''
import numpy as np
def generalized_lasso_ADMM(A, b, D, C, x_k, y_k, prox_y, rho_k=10, max_iter=500, eta = 10, zeta1 = 2, zeta2 = 2, eps3 = 1.e-3, eps4 = 1.e-4):
    AtA, Atb, DtD = A.T @ A, A.T @ b, D.T @ D
    sqrt_l_eps, sqrt_n_eps = eps3* np.sqrt(len(y_k)), eps3*np.sqrt(len(x_k))
    mu_k = np.zeros_like(y_k)
    for k in range(max_iter):
        x_k = np.linalg.solve(AtA + rho_k*DtD, Atb + rho_k*D.T @ (y_k - mu_k/rho_k)) # 線形方程式を解いて x を更新
        Dx = D@x_k
        y_k_old = np.copy(y_k)
        y_k = prox_y(Dx + mu_k/rho_k, C / rho_k) # y を更新
        r_k, s_k = Dx - y_k, rho_k*D.T@(y_k - y_k_old) # ペナルティrhoの更新の基準値r, sを計算
        norm_r, norm_s  = np.linalg.norm(r_k), np.linalg.norm(s_k) 
        mu_k = mu_k + rho_k*r_k # muの更新（ラグランジュ乗数の更新）
        eqs1 = sqrt_l_eps + eps4*max(np.linalg.norm(Dx),np.linalg.norm(y_k))
        eqs2 = sqrt_n_eps + eps4*np.linalg.norm(D.T*mu_k)
        if norm_r <= eqs1 and  norm_s <= eqs2:
            break
        if norm_r > eta*norm_s:
            rho_k *= zeta1 # rhoを拡大
        elif norm_s > eta*norm_r:
            rho_k /= zeta2 # rhoを縮小
    print(f"ADMM: 反復回数{k+1:d}, 目的関数値{sum((A@x_k-b)**2)+C*np.linalg.norm(Dx,1):.3e}, ||r_k||={norm_r:.3e}, ||s_k||={norm_s:.3e}, rho_k={rho_k:.3e}")
    return x_k, y_k, mu_k

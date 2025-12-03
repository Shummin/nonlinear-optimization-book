'''
コード7.4 ■ SQP 法(アルゴリズム7.4) のコード
'''
import numpy as np
import cvxpy  as cp
from code7_3 import * # 修正BFGSのコードを読み込み
def SQP(obj_f, ineq_g, eq_h, nab_f, nab_g, nab_h, x_k, max_iter=100, xi = 0.1, eps = 1e-8):
    n, m, l = len(x_k), len(ineq_g(x_k)), len(eq_h(x_k)) # 問題の次元，制約の数を計算
    B_k, lamb, mu =  np.eye(n), np.zeros(m), np.zeros(l) # 初期行列の計算と変数 \lambda, \mu を定義
    def merit_func(x, rho): # l_1型正確なペナルティを定義
        g_values, h_values = ineq_g(x), np.abs(eq_h(x))
        for i in range(m):
            g_values[i] = max(0, g_values[i])
        return obj_f(x)+rho*(np.sum(g_values) + np.sum(h_values))
    def merit_func_l(x, dx, rho): # l_1型正確なペナルティ（１次近似）を定義
        g_values, h_values = ineq_g(x)+ nab_g(x_k).T @ dx, np.abs(eq_h(x)+nab_h(x_k).T@ dx)
        for i in range(m):
            g_values[i] = max(0, g_values[i])
        return obj_f(x)+rho*(np.sum(g_values) + np.sum(h_values))
    # メインループ
    for k in range(max_iter):    
        delta_x = cp.Variable(n) # CVX の変数を定義
        subQP = cp.Problem(cp.Minimize((1/2)*cp.quad_form(delta_x, B_k) + nab_f(x_k).T @ delta_x),
                 [ineq_g(x_k) + nab_g(x_k).T @ delta_x <= 0,
                  eq_h(x_k) + nab_h(x_k).T@ delta_x == 0]) # CVX の問題を定義
        subQP.solve() # CVX 用いて部分問題を解く
        # CVXの結果のdelta_x, 不等式・等式制約に対するラグランジュ乗数を代入
        delta_x_k , lamb, mu = delta_x.value, subQP.constraints[0].dual_value, subQP.constraints[1].dual_value
        if np.linalg.norm(delta_x_k) < eps: # 終了判定
            break
        rho = 2*np.max([np.max(np.abs(lamb)),np.max(np.abs(mu))]) # メリット関数が降下方向になるrhoを計算
        x_k_old = x_k
        alpha, x_k = 1, x_k_old+delta_x_k
        P_old = merit_func(x_k_old,rho)
        delta_P = merit_func_l(x_k_old, delta_x_k, rho) - P_old
        while( merit_func(x_k,rho) > P_old+xi*alpha*delta_P ): # Armijo 条件のチェック 
            alpha = 0.5*alpha # ステップ幅を更新
            x_k = x_k_old+alpha*delta_x_k
            if alpha<1e-5:
                break           
        s_k = x_k - x_k_old # s_k を計算
        y_k = nab_f(x_k) - nab_f(x_k_old)\
            + (nab_g(x_k)-nab_g(x_k_old))@lamb \
            + (nab_h(x_k)-nab_h(x_k_old))@mu # y_k を計算
        B_k = BFGS_Powell(B_k, s_k, y_k) # ヘッセ行列の更新
        print('iter = ', k+1, ', f(x) = ', obj_f(x_k), ', g(x) = ', ineq_g(x_k),  ', h(x) = ', eq_h(x_k), ', \n x = ', x_k, ', lambda = ', lamb, ', mu = ', mu)
    return x_k
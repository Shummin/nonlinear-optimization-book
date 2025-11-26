'''
コード7.1 ■ 外点ペナルティ関数法(アルゴリズム7.2) のコード
'''
import numpy as np
from scipy import optimize
def Exter_penalty_method(obj_f, ineq_g, eq_h, x_k, rho = 1, max_iter=10, eps = 1e-8):
    m, l = len(ineq_g(x_k)), len(eq_h(x_k)) # 不等式制約，等式制約の数を計算
    def pen_val(x): # ペナルティ関数を定義
        g_values, h_values = ineq_g(x), eq_h(x)
        for i in range(m):
            g_values[i] = max(0, g_values[i])
        return (np.sum(g_values**2) + np.sum(h_values**2))
    for k in range(max_iter):
        sub_prob = lambda x: obj_f(x)+rho*pen_val(x) # 部分問題の関数を定義
        result_scipy = optimize.minimize(sub_prob , x_k) # x_k を初期点として、scipy を用いて sub_prob を解く
        x_k = result_scipy.x # scipy の解をx_k に代入する
        print('iter = ', k+1, ', f(x) = ', obj_f(x_k), ', P(x) = ', pen_val(x_k),  ', \n g(x) = ', ineq_g(x_k), ', h(x) = ', eq_h(x_k), ', x = ', x_k)
        if pen_val(x_k) < eps:
            break
        rho *= 5 # ペナルティを拡大
    return x_k
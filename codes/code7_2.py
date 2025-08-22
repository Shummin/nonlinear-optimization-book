'''
コード7.2 ■ 拡張ラグランジュ法(アルゴリズム7.3) のコード
'''
import numpy as np
from scipy import optimize
def ALagrangian_method(obj_f, eq_h, x_k, mu_k,  rho = 1, max_iter=10, eps = 1e-8):
    def pen_val(x): # ペナルティを定義
        h_values =  eq_h(x)
        return np.sum(h_values**2)
    for k in range(max_iter):
        sub_prob = lambda x: obj_f(x)+ eq_h(x)@mu_k+rho*pen_val(x)
        result_scipy = optimize.minimize(sub_prob, x_k) # scipy を用いて解く
        x_k = result_scipy.x # scipy の解を x_k に代入する
        print('iter = ', k+1, ', f(x) = ', obj_f(x_k), ', P(x) = ', pen_val(x_k),  ', \n h(x) = ', eq_h(x_k), ', x = ', x_k, ', mu = ', mu_k)
        if pen_val(x_k) < eps: # 終了判定
            break
        mu_k = mu_k + rho*eq_h(x_k) # ラグランジュ乗数を更新
        rho *= 5 # ペナルティを拡大
    return x_k
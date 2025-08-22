'''
コード2.2 ■ 挟み込みアルゴリズム(アルゴリズム2.3–アルゴリズム2.4) のコード
'''
import numpy as np
def line_Wolfe(obj_f, nab_f, x_k, d_k, alpha=1, alpha_max=5, sig1=0.0001, sig2=0.5):
    alpha_old, f0, nab_fTd = 0, obj_f(x_k), nab_f(x_k)@d_k
    def zoom(alpha_l, alpha_h, x_k, d_k, f_new, f_old): #アルゴリズム 2.3
        for j in range(10):
            tau =  - nab_fTd*alpha_h/(2*(f_new-f0-nab_fTd*alpha_h)) # ２次補間によるステップ幅の更新
            alpha_zoom =  max(0.1, min(tau, 0.9))*alpha_h
            if obj_f(x_k+alpha_zoom*d_k) >f0+sig1*alpha*nab_fTd or f_new >= f_old:
                alpha_l = alpha_zoom
            nab_fTd_new = nab_f(x_k+alpha_zoom*d_k)@d_k
            if sig2*np.abs(nab_fTd) >= np.abs(nab_fTd_new) :
                return alpha_zoom
            if nab_fTd_new*(alpha_h-alpha_l)>=0:
                alpha_h = alpha_l
            alpha_l, f_old = alpha_zoom, f_new
            f_new = obj_f(x_k+alpha_zoom*d_k)
        return alpha_zoom # zoomアルゴリズムによって得られたステップ幅を返す
    f_new = f_old = obj_f(x_k+alpha*d_k)
    for i in range(10): # アルゴリズム 2.4のループ
        if f_new > f0+sig1*alpha*nab_fTd or (i >0 and f_new >= f_old):
            alpha =  zoom(alpha_old, alpha, x_k, d_k, f_new, f_old)
            return alpha
        gd_new = nab_f(x_k+alpha*d_k)@d_k
        if np.abs(nab_fTd) >= sig2*np.abs(gd_new) :
            return alpha 
        if gd_new >= 0:
            alpha = zoom(alpha_old, alpha, x_k, d_k, f_new, f_old)
            return alpha 
        alpha_old = alpha
        alpha = (alpha+alpha_max)/2 # ２分法によるステップ幅の更新
        f_old = f_new
        f_new = obj_f(x_k+alpha*d_k)
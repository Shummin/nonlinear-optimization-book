'''
コード8.4 ■ ソフト閾値関数のコード
'''
import numpy as np
def soft_thresholding(v, C):
    return np.sign(v)*np.maximum(np.abs(v) - C, 0)
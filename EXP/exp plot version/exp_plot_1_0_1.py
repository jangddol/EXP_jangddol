import numpy as np
import uncertainties
from uncertainties import ufloat
from uncertainties.umath import *
import copy
from copy import deepcopy
import matplotlib.pyplot as plt
from exp import *
import time as time
from linfit import *


UF = "<class 'uncertainties.core.Variable'>"
UFAS = "<class 'uncertainties.core.AffineScalarFunc'>"
UFlist = [UF, UFAS]


# 1_0_1 업데이트
    # 일차함수 피팅 함수를 linfit.py로 넘겼다.
    # 이는 exp.py와 exp_plot.py에 동시에 피팅함수를 부르기 위함이다.

def superplt(X, Y, **kwargs):
    
    # control kwargs
    keyword = []
    for v, w in kwargs:
        keyword.append(v)
    if 'line' in keyword:
        pass
    else:
        line = True
    if 'scatter' in keyword:
        pass
    else:
        scatter = True
    if 'lclr' in keyword:
        pass
    else:
        lclr = 'black'
    if 'mclr' in keyword:
        pass
    else:
        mclr = 'black'
    if 'lsize' in keyword:
        pass
    else:
        lsize = 1
    if 'msize' in keyword:
        pass
    else:
        msize = 1
    if 'lshape' in keyword:
        pass
    else:
        lshape = '-'
    if 'mshape' in keyword:
        pass
    else:
        mshape = 'o'
    if 'save' in keyword:
        pass
    else:
        save = True
    
    #control dataX and dataY
    
    
    # draw plot
    plt.figure()
    if line is True:
        plt.plot()
    if scatter is True:
        plt.scatter()
    if save is True:
        pass
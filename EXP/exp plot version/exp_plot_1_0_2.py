import numpy as np
import matplotlib.pyplot as plt

import uncertainties
from uncertainties import ufloat
from uncertainties.umath import *

import os

import datetime
import pytz

import copy
import traceback, threading, time

import inspect

from exp import *
from linfit import *


UF = "<class 'uncertainties.core.Variable'>"
UFAS = "<class 'uncertainties.core.AffineScalarFunc'>"
UFlist = [UF, UFAS]


# 자동으로 좋은 plot을 짜주는 함수를 짜고 싶다.
# line과 scatter를 따로 짜는게 좋을 것 같다.
# animation도 목표로 넣도록 하자.
# line이든 scatter든, 중요한 것들은 다음과 같다.
    # 제목을 넣을 수 있어야 한다.
    # 축 label을 넣어야 한다.
    # legend도 넣어야 한다.
    # plot의 결과물과 내용물을 txt와 png로 전부 저장할 수 있어야 한다.
        # 이는 불러올 수도 있어야 할까...?
    # 저장은 시간값을 반드시 넣어야 한다.

# line인 경우
    # 선의 종류와, 색깔, 굵기를 정해야 한다.
    # 마커의 종류와, 색깔, 크기를 정해야 한다.
    # 에러에 대한 에러바의 종류를 정할 수 있어야 한다.

# scatter의 경우
    # 마커의 종류와, 색깔, 크기를 정해야 한다.
    # 에러에 대한 에러바의 종류를 정할 수 있어야 한다.

# 함수를 귀찮게 짜지 않으려면, matplotlib의 method를 그대로 불러오는 방식을 취해야 할 것이다.

class superplt():
    def write_superplt():
        # plot을 저장하는 함수가 될 예정.
        # 어떻게 짤 지는 미정
        pass
    
    
    def plot(relation):
        # 자동으로 그럴싸한 plot을 짜주는 녀석
        # line에 해당한다.
        
        # 입력 검사.
        temp = Shvar()
        
        if type(relation) == type(temp):
            if relation.tp == 'double group':
                pass
            else:
                pass
        else:
            pass

        plt.plot(X, Y)

        plt.title(Title)

        plt.xlable(Xlabel)
        plt.ylable(Ylabel)

        if relation.slope is None:
            pass
        else:
            if relation.slope.n > 0:
                legend_pos = 'upper left'
            else:
                legend_pos = 'upper right'
        plt.legend()

        plt.show()
        superplt.write_superplt():
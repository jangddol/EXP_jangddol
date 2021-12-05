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
    
    
    def plot(relation, legend = None, fitlegend = None, fitxnum = 1001):
        # plot을 빠르게 생성해주는 녀석
        # 피팅값이 있는 경우 자동으로 만들어준다.
        # 반드시 하나의 relation에 대해서만 그려준다.
        # line에 해당한다.
        
        # 입력 검사.
            # 입력은 반드시 Shvar 클래스여야 한다.
        temp = Shvar()
        
        if type(relation) == type(temp):
            if relation.tp == 'double group':
                pass
            else:
                raise TypeError("The input 'relation' should be a Shvar class and have 'double group' type.")
        else:
            raise TypeError("The input 'relation' should be a Shvar class.")

        # 입력으로 받은 'double group'의 리스트들이 uflist인 경우 mean value만을 받는다.
        X = relation.XYdata[0]
        if relation.grouptp[0] is in ['uflist', 'ufdatalist']:
            X = [x.n for x in X]
        Y = relation.XYdata[1]
        if realtion.grouptp[1] is in ['uflist', 'ufdatalist']:
            Y = [y.n for y in Y]

        plt.plot(X, Y)

        if relation.slope is not None:
            def fitting_func(x):
                y = relation.slope.n * x + relation.intercept.n
                return y
            
            minx = min(X).n
            maxx = max(X).n
            firstx = -0.1*(maxx - minx) + minx
            endx = 0.1*(maxx - minx) + maxx
            fitX = np.linspace(firstx, endx, fitxnum)
            fitY = [fitting_func(x) for x in fitX]
            plt.plot(fitX, fitY)
        else:
            pass


    def plot_comp(relation, legend = None, fitlegend = None, fitxnum = 1001, 
                  Title = None, Xlabel = relation.names[0], Ylabel = relaiton.names[1], pos_leg = 'normal')
        
        plt.figure()

        X = relation.XYdata[0]
        if relation.grouptp[0] is in ['uflist', 'ufdatalist']:
            X = [x.n for x in X]
        Y = relation.XYdata[1]
        if realtion.grouptp[1] is in ['uflist', 'ufdatalist']:
            Y = [y.n for y in Y]

        plt.plot(X, Y)

        if relation.slope is not None:
            def fitting_func(x):
                y = relation.slope.n * x + relation.intercept.n
                return y
            
            minx = min(X).n
            maxx = max(X).n
            firstx = -0.1*(maxx - minx) + minx
            endx = 0.1*(maxx - minx) + maxx
            fitX = np.linspace(firstx, endx, fitxnum)
            fitY = [fitting_func(x) for x in fitX]
            plt.plot(fitX, fitY)
        
            plt.text()
            plt.text()
            plt.text()
        else:
            pass
        
        plt.xlabel()
        plt.ylabel()
        
        plt.title()

        plt.legend()

        plt.show()
        
        plt.savefig('./sin.png')
        write_superplt(relation)

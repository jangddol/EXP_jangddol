import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.rcParams['text.usetex'] = False

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
    
    
    def plot(relation, legend = None, fitlegend = None, fitxnum = 1001, Capsize = 3, Label = None):
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

        # 에러바 때문에 uflist가 있는지 없는지 자세히 알아야 하며, 따라서 Xtype이라는 변수를 만들어서 저장한다.
        if relation.grouptp[0] in ['uflist', 'ufdatalist']:
            Xtype = 'uf'
        else:
            Xtype = 'scalar'
        if relation.grouptp[1] in ['uflist', 'ufdatalist']:
            Ytype = 'uf'
        else:
            Ytype = 'scalar'
        
        

        # X, Y가 각각 uf인 경우/아닌 경우에 맞춰 에러바를 그린다.
        if Xtype == 'uf':
            Xn = [x.n for x in relation.XYdata[0]]
            Xs = [x.s for x in relation.XYdata[0]]
            XERR = Xs
        else:
            Xn = copy.copy(relation.XYdata[0])
            XERR = None

        if Ytype == 'uf':
            Yn = [y.n for y in relation.XYdata[1]]
            Ys = [y.s for y in relation.XYdata[1]]
            YERR = Ys
        else:
            Yn = copy.copy(relation.XYdata[1])
            YERR = None

        # plt.plot(Xn, Yn)
        plt.errorbar(Xn, Yn, xerr=XERR, yerr=YERR, capsize=Capsize, label=Label)
        
        if relation.slope is not None:
            def fitting_func(x):
                y = relation.slope.n * x + relation.intercept.n
                return y
            
            minx = min(Xn)
            maxx = max(Xn)
            firstx = -0.1*(maxx - minx) + minx
            endx = 0.1*(maxx - minx) + maxx
            fitX = np.linspace(firstx, endx, fitxnum)
            fitY = [fitting_func(x) for x in fitX]
            plt.plot(fitX, fitY, label='linear regression')
        else:
            pass


    def plot_comp(relation, legend = None, fitlegend = None, fitxnum = 1001, 
                  Title = None, Xlabel = 'normal', 
                  Ylabel = 'normal', pos_leg = 'best', Capsize = 3, Label = None):
        
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

        # 입력 검사. - pos_leg 는 'right'거나 'left'여야 한다.
        if pos_leg != 'right' and pos_leg != 'left' and pos_leg != 'best':
            raise ValueError("'pos_leg' should be 'right' or 'left'.")
        
        # 에러바 때문에 uflist가 있는지 없는지 자세히 알아야 하며, 따라서 Xtype이라는 변수를 만들어서 저장한다.
        if relation.grouptp[0] in ['uflist', 'ufdatalist']:
            Xtype = 'uf'
        else:
            Xtype = 'scalar'
        if relation.grouptp[1] in ['uflist', 'ufdatalist']:
            Ytype = 'uf'
        else:
            Ytype = 'scalar'

        # X, Y가 각각 uf인 경우/아닌 경우에 맞춰 에러바를 그린다.
        if Xtype == 'uf':
            Xn = [x.n for x in relation.XYdata[0]]
            Xs = [x.s for x in relation.XYdata[0]]
            XERR = Xs
        else:
            Xn = copy.copy(relation.XYdata[0])
            XERR = None

        if Ytype == 'uf':
            Yn = [y.n for y in relation.XYdata[1]]
            Ys = [y.s for y in relation.XYdata[1]]
            YERR = Ys
        else:
            Yn = copy.copy(relation.XYdata[1])
            YERR = None

        fig = plt.figure() ## 캔버스 생성
        fig.set_facecolor('white') ## 캔버스 색상 설정
        ax = fig.add_subplot() ## 프레임(그림 뼈대) 생성

        ax.errorbar(Xn, Yn, xerr=XERR, yerr=YERR, capsize=Capsize, label=Label, marker='o', markersize=5)
        
        if relation.slope is not None:
            def fitting_func(x):
                y = relation.slope.n * x + relation.intercept.n
                return y
            
            minx = min(Xn)
            maxx = max(Xn)
            firstx = -0.1*(maxx - minx) + minx
            endx = 0.1*(maxx - minx) + maxx
            fitX = np.linspace(firstx, endx, fitxnum)
            fitY = [fitting_func(x) for x in fitX]
            ax.plot(fitX, fitY, label='linear regression')

            if pos_leg == 'right' or pos_leg == 'best': ########################## slope 값에 따라 달라지게 고쳐야 함.
                ax.text(minx, 0.9*fitY[-1], r'$y=ax+b$', fontsize=10)
                ax.text(minx, 0.8*fitY[-1], 'a={:.2uP}'.format(relation.slope), fontsize=10)
                ax.text(minx, 0.7*fitY[-1], 'b={:.2uP}'.format(relation.intercept), fontsize=10)
            else:
                ax.text(minx, 0.9*fitY[-1], r'$y=ax+b$', fontsize=10)
                ax.text(minx, 0.8*fitY[-1], 'a={:.2uP}'.format(relation.slope), fontsize=10)
                ax.text(minx, 0.7*fitY[-1], 'b={:.2uP}'.format(relation.intercept), fontsize=10)
        else:
            pass
        
        # Xlabel 작성
        if Xlabel == 'normal':
            ax.set_xlabel(relation.names[0])
        else:
            ax.set_xlabel(Xlabel)
        
        # Ylabel 작성
        if Ylabel == 'normal':
            ax.set_ylabel(relation.names[1])
        else:
            ax.set_xlabel(Ylabel)
        
        # 제목 작성
        ax.set_title(Title)

        # 범례 작성   
        if Label is not None:
            handles, labels = ax.get_legend_handles_labels() ## 범례 처리되는 요소와 해당 라벨
            dict_labels_handles = dict(zip(labels, handles)) ## 라벨을 키로 요소를 밸류로 하는 딕셔너리 생성
            labels = [Label, 'linear regression'] ## 원하는 순서 라벨
            handles = [dict_labels_handles[l] for l in labels] ## 라벨 순서에 맞게 요소 재배치
            if pos_leg == 'best':
                ax.legend(handles, labels, loc='best')
            else:
                if pos_leg == 'right':
                    if relation.slope.n > 0:
                        ax.legend(handles, labels, loc='lower right')
                    else:
                        ax.legend(handles, labels, loc='upper right')
                else:
                    if relation.slope.n > 0:
                        ax.legend(handles, labels, loc='upper left')
                    else:
                        ax.legend(handles, labels, loc='lower left')

        fig.show()
        
        # plt.savefig('./sin.png')
        # write_superplt(relation)

import numpy as np

import uncertainties
from uncertainties import ufloat
from uncertainties.umath import *

import os
import datetime
import pytz

import copy
from copy import deepcopy
import traceback, threading, time

from RIC import RememberInstanceCreationInfo
from RIC import InstanceCreationError

from linfit import *
from uprint import *

"pip install --upgrade uncertainties"

UF = "<class 'uncertainties.core.Variable'>"
UFAS = "<class 'uncertainties.core.AffineScalarFunc'>"
UFlist = [UF, UFAS]


# 2_4_1 버젼
    # linfit_py를 만들었다. 자세한 내용은 exp_plot_1_0_1.py 참조.

def test_func(SELF):
        for frame, line in traceback.walk_stack(None):
            varnames = frame.f_code.co_varnames
            if varnames is ():
                break
            if frame.f_locals[varnames[0]] not in (SELF, SELF.__class__):
                break
                # if the frame is inside a method of this instance,
                # the first argument usually contains either the instance or
                #  its class
                # we want to find the first frame, where this is not the case
        else:
            pass
            # raise InstanceCreationError("No suitable outer frame found.")
        SELF._outer_frame = frame
        SELF.creation_module = frame.f_globals["__name__"]
        SELF.creation_text = traceback.extract_stack(frame, 1)[0][3]
        SELF.creation_name = SELF.creation_text.split("=")[0].strip()
        return SELF.creation_name


class Shvar(RememberInstanceCreationInfo):
    #mean+-SE의 형태를 가진 타입을 ufdata라고 하겠다.
    #data는 단순한 입력을 주로 지칭한다.
    #ufdatalist는 ufdata의 list인 것으로 하겠다.


    def __init__(self):
        super().__init__()
        
        # Shvar Variable이 가지는 기본적인 저장값.
        self.tp = None
        self.origin = None
        self.length = None
        self.tbe = None
        self.mean = None
        self.vrc = None
        self.stdv = None
        self.SE = None
        self.uf = None
        self.iftval = None
        self.tval = None
        self.length = None
        self.dsc = None
        self.scalar = None
        self.scalarlist = None
        self.intercept = None
        self.slope = None
        self.R2 = None
        self.names = None
        self.varname = self.creation_name
        self.writeshvartext()


    def setufdata(self, data, epdf='rec', ec=0, iftval=False, tval=0):
        """
        A method converting A list of number(data) to ufloat(mean+-SE).
        
        Parameter
            data   : A list of numbers. datalist.
            epdf   : Error Probability Distribution Funtion
                Available epdf : 'rec'
            ec     : Error Coefficient ('rec' -> Error bound)
            iftval : Whether True Value needed.
            tval   : True Value.
        
        This method produce following class variable.
            .tp     : ='ufdata'
            .origin : Original data. =data
            .length : =len(data)
            .tbe    : Type-B uncertainty
            .mean   : The average of data.
            .vrc    : The variance of data. if iftval==True, meaningless.
            .stdv   : The standard deviation of data.
            .SE     : The standard Error of data.
            .uf     : =ufloat(self.mean, self.SE)
            .iftval : Whether True Value needed.
            .tval   : True Value.
        
        This method return .uf.
        """
        
        # Error Handling
        try:
            data[0]
        except TypeError:
            raise TypeError("Input 'data' should be subscriptable.")
        if epdf == 'rec': # Error Probability Distribution Function Checking
            if isinstance(ec, (int, float)):
                if ec >= 0:
                    self.tbe = ec/np.sqrt(3)
                else:
                    raise ValueError("Pareameter 'ec' should be positive.")
            else:
                raise TypeError("Parameter 'ec' should be a positive number.")
        else:
            raise TypeError("%s is not available 'epdf'."%(epdf))
        if isinstance(tval, (int, float)):
            pass
        else:
            raise TypeError("Parameter 'tval' should be a number.")
        N = len(data)
        self.tp = 'ufdata'
        self.origin = data
        self.length = N
        self.mean = np.average(data)
        if iftval is True: # If data has true value, then the degree of freedom is 'n'.
            self.vrc = sum([(data[i]-tval)**2 for i in range(N)])/N
        elif iftval is False: # If data has no true value, then the degree of freedom is 'n-1'.
            self.vrc = sum([(data[i]-self.mean)**2 for i in range(N)])/(N-1)
        else:
            raise TypeError("Parameter 'iftval' should be True or False.")
        self.stdv = np.sqrt(self.vrc)
        self.SE = np.sqrt((self.stdv**2)/N + self.tbe**2) # Propagation of Uncertainties.
        self.uf = ufloat(self.mean, self.SE)
        self.writeshvartext()
        return self.uf


    def setufdatalist(self, data, epdf='rec', ec=0, iftval=False, tval=0):
        """
        A method converting A list of list of number(data) to A list of ufloat(mean+-SE).

        Parameter
            data   : A list of list of numbers. list of datalist.A method converting A list of number(data) to ufloat(mean+-SE).
            epdf   : Error Probability Distribution Funtion
                Available epdf : 'rec'
            ec     : Error Coefficient ('rec' -> Error bound)
            iftval : Whether True Value needed.
            tval   : True Value.
        
        This method produce following class variable.
            .tp     : ='ufdatalist'
            .origin : Original data. =data
            .length : The list of len(data)
            .tbe    : Type-B uncertainty
            .mean   : The list of average of data.
            .vrc    : The list of variance of data. if iftval==True, meaningless.
            .stdv   : The list of standard deviation of data.
            .SE     : The list of standard Error of data.
            .uf     : The list of ufloat(self.mean, self.SE)
            .iftval : Whether True Value needed.
            .tval   : The list of True Value.
        
        This method return .uf.
        """
        
        # Error Handling
        try:
            data[0]
        except TypeError:
            raise TypeError("Input 'data' should be subscriptable.")
        N = len(data)
        for X in data:
            try:
                X[0]
            except TypeError:
                raise TypeError("The elements of Input 'data' should be subscriptable.")
        for X in data:
            for y in X:
                if isinstance(y, (int, float)):
                    pass
                else:
                    raise TypeError("Input 'data' should be a list of list of numbers.")
        if epdf == 'rec': # Error Probability Distribution Function Check
            if isinstance(ec, (int, float)):
                if ec >= 0:
                    self.tbe = ec/np.sqrt(3)
                else:
                    raise ValueError("Pareameter 'ec' should be positive.")
            else:
                raise TypeError("Parameter 'ec' should be a positive number.")
        else:
            raise TypeError("%s is not available 'epdf'."%(epdf))
        self.tp = 'ufdatalist'
        self.origin = data
        self.length = [len(data[i]) for i in range(N)]
        self.mean = [np.average(data[i]) for i in range(N)]
        if iftval is True: # If data has True Value, the degree of freedom is 'n'.
            # tval should be a list of numbers which have same length with 'data'.
            try:
                tval[0]
            except TypeError:
                raise TypeError("If iftval is True, parameter 'tval' should subscriptable.")
            if len(tval) != N:
                raise TypeError("If iftval is True, 'data' and 'tval' should have same length.")
            for x in tval:
                if isinstance(x, (int, float)):
                    pass
                else:
                    raise TypeError("If iftval is True, 'tval' should be a list of numbers.")
            self.vrc = [sum([(data[i][j]-tval[i])**2 for j in range(len(data[i]))])/N for i in range(N)]
        elif iftval is False: # If data has no True Value, the degree of freedom is 'n-1'.
            self.vrc = [sum([(data[i][j]-self.mean[i])**2 for j in range(len(data[i]))])/(N-1) for i in range(N)]
        else:
            raise TypeError("Parameter 'iftval' should be True or False.")
        self.stdv = [np.sqrt(self.vrc[i]) for i in range(N)]
        self.SE = [np.sqrt((self.stdv[i]**2)/N + self.tbe**2) for i in range(N)]
        self.uf = [ufloat(self.mean[i], self.SE[i]) for i in range(N)]
        self.writeshvartext()
        return self.uf


    def desc(self, description):
        """
        A method making a description to variable.

        Parameter
            description : A string to be a description of self.
        """
        
        if type(description) is str:
            pass
        else:
            raise TypeError("The description should be a string.")
        self.dsc = description
        self.writeshvartext()
    

    def simplescalar(self, scalar):
        """
        A method making a scalar.

        Parameter
            scalar : A scalar to generate
        
        This method makes the following variables.
            .scalar : = scalar
            .tp     : 'scalar'
        """

        if isinstance(scalar, (int, float)):
            pass
        else:
            raise TypeError("'scalar' should be a number.")
        self.scalar = scalar
        self.tp = 'scalar'
        self.writeshvartext()
        return scalar
    

    def simplescalarlist(self, scalarlist):
        """
        A method making a scalarlist.

        Parameter
            scalarlist : A list of numbers.
        
        This method make the following values.
            .length     : The length of scalarlist 
            .tp         : 'scalarlist'
            .scalarlist : The list of numbers
        
        This method return the scalarlist.
        """
        
        # Error Handling
        try:
            scalarlist[0]
        except Exception:
            raise TypeError("The scalarlist should be subscriptable.")

        for scalar in scalarlist:
            if isinstance(scalar, (int, float)):
                pass
            else:
                raise TypeError("The scalarlist should be a list of numbers.")
                break
        self.length = len(scalarlist)
        self.tp = 'scalarlist'
        self.scalarlist = scalarlist
        self.writeshvartext()
        return scalarlist
    

    def simpleuf(self, *args):
        """
        A method making an ufloat type variable.
        
        Parameter
            If args is one, the argument should be an ufloat value
            If args are two, the arguments should be two numbers.
                These numbers would be the mean and the standard error values.
        
        This method makes the following variables.
            .mean : A mean value.
            .SE : A standard error value.
            .uf   : An ufloat value.
            .tp   : 'uf'
        """
        
        # 오리지널 데이터가 없어도 클래스 변수를 입력하는 메서드
        # 숫자 2개가 들어오면 uf를 반환
        # uf를 입력받으면 평균과 표준편차를 반환
        if len(args) == 1:
            if str(type(args[0])) not in UFlist:
                raise TypeError("The argument should be an ufloat value.")
            else:
                a = args[0]
                self.mean = a.n
                self.SE = a.s
                self.uf = a
                self.tp = 'uf'
        elif len(args) == 2:
            if isinstance(args[0], (int, float)) and isinstance(args[1], (int, float)):
                self.mean = args[0]
                self.SE = args[1]
                self.uf = ufloat(self.mean, self.SE)  
                self.tp = 'uf'
            else:
                raise TypeError("The arguments should be two numbers.")
        else:
            raise TypeError("The number of arguments should be one or two.")
        self.writeshvartext()
        return self.uf
    
    
    def simpleuflist(self, *args):
        """
        A method making an ufloat type variable.
        
        Parameter
            If args is one, the argument should be a list of ufloat value
            If args are two, the arguments should be two lists of numbers.
                These lists would be the list of means and the list of standard errors.
        
        This method makes the following variables.
            .mean : The list of mean value.
            .SE   : The list of standard error value.
            .uf   : The list of ufloat value.
            .tp   : 'uf'
        """

        # Error Handling
        for a in args:
            try:
                a[0]
            except TypeError:
                raise TypeError("The arguments should be one or two lists.")
        if len(args) == 1:
            for x in args[0]:
                if str(type(x)) not in UFlist:
                    raise TypeError("The argument should be a list of ufloats.")
                else:
                    pass
            self.uf = args[0]
            self.mean = [x.n for x in args[0]]
            self.SE = [x.s for x in args[0]]
            self.tp = 'uflist'
        elif len(args) == 2:
            if len(args[0]) != len(args[1]):
                raise TypeError("The two arguments should have same length.")
            for x in args[0]:
                if isinstance(x, (int, float)):
                    pass
                else:
                    raise TypeError("The first argument(mean) should be a list of numbers.")
            for y in args[1]:
                if isinstance(y, (int, float)) and y > 0:
                    pass
                else:
                    raise TypeError("The second argument(SE) should be a list of positive numbers.")
            self.uf = [ufloat(args[0][i], args[1][i]) for i in range(len(args[0]))]
            self.mean = args[0]
            self.SE = args[1]
            self.tp = 'uflist'
        else:
            raise TypeError("The number of arguments should be one or two.")
        self.writeshvartext()
        return self.uf


    def __add__(self, other):
        # 'ufdatalist', 'uflist', 'ufdata', 'uf', 'scalar', 'scalarlist'의 경우를 모두 고려해야한다.
        # 'ufdatalist', 'uflist', 'scalarlist' 를 'list',
        # 'ufdata', 'uf', 'scalar' 를 'uf'로 단순화할 때,
            # 'list' + 'uf', 'uf' + 'list', 'list' + num, 'uf' + 'uf', 'uf' + num
            # 가 가능한 케이스다.
        # Shvar가 아닌 경우는 int, float, ufloat을 고려한다.
        self.creation_name = test_func(self)
        
        result = Shvar()
        mode = 0
        if self.tp == 'ufdatalist' or self.tp == 'uflist':
            mode = 'uflist'
        elif self.tp == 'ufdata' or self.tp == 'uf':
            mode = 'uf'
        elif self.tp == 'scalarlist':
            mode = 'slist'
        elif self.tp == 'scalar':
            mode = 's'
        else:
            print(self.tp)
            raise TypeError("Non-addable type")
        
        if mode == 'uflist':
            if type(other) == type(result): # 'uflist' + Shvar
                if other.tp == 'uf' or other.tp == 'ufdata':
                    result.simpleuflist([x + other.uf for x in self.uf])
                elif other.tp == 'scalar':
                    result.simpleuflist([x + other.scalar for x in self.uf])
                else:
                    raise TypeError("Non-addable type.")
            else: # 'uflist' + ufloat, 'uflist' + float
                result.simpleuflist([x + other for x in self.uf])
        elif mode == 'uf':
            if type(other) == type(result): # 'uf' + Shvar
                if other.tp == 'uf' or other.tp == 'ufdata':
                    result.simpleuf(self.uf + other.uf)
                elif other.tp == 'scalar':
                    result.simpleuf(self.uf + other.scalar)
                elif other.tp == 'uflist' or other.tp == 'ufdatalist' or 'scalarlist':
                    result = other + self
                else:
                    raise TypeError("Non-addable type.")
            else: # 'uf' + ufloat, 'uf' + float
                result.simpleuf(self.uf + other)
        elif mode == 'slist':
            if type(other) == type(result):
                if other.tp == 'uf' or other.tp == 'ufdata':
                    result.simpleuflist([x + other.uf for x in self.scalarlist])
                elif other.tp == 'scalar':
                    result.simplescalarlist([x + other.scalar for x in self.scalarlist])
                else:
                    raise TypeError("Non-addable type.")
        elif mode == 's':
            if type(other) == type(result):
                result = other + self
            elif str(type(other)) in UFlist:
                result.simpleuf(self.scalar + other)
            else:
                result.simplescalar(self.scalar + other)
        else:
            raise TypeError("Something is Wrong. fxxk.")
        result.varname = self.creation_name
        result.writeshvartext()
        return result


    def __radd__(self, other):
        result = self + other
        result.writeshvartext()
        return result
    

    def __pos__(self):
        self.creation_name = test_func(self)

        List = ['uf', 'ufdata', 'uflist', 'ufdatalist', 'scalarlist', 'scalar']
        if self.tp in List:
            self.varname = self.creation_name
            result.writeshvartext()
            return self
        else:
            raise TypeError("Non-plus-attachable.")


    def __neg__(self):
        self.creation_name = test_func(self)
        
        List = ['uf', 'ufdata', 'scalar', 'uflist', 'ufdatalist', 'scalarlist']
        if self.tp in List:
            pass
        else:
            raise TypeError("Non-minus-attachable.")
        
        result = Shvar()
        if self.tp == 'uf' or self.tp == 'ufdata':
            result.simpleuf(-self.uf)
        elif self.tp == 'scalar':
            result.simplescalar(-self.scalar)
        elif self.tp == 'uflist' or self.tp == 'ufdatalist':
            result.simpleuflist([-x for x in self.uf])
        elif self.tp == 'scalarlist':
            result.simplescalarlist([-x for x in self.scalarlist])
        else:
            raise ValueError("Something is Wrong, fxxk.")
        
        result.varname = self.creation_name
        result.writeshvartext()
        return result
    

    def __sub__(self, other):
        self.creation_name = test_func(self)

        result = self + other.__neg__()
        result.varname = self.creation_name
        result.writeshvartext()
        return result
    

    def __rsub__(self, other):
        result = -(self - other)
        result.writeshvartext()
        return result
    

    def __mul__(self, other):
        self.creation_name = test_func(self)

        result = Shvar()
        mode = 0
        if self.tp == 'uf' or self.tp == 'ufdata':
            mode = 'uf'
        elif self.tp == 'scalar':
            mode = 's'
        elif self.tp == 'uflist' or self.tp == 'ufdatalist':
            mode = 'uflist'
        elif self.tp == 'scalarlist':
            mode = 'slist'
        else:
            raise TypeError("Non-multiplable type.")
        
        if mode == 'uf':
            if type(other) == type(result):
                if other.tp == 'uf' or other.tp == 'ufdata':
                    result.simpleuf(self.uf * other.uf)
                elif other.tp == 'scalar':
                    result.simpleuf(self.uf * other.scalar)
                elif other.tp == 'uflist' or other.tp == 'ufdatalist':
                    result.simpleuflist([self.uf * x for x in other.uf])
                elif other.tp == 'scalarlist':
                    result.simpleuflist([self.uf * x for x in other.scalarlist])
                else:
                    raise TypeError("Non-multiplable type.")
            else:
                result.simpleuf(self.uf * other)
        elif mode == 'uflist':
            if type(other) == type(result):
                if other.tp == 'uf' or other.tp == 'ufdata':
                    result.simpleuflist([x * other.uf for x in self.uf])
                elif other.tp == 'scalar':
                    result.simpleuflist([x * other.scalar for x in self.uf])
                else:
                    raise TypeError("Non-multiplable type.")
            else:
                result.simpleuflist([x * other for x in self.uf])
        elif mode == 's':
            if type(other) == type(result):
                if other.tp == 'uf' or other.tp == 'ufdata':
                    result.simpleuf(self.scalar * other.uf)
                elif other.tp == 'scalar':
                    result.simplescalar(self.scalar * other.scalar)
                elif other.tp == 'uflist' or other.tp == 'ufdatalist':
                    result.simpleuflist([self.scalar * x for x in other.uf])
                elif other.tp == 'scalarlist':
                    result.simplescalarlist([self.scalar * x for x in other.scalarlist])
                else:
                    raise TypeError("Non-multiplable type.")
            elif str(type(other)) in UFlist:
                result.simpleuf(self.scalar * other)
            else:
                result.simplescalar(self.scalar * other)
        elif mode == 'slist':
            if type(other) == type(result):
                if other.tp == 'uf' or other.tp == 'ufdata':
                    result.simpleuflist([x * other.uf for x in self.scalarlist])
                elif other.tp == 'scalar':
                    result.simplescalarlist([x * other.scalar for x in self.scalarlist])
                else:
                    raise TypeError("Non-multiplable type.")
            elif str(type(other)) in UFlist:
                result.simpleuflist([x * other for x in self.scalarlist])
            else:
                result.simplescalarlist([x * other for x in self.scalarlist])
        else:
            raise ValueError("Something is Wrong. fxxk.")
        
        result.varname = self.creation_name
        result.writeshvartext()
        return result


    def reciprocal(self):
        self.creation_name = test_func(self)
        
        List = ['uf', 'ufdata', 'scalar', 'uflist', 'ufdatalist', 'scalarlist']
        if self.tp in List:
            pass
        else:
            raise TypeError("Non-calculable type(reciprocal).")
        
        result = Shvar()
        if self.tp == 'uf' or self.tp == 'ufdata':
            result.simpleuf(1/self.uf)
        elif self.tp == 'scalar':
            result.simplescalar(1/self.scalar)
        elif self.tp == 'uflist' or self.tp == 'ufdatalist':
            result.simpleuflist([1/x for x in self.uf])
        elif self.tp == 'scalarlist':
            result.simplescalarlist([1/x for x in self.scalarlist])
        else:
            raise ValueError("Something is Wrong, fxxk.")

        result.varname = self.creation_name
        result.writeshvartext()
        return result


    def __truediv__(self, other):
        self.creation_name = test_func(self)
        
        result = Shvar()
        if type(other) == type(result):
            result = self * other.reciprocal()
        else:
            result = self * (1/other)
        result.varname = self.creation_name
        result.writeshvartext()
        return result


    def __rtruediv__(self, other):
        self.creation_name = test_func(self)
        
        result = Shvar()
        if type(other) == type(result):
            result = other / self
        else:
            result = other * self.reciprocal()
        result.varname = self.creation_name
        result.writeshvartext()
        return result
    

    def linear_rel(self, X, Y, fit=True):
        """
        A method group two lists with linear fitting.
        If both X and Y have uncertainties, then this method use the method of York(1966).

        Parameter
            X   : A list of numbers or a list of ufloats. The data of x-axis.
            Y   : A list of numbers or a list of ufloats. The data of y-axis.
            fit : If True, this method calculate the fitting graph
        
        This method make the following value.
            .tp : 'double group'
            .names : [name of X, name of Y]

        If fit==True, this method make the following values.
            .slope     : The slope of fitting graph
            .intercept : The intercept of y-axis of fitting graph
            .R2        : The coefficient of Determination of fitting graph
        """

        # Error Handling
        if type(X) == type(self) and type(Y) == type(self):
            pass
        else:
            raise TypeError("Input X and Y should be objects of class Shvar.")
        if fit != True and fit != False:
            raise TypeError("Parameter fit should be True or False.")
        
        if fit:
            if (X.tp == 'uflist' or X.tp == 'ufdatalist') and (Y.tp == 'uflist' or Y.tp == 'ufdatalist'):
                temp = linear_fitting_York(X.uf, Y.uf, initguess='yerr')
                self.slope = ufloat(temp[0], temp[4])
                self.intercept = ufloat(temp[1], temp[5])
                self.R2 = temp[6]
            elif X.tp == 'scalarlist' and (Y.tp == 'uflist' or Y.tp == 'ufdatalist'):
                temp = linear_fitting_Yerr(X.scalarlist, Y.mean, Y.SE)
                self.slope = temp[0]
                self.intercept = temp[1]
                self.R2 = temp[2]
            elif (X.tp == 'uflist' or X.tp == 'ufdatalist') and Y.tp == 'scalarlist':
                temp = linear_fitting_Yerr(Y.scalarlist, X.mean, X.SE)
                self.slope = 1/temp[0]
                self.intercept = -temp[1]/temp[0]
                self.R2 = temp[2]
            elif X.tp == 'scalarlist' and Y.tp == 'scalarlist':
                temp = linear_fitting(X.scalarlist, Y.scalarlist)
                self.slope = temp[0]
                self.intercept = temp[1]
                self.R2 = temp[2]
            else:
                raise TypeError("Type value of Input X and Y should be 'list' or 'scalarlist'.")
        self.tp = 'double group'
        self.names = [X.varname, Y.varname]
        self.writeshvartext()
    

    def printfull(self):
        """
        This Method print the information of Shvar Variable.
        """
        print("varname :", self.varname)
        print("tp :", self.tp)
        if self.tp == 'uf':
            print("Mean Value :", self.mean)
            print("Standard Error :", self.SE)
            printp(self.uf, txt="uf :")
        elif self.tp == 'ufdata':
            print("Type B Error :", self.tbe)
            print("Mean Value :", self.mean)
            if self.iftval is False:
                print("Variance :", self.vrc)
                print("Standard Deviation :", self.stdv)
            print("Standard Error :", self.SE)
            printp(self.uf, txt="uf :")
            if self.iftval:
                print("True Value :", self.tval)
            print("Origin data :", self.origin)
        elif self.tp == 'uflist':
            print("Length :", self.length)
            print("Mean Value :", self.mean)
            print("Standard Error :", self.SE)
            print("uf :", self.uf)
        elif self.tp == 'ufdatalist':
            print("Length :", self.length)
            print("Type B Error :", self.tbe)
            print("Mean Value :", self.mean)
            if self.iftval is False:
                print("Variance :", self.vrc)
                print("Standard Deviation :", self.stdv)
            print("Standard Error :", self.SE)
            print("uf :", self.uf)
            if self.iftval:
                print("True Value :", self.tval)
            print("Origin data :", self.origin)
        elif self.tp == 'scalar':
            print("scalar :", self.scalar)
        elif self.tp == 'scalarlist':
            print("Length :", self.length)
            print("The list of scalar:", self.scalarlist)
        elif self.tp == 'double group':
            print("names :", self.names)
            if self.slope is not None:
                printp(self.slope, txt="slope :")
                printp(self.intercept, txt="intercept :")
                print("Coefficient of Detemination :", '{:.3f}'.format(self.R2))
        elif self.tp is None:
            pass
        else:
            raise TypeError("Hey, the developer need to fix this method!")
    

    def writeshvartext(self):
        # Shvar 타입의 variable 정보를 저장할 텍스트파일을 생성
        

        # 만약 varname이 존재하지 않는다면 쓰지 않는것으로 결정한다.
            # 쓰지 않는 것을 알린다.
        
        # 오늘의 날짜를 불러온다.
        # 오늘의 날짜에 해당하는 폴더가 있는지 검사한다.
            # 만약에 있다면 그곳에 저장한다.
            # 만약에 없다면 폴더를 생성하고 그곳에 저장한다.
        # 경로에서 파일 이름들을 불러오고, varname으로 시작하는 파일들을 검색한다.
            # 있다면 있는 것이다.
        # using now() to get current time  
        current_time = datetime.datetime.now(pytz.timezone('Asia/Seoul')) # 현재 서울 시간

        Year = str(current_time.year)
        Month = str(current_time.month)
        if len(Month) == 1:
            Month = '0' + Month
        Day = str(current_time.day)
        if len(Day) == 1:
            Day = '0' + Day
        date = Year + '-' + Month + '-' + Day # 오늘 날짜

        Hour = str(current_time.hour)
        if len(Hour) == 1:
            Hour = '0' + Hour
        Minute = str(current_time.minute)
        if len(Minute) == 1:
            Minute = '0' + Minute
        TitleMinute = str(30 * int(current_time.minute / 30))
        if len(TitleMinute) == 1:
            TitleMinute = '0' + TitleMinute
        Second = str(current_time.second)
        if len(Second) == 1:
            Second = '0' + Second
        time = Hour + ':' + Minute + ':' + Second # 그냥 현재 시간
        Titletime = Hour + ':' + TitleMinute + ':' + '00' # 30분 단위의 현재시간
        
        # 검색하는 파트
            # return True if path is an existing directory
        # 경로변경
        os.chdir('/content/drive/MyDrive/Coding/EXP')
        if os.path.isdir('/content/drive/MyDrive/Coding/EXP/var/' + date):
            pass
        else:
            os.mkdir('/content/drive/MyDrive/Coding/EXP/var/' + date)

        # 텍스트 내용을 정한다.
            # 텍스트 내용은 class variable로 한다.
            # 마지막에 작성 시간을 추가한다.
        data = [None for i in range(20)]
        data[0] = 'tp, ' + str(self.tp)
        data[1] = 'origin, ' + str(self.origin)
        data[2] = 'length, ' + str(self.length)
        data[3] = 'tbe, ' + str(self.tbe)
        data[4] = 'mean, ' + str(self.mean)
        data[5] = 'vrc, ' + str(self.vrc)
        data[6] = 'stdv, ' + str(self.stdv)
        data[7] = 'SE, ' + str(self.SE)
        data[8] = 'uf, ' + str(self.uf)
        data[9] = 'iftval, ' + str(self.iftval)
        data[10] = 'tval, ' + str(self.tval)
        data[11] = 'length, ' + str(self.length)
        data[12] = 'dsc, ' + str(self.dsc)
        data[13] = 'scalar, ' + str(self.scalar)
        data[14] = 'scalarlist, ' + str(self.scalarlist)
        data[15] = 'intercept, ' + str(self.intercept)
        data[16] = 'slope, ' + str(self.slope)
        data[17] = 'R2, ' + str(self.R2)
        data[18] = 'names, ' + str(self.names)
        data[19] = 'varname, ' + str(self.varname)

        # 텍스트 제목을 정한다.
            # 텍스트의 제목은 varname_시간 으로 결정한다.
            # 시간은 30분 간격이다.
            # 시간은 날짜를 포함한다.
    
        title = str(self.varname) + '_' + Titletime

        # 저장 경로 폴더에 해당 제목의 파일이 있는지 검사한다.
            # 만약에 있다면 덮어쓰기.
            # 만약에 없다면 생성하기.
        f = open('/content/drive/MyDrive/Coding/EXP/var/'+date+'/'+title+'.txt', 'w')
        for x in data:
            f.write(x + '\n')
        f.write('Last fixed time : ' + date + ' ' + time)
        f.close()
    

    def readshvartext(varname, Date='today', Time='latest', imp=False):
        # Shvar object의 Class variable 값을 불러온다.
        # 원하면 로딩한다.

        # 시간을 불러온다.
        current_time = datetime.datetime.now(pytz.timezone('Asia/Seoul')) # 현재 서울 시간

        Year = str(current_time.year)
        Month = str(current_time.month)
        if len(Month) == 1:
            Month = '0' + Month
        Day = str(current_time.day)
        if len(Day) == 1:
            Day = '0' + Day
        date = Year + '-' + Month + '-' + Day # 오늘 날짜

        Hour = str(current_time.hour)
        if len(Hour) == 1:
            Hour = '0' + Hour
        Minute = str(current_time.minute)
        if len(Minute) == 1:
            Minute = '0' + Minute
        TitleMinute = str(30 * int(current_time.minute / 30))
        if len(TitleMinute) == 1:
            TitleMinute = '0' + TitleMinute
        Second = str(current_time.second)
        if len(Second) == 1:
            Second = '0' + Second
        time = Hour + ':' + Minute + ':' + Second # 그냥 현재 시간 (파일을 찾는 데에는 사용하지 않는다.)
        Titletime = Hour + ':' + TitleMinute + ':' + '00' # 30분 단위의 현재시간

        # 원하는 날짜를 입력할 수 있게 한다.
            # 기본값은 오늘
            # 형식은 '년월일'
            # 년은 4자리, 월과 일은 2자리
            # 총 8자리
        if Date == 'today': # 오늘일 경우의 디렉토리이름
            Directorytitle = date
        elif Date == 'yesterday':
            yesterday = datetime.datetime.now(pytz.timezone('Asia/Seoul')) - datetime.timedelta(days=1)
            Year = str(yesterday.year)
            Month = str(yesterday.month)
            if len(Month) == 1:
                Month = '0' + Month
            Day = str(yesterday.day)
            if len(Day) == 1:
                Day = '0' + Day
            Directorytitle = Year + '-' + Month + '-' + Day # 어제 날짜
        else:
            if isinstnace(Date, int):
                if len(str(Date)) == 8:
                    Year = int(str(Date)[:4])
                    Month = int(str(Date)[4:6])
                    if Month >= 1 and Month <= 12:
                        pass
                    else:
                        raise TypeError("Month should be in 01 ~ 12.")
                    Day = int(str(Date)[6:])
                    
                    if Month == 2:
                        if Year % 4 == 0: # 윤년
                            if Year % 100 == 0: # 윤년 아님
                                if Year % 400 == 0: # 윤년
                                    if Day >= 1 and Day <= 29:
                                        pass
                                    else:
                                        raise ValueError("Day should be in 01 ~ 29.")
                                else:
                                    if Day >= 1 and Day <= 28:
                                        pass
                                    else:
                                        raise ValueError("Day should be in 01 ~ 28.")
                            else:
                                if Day >= 1 and Day <= 29:
                                    pass
                                else:
                                    raise ValueError("Day should be in 01 ~ 29.")
                        else:
                            if Day >= 1 and Day <= 28:
                                pass
                            else:
                                raise ValueError("Day should be in 01 ~ 28.")
                    elif Month in [1, 3, 5, 7, 8, 10, 12]: # 날짜가 31까지만 있음
                        if Day >= 1 and Day <= 31:
                            pass
                        else:
                            raise ValueError("Day should be in 01 ~ 31.")
                    elif Month in [4, 6, 9, 11]: # 날짜가 30까지만 있음
                        if Day >= 1 and Day <= 30:
                            pass
                        else:
                            raise ValueError("Day should be in 01 ~ 30.")
                    else:
                        raise ValueError("Something is Wrong at Month value.")
                    Year = str(Year)
                    Month = str(Month)
                    if len(Month) == 1:
                        Month = '0' + Month
                    Day = str(Day)
                    if len(Day) == 1:
                        Day = '0' + Day
                    Directorytitle = Year + '-' + Month + '-' + Day # 지정날짜
                else:
                    raise TypeError("Date should be 'today', 'yesterday', or 8 digits numbers.")
            else:
                raise TypeError("Date should be 'today', 'yesterday', or 8 digits numbers.")
        
        # 원하는 시간을 입력할 수 있게 한다.
            # 기본값은 최신값
            # 형식은 '시간분'
            # 시간은 24시간기준이며, 시간, 분 전부 2자리다.
            # 총 4자리
        if Time == 'latest':
            pass
        else:
            if isinstance(Time, int) and len(str(Time)) == 4:
                Hour = int(str(Time)[:2])
                TitleMinute = int(int(str(Time)[2:])/30)*30

                if Hour >=0 and Hour < 24:
                    pass
                else:
                    raise ValueError("Hour must be in 0 ~ 23")
                
                if TitleMinute ==0 or TitleMinute == 30:
                    pass
                else:
                    raise ValueError("Minute must be in 0 ~ 59")
                
                Hour = str(Hour)
                if len(Hour) == 1:
                    Hour = '0' + Hour
                TitleMinute = str(TitleMinute)
                if len(TitleMinute) == 1:
                    TitleMinute = '0' + TitleMinute

                Titletime = Hour + ':' + TitleMinute + ':00'
            else:
                raise TypeError("Time should be 'latest' or 4 digits numbers.")
        
        Filename = varname + '_' + Titletime + '.txt'
        # 선택된 파일로부터 Shvar object의 값을 불러온다.
            # 단순히 print
        exist = 0
        if os.path.isdir('/content/drive/MyDrive/Coding/EXP/var/' + Directorytitle):
            FileList = os.listdir('/content/drive/MyDrive/Coding/EXP/var/' + Directorytitle)
            if Filename in FileList:
                exist = 1
            else:
                print("There is no " + "'" + Filename + "'.")
        else:
            print("There is no " + "'" + Directorytitle + "'.")
        


        if exist == 1:
            f = open('/content/drive/MyDrive/Coding/EXP/var/'+Directorytitle+'/'+Filename, 'r')
            lines = f.readlines()
            print(' ')
            for x in lines:
                print(x)
            print(' ')

            if imp is True: ##############################################################여기 하다 말았음
                # 텍스트의 내용을 앞부분과 뒷부분으로 찢은 뒤,
                # 앞의 내용에 맞춰서 result라는 새로운 shvar object에 삽입
                # 최종적으로 varname에 맞춰서 result에 동기화
                pass
        

        # import의 여부는 따로 선택할 수 있게 한다.
            # 기본값은 x
            # import를 하는 경우
                # 덮어쓰기
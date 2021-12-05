import numpy as np
import uncertainties
from uncertainties import ufloat
from uncertainties.umath import *
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


# 2_5_0 버젼
    # RIC의 코드를 모든  operator 에다가 꼴아박아서 일단은 varname 을 뽑아내도록 하긴 했다.
    # 코드가 ㅈㄴ 더럽다
    # 맘에 안든다
    # 어떻게 할 수 없는건가
    # 아무쪼록 바꾸고 싶다
    # 당분간은 디버깅의 기간이 될 것 같다.
    # __add__에서 scalar + scalar 에서 터지는 것을 확인하여 디버깅 하였다.
        # __mul__에서도 터질 것 같다.
        


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
        self.varname = self.creation_name


    def copy(self):
        # 복사용 함수
        return copy.copy(self)

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
        return 0
    
    
    def simplelist(self, *args):
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
    

    def __add__(self, other):
        # 'ufdatalist', 'uflist', 'ufdata', 'uf', 'scalar', 'scalarlist'의 경우를 모두 고려해야한다.
        # 'ufdatalist', 'uflist', 'scalarlist' 를 'list',
        # 'ufdata', 'uf', 'scalar' 를 'uf'로 단순화할 때,
            # 'list' + 'uf', 'uf' + 'list', 'list' + num, 'uf' + 'uf', 'uf' + num
            # 가 가능한 케이스다.
        # Shvar가 아닌 경우는 int, float, ufloat을 고려한다.
        for frame, line in traceback.walk_stack(None):
            varnames = frame.f_code.co_varnames
            if varnames is ():
                break
            if frame.f_locals[varnames[0]] not in (self, self.__class__):
                break
                # if the frame is inside a method of this instance,
                # the first argument usually contains either the instance or
                #  its class
                # we want to find the first frame, where this is not the case
        else:
            raise InstanceCreationError("No suitable outer frame found.")
        self._outer_frame = frame
        self.creation_module = frame.f_globals["__name__"]
        self.creation_file, self.creation_line, self.creation_function, \
            self.creation_text = \
            traceback.extract_stack(frame, 1)[0]
        self.creation_name = self.creation_text.split("=")[0].strip()
        
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
            raise TypeError("Non-addable type")
        
        if mode == 'uflist':
            if type(other) == type( ): # 'uflist' + Shvar
                if other.tp == 'uf' or other.tp == 'ufdata':
                    result.simplelist([x + other.uf for x in self.uf])
                elif other.tp == 'scalar':
                    result.simplelist([x + other.scalar for x in self.uf])
                else:
                    raise TypeError("Non-addable type.")
            else: # 'uflist' + ufloat, 'uflist' + float
                result.simplelist([x + other for x in self.uf])
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
                    result.simplelist([x + other.uf for x in self.scalarlist])
                elif other.tp == 'scalar':
                    result.simplescalarlist([x + other.scalar for x in self.scalarlist])
                else:
                    raise TypeError("Non-addable type.")
        elif mode == 's':
            if type(other) == type(result):
                result = other + self.scalar
            elif str(type(other)) in UFlist:
                result.simpleuf(self.scalar + other)
            else:
                result.simplescalar(self.scalar + other)
        else:
            raise TypeError("Something is Wrong. fxxk.")
        result.varname = self.creation_name
        return result.copy()


    def __radd__(self, other):
        return self + other
    

    def __pos__(self):
        for frame, line in traceback.walk_stack(None):
            varnames = frame.f_code.co_varnames
            if varnames is ():
                break
            if frame.f_locals[varnames[0]] not in (self, self.__class__):
                break
                # if the frame is inside a method of this instance,
                # the first argument usually contains either the instance or
                #  its class
                # we want to find the first frame, where this is not the case
        else:
            raise InstanceCreationError("No suitable outer frame found.")
        self._outer_frame = frame
        self.creation_module = frame.f_globals["__name__"]
        self.creation_file, self.creation_line, self.creation_function, \
            self.creation_text = \
            traceback.extract_stack(frame, 1)[0]
        self.creation_name = self.creation_text.split("=")[0].strip()
        List = ['uf', 'ufdata', 'uflist', 'ufdatalist', 'scalarlist', 'scalar']
        if self.tp in List:
            self.varname = self.creation_name
            return self
        else:
            raise TypeError("Non-plus-attachable.")


    def __neg__(self):
        for frame, line in traceback.walk_stack(None):
            varnames = frame.f_code.co_varnames
            if varnames is ():
                break
            if frame.f_locals[varnames[0]] not in (self, self.__class__):
                break
                # if the frame is inside a method of this instance,
                # the first argument usually contains either the instance or
                #  its class
                # we want to find the first frame, where this is not the case
        else:
            raise InstanceCreationError("No suitable outer frame found.")
        self._outer_frame = frame
        self.creation_module = frame.f_globals["__name__"]
        self.creation_file, self.creation_line, self.creation_function, \
            self.creation_text = \
            traceback.extract_stack(frame, 1)[0]
        self.creation_name = self.creation_text.split("=")[0].strip()
        
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
            result.simplelist([-x for x in self.uf])
        elif self.tp == 'scalarlist':
            result.scalarlist = [-x for x in self.scalarlist]
        else:
            raise ValueError("Something is Wrong, fxxk.")
        
        result.varname = self.creation_name
        return result
    

    def __sub__(self, other):
        for frame, line in traceback.walk_stack(None):
            varnames = frame.f_code.co_varnames
            if varnames is ():
                break
            if frame.f_locals[varnames[0]] not in (self, self.__class__):
                break
                # if the frame is inside a method of this instance,
                # the first argument usually contains either the instance or
                #  its class
                # we want to find the first frame, where this is not the case
        else:
            raise InstanceCreationError("No suitable outer frame found.")
        self._outer_frame = frame
        self.creation_module = frame.f_globals["__name__"]
        self.creation_file, self.creation_line, self.creation_function, \
            self.creation_text = \
            traceback.extract_stack(frame, 1)[0]
        self.creation_name = self.creation_text.split("=")[0].strip()
        result = self + __neg__(other)
        result.varname = self.creation_name
        return result
    

    def __rsub__(self, other):
        return - (self - other)
    

    def __mul__(self, other):
        for frame, line in traceback.walk_stack(None):
            varnames = frame.f_code.co_varnames
            if varnames is ():
                break
            if frame.f_locals[varnames[0]] not in (self, self.__class__):
                break
                # if the frame is inside a method of this instance,
                # the first argument usually contains either the instance or
                #  its class
                # we want to find the first frame, where this is not the case
        else:
            raise InstanceCreationError("No suitable outer frame found.")
        self._outer_frame = frame
        self.creation_module = frame.f_globals["__name__"]
        self.creation_file, self.creation_line, self.creation_function, \
            self.creation_text = \
            traceback.extract_stack(frame, 1)[0]
        self.creation_name = self.creation_text.split("=")[0].strip()
        
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
                    result.simpleuf(self.uf * other)
                elif other.tp == 'uflist' or other.tp == 'ufdatalist':
                    result.simplelist([self.uf * x for x in other.uf])
                elif other.tp == 'scalarlist':
                    result.simplelist([self.uf * x for x in other.scalarlist])
                else:
                    raise TypeError("Non-multiplable type.")
            else:
                result.simpleuf(self.uf * other)
        elif mode == 'uflist':
            if type(other) == type(result):
                if other.tp == 'uf' or other.tp == 'ufdata':
                    result.simplelist([x * other.uf for x in self.uf])
                elif other.tp == 'scalar':
                    result.simplelist([x * other.scalar for x in self.uf])
                else:
                    raise TypeError("Non-multiplable type.")
            else:
                result.simplelist([x * other for x in self.uf])
        elif mode == 's':
            if type(other) == type(result):
                if other.tp == 'uf' or other.tp == 'ufdata':
                    result.simpleuf(self.scalar * other.uf)
                elif other.tp == 'scalar':
                    result.simplescalar(self.scalar * other.scalar)
                elif other.tp == 'uflist' or other.tp == 'ufdatalist':
                    result.simplelist([self.scalar * x for x in other.uf])
                elif other.tp == 'scalarlist':
                    result.scalarlist([self.scalar * x for x in other.scalarlist])
                else:
                    raise TypeError("Non-multiplable type.")
            elif str(type(other)) in UFlist:
                result.simpleuf(self.scalar * other)
            else:
                result.simplescalar(self.scalar * other)
        elif mode == 'slist':
            if type(other) == type(result):
                if other.tp == 'uf' or other.tp == 'ufdata':
                    result.simplelist([x * other.uf for x in self.scalarlist])
                elif other.tp == 'scalar':
                    result.simplescalarlist([x * other.scalar for x in self.scalarlist])
                else:
                    raise TypeError("Non-multiplable type.")
            elif str(type(other)) in UFlist:
                result.simplelist([x * other for x in self.scalarlist])
            else:
                result.simplescalarlist([x * other for x in self.scalarlist])
        else:
            raise ValueError("Something is Wrong. fxxk.")
        
        result.varname = self.creation_name
        return result


    def reciprocal(self):
        for frame, line in traceback.walk_stack(None):
            varnames = frame.f_code.co_varnames
            if varnames is ():
                break
            if frame.f_locals[varnames[0]] not in (self, self.__class__):
                break
                # if the frame is inside a method of this instance,
                # the first argument usually contains either the instance or
                #  its class
                # we want to find the first frame, where this is not the case
        else:
            raise InstanceCreationError("No suitable outer frame found.")
        self._outer_frame = frame
        self.creation_module = frame.f_globals["__name__"]
        self.creation_file, self.creation_line, self.creation_function, \
            self.creation_text = \
            traceback.extract_stack(frame, 1)[0]
        self.creation_name = self.creation_text.split("=")[0].strip()
        
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
            result.simplelist([1/x for x in self.uf])
        elif self.tp == 'scalarlist':
            result.simplescalarlist([1/x for x in self.scalarlist])
        else:
            raise ValueError("Something is Wrong, fxxk.")

        result.varname = self.creation_name
        return result


    def __truediv__(self, other):
        for frame, line in traceback.walk_stack(None):
            varnames = frame.f_code.co_varnames
            if varnames is ():
                break
            if frame.f_locals[varnames[0]] not in (self, self.__class__):
                break
                # if the frame is inside a method of this instance,
                # the first argument usually contains either the instance or
                #  its class
                # we want to find the first frame, where this is not the case
        else:
            raise InstanceCreationError("No suitable outer frame found.")
        self._outer_frame = frame
        self.creation_module = frame.f_globals["__name__"]
        self.creation_file, self.creation_line, self.creation_function, \
            self.creation_text = \
            traceback.extract_stack(frame, 1)[0]
        self.creation_name = self.creation_text.split("=")[0].strip()
        
        result = Shvar()
        if type(other) == type(temp):
            result = self * other.reciprocal
        else:
            result = self * (1/other)
        result.varname = self.creation_name
        return result


    def __rtruediv__(self, other):
        for frame, line in traceback.walk_stack(None):
            varnames = frame.f_code.co_varnames
            if varnames is ():
                break
            if frame.f_locals[varnames[0]] not in (self, self.__class__):
                break
                # if the frame is inside a method of this instance,
                # the first argument usually contains either the instance or
                #  its class
                # we want to find the first frame, where this is not the case
        else:
            raise InstanceCreationError("No suitable outer frame found.")
        self._outer_frame = frame
        self.creation_module = frame.f_globals["__name__"]
        self.creation_file, self.creation_line, self.creation_function, \
            self.creation_text = \
            traceback.extract_stack(frame, 1)[0]
        self.creation_name = self.creation_text.split("=")[0].strip()

        result = Shvar()
        if type(other) == type(temp):
            result = other / self
        else:
            result = other * (self.reciprocal)
        result.varname = self.creation_name
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
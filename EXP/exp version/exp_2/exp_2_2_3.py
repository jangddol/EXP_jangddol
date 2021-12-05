import numpy as np
import uncertainties
from uncertainties import ufloat
from uncertainties.umath import *
import copy
from copy import deepcopy

"pip install --upgrade uncertainties"

UF = "<class 'uncertainties.core.Variable'>"
UFAS = "<class 'uncertainties.core.AffineScalarFunc'>"
UFlist = [UF, UFAS]


# 아래의 코멘트는 버젼을 크게 바꾸게 된 이유가 될 수 있다.
# 이참에 .tp의 종류 이름도 좀 바꿔보고자 한다.

# 실험을 하면 무엇을 측정하고, 무엇을 계산하는가?
# 측정을 하는 과정에는 다음과 같은 종류가 있다.
    # 자와 같은 최소 눈금이 있는 측정기구로 하는 측정
    # 디지털 숫자를 읽게 되는 측정
    # 위와 같이 직접 한 측정을 이용해 계산하여 얻은 간접적 측정
# 위와 같이, 측정을 하는 과정이 다르면, 오차가 가지는 확률분포함수(PDF)가 달라진다.
    # 현재 위에는 가우시안과 직사각형이 있다.
    # 만약 T-분포나 감마분포를 추가해야 한다면 어떻게 해야할까.
        # 아마 이는 버젼 업그레이드의 원인이 될지도 모르겠다.
        # 우선은 이를 추가한다는 생각으로 코드를 짜야겠다.
# 측정을 하면 평균과 표준편차, 표준오차가 생성된다.
# 중요한 것은 표준오차 역시 일종의 표준편차라는 것이다.
# 이를 어떻게 누구나 이해하기 쉽게 class variable로 남길 것인가.
# 표준오차가 필요한 때는 확률 변수의 평균의 표준편차가 필요한 때일 것이다.
# 이 확률 변수의 평균은 무엇을 의미하는가?
# 많은 횟수를 거친 측정에서 나오는 평균을 측정값으로 사용하겠다는 의미가 될 것이다.
# 그러므로 측정값의 표준편차가 표준오차가 되는 것일 것이다.
# ufloat끼리 계산하는 일이 필요하다면, 이는 아마 표준오차끼리 계산되는 것일 것이다.
# 직접 집어넣는 ufloat은 표준오차로써 넣는 것이 맞을 것이다.
# 눈금이나 디지털에 의해 얻는 측정은 단 1회의 측정도 오차를 가지게 된다.
    # 이는 직사각형의 PDF를 가정한다.
# 이 역시 고려하게 되는데, 이는 B-type의 오차로써 작동하게 된다.
# 오차(uncertainty)는 A-type과 B-type으로 나누어진다.
    # A-type은 통계에서 오는 오차이다.
    # B-type은 통계 외에서 오는 오차이다.
        # 눈금의 정밀성 부족에 의한 오차도 여기에 속한다.
# 최종적인 오차는 다음과 같이 구해진다.
    # A-type 오차를 ua, B-type 오차를 ub라고 할 때,
    # u = sqrt(ua^2 + ub^2)
    # 여기서 ua는 standard error다.
# A-type 오차를 구할 때
    #표준편차는 통상 n-1로 나누는 것으로 알고 있지만, 참값이 있는 경우에는 n으로 나눈다.


class Shvar():
    #mean+-SE의 형태를 가진 타입을 ufdata라고 하겠다.
    #data는 단순한 입력을 주로 지칭한다.
    #ufdatalist는 ufdata의 list인 것으로 하겠다.


    def __init__(self):
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


    def copy(self):
        return copy.copy(A)

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
        
        try:
            data[0]
        except TypeError:
            raise TypeError("Input 'data' should be a list of numbers.")
        if epdf == 'rec':
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
        if iftval is True:
            self.vrc = sum([(data[i]-tval)**2 for i in range(N)])/N
        elif iftval is False:
            self.vrc = sum([(data[i]-self.mean)**2 for i in range(N)])/(N-1)
        else:
            raise TypeError("Parameter 'iftval' should be True or False.")
        self.stdv = np.sqrt(self.vrc)
        self.SE = np.sqrt((self.stdv**2)/N + self.tbe**2)
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
            .tval   : True Value.
        
        This method return .uf.
        """
        
        try:
            data[0]
        except TypeError:
            raise TypeError("Input 'data' should be a list of list of numbers.")
        N = len(data)
        for X in data:
            try:
                X[0]
            except TypeError:
                raise TypeError("Input 'data' should be a list of list of numbers.")
        for X in data:
            for y in X:
                if isinstance(y, (int, float)):
                    pass
                else:
                    raise TypeError("Input 'data' should be a list of list of numbers.")
        if epdf == 'rec':
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
        if iftval is True:
            try:
                tval[0]
            except TypeError:
                raise TypeError("If iftval is True, parameter 'tval' should subscriptable.")
            if len(tval) != N:
                raise TypeError("If iftval is True, 'data' and 'tval' should have same length.")
            self.vrc = [sum([(data[i][j]-tval[i])**2 for j in range(len(data[i]))])/N for i in range(N)]
        elif iftval is False:
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

        # 그냥 숫자
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
        
        # 그냥 숫자의 리스트
        try:
            scalarlist[0]
        except Exception:
            raise TypeError("The scalarlist should be a list of numbers.")

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
                pass
            else:
                raise TypeError("The arguments should be two numbers.")
            self.mean = args[0]
            self.SE = args[1]
            self.uf = ufloat(self.mean, self.SE)  
            self.tp = 'uf'  
        else:
            raise TypeError("The number of arguments should be one or two.")
        return 0
    
    
    def simplelist(self, *args):
        # 그냥 uf의 리스트
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
    
    
    def ufpp(self, txt=0, txtxt=0, num=2):
        """
        A method printing the uf value of a data or a data list.

        Parameter
            txt   : parameter visualized on the left side of uf value. 
            txtxt : parameter visuzlized on the right side of uf value.
            num   : integer, base = 2, the number of decimal places.
        """
        
        # num is an integer.
        try:
            int(num)
        except Exception:
            raise ValueError("Parameter 'num' should be an integer.")
        
        temp = '{:.' + str(num) + 'uP}'
        if self.tp == 'ufdatalist' or self.tp == 'uflist':
            if txt != 0 :
                if txtxt != 0:
                    i = 0
                    while i < len(self.origin):
                        print(txt, temp.format(self.uf[i]), txtxt)
                        i = i+1
                else:
                    i = 0
                    while i < len(self.origin):
                        print(txt, temp.format(self.uf[i]))
                        i = i+1
            else:
                if txtxt != 0:
                    i = 0
                    while i < len(self.origin):
                        print(temp.format(self.uf[i]), txtxt)
                        i = i+1
                else:
                    i = 0
                    while i < len(self.origin):
                        print(temp.format(self.uf[i]))
                        i = i+1
        elif self.tp == 'ufdata' or self.tp == 'uf':
            if txt != 0 :
                if txtxt != 0:
                    print(txt, temp.format(self.uf), txtxt)
                else:
                    print(txt, temp.format(self.uf))
            else:
                if txtxt != 0:
                    print(temp.format(self.uf), txtxt)
                else:
                    print(temp.format(self.uf))
        else:
            raise TypeError("self.tp should be 'ufdatalist', 'uflist', 'ufdata' or 'uf'.")
    

    def __add__(self, other):
        # 'ufdatalist', 'uflist', 'ufdata', 'uf', 'scalar', 'scalarlist'의 경우를 모두 고려해야한다.
        # 'ufdatalist', 'uflist', 'scalarlist' 를 'list',
        # 'ufdata', 'uf', 'scalar' 를 'uf'로 단순화할 때,
            # 'list' + 'uf', 'uf' + 'list', 'list' + num, 'uf' + 'uf', 'uf' + num
            # 가 가능한 케이스다.
        # Shvar가 아닌 경우는 int, float, ufloat을 고려한다.
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
                result = other + self
            elif str(type(other)) in UFlist:
                result.simpleuf(self.scalar + other)
            else:
                result.simplescalar(self.scalar + other)
        else:
            raise TypeError("Something is Wrong. fxxk.")
        return result


    def __radd__(self, other):
        return self + other
    

    def __pos__(self):
        List = ['uf', 'ufdata', 'uflist', 'ufdatalist', 'scalarlist', 'scalar']
        if self.tp in List:
            return self
        else:
            raise TypeError("Non-plus-attachable.")


    def __neg__(self):
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
        return result
    

    def __sub__(self, other):
        return self + __neg__(other)
    

    def __rsub__(self, other):
        return -(self - other)
    

    def __mul__(self, other):
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
        return result


    def reciprocal(self):
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
        return result


    def __truediv__(self, other):
        result = Shvar()
        if type(other) == type(temp):
            result = self * other.reciprocal
        else:
            result = self * (1/other)
        return result


    def __rtruediv__(self, other):
        result = Shvar()
        if type(other) == type(temp):
            result = other / self
        else:
            result = other * (self.reciprocal)
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

        If fit==True, this method make the following values.
            .slope     : The slope of fitting graph
            .intercept : The intercept of y-axis of fitting graph
            .R2        : The coefficient of Determination of fitting graph
        """

        Temp = Shvar()
        if type(X) == type(Temp) and type(Y) == type(Temp):
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
                self.R2 = temp[2] # ?
            elif X.tp == 'scalarlist' and Y.tp == 'scalarlist':
                temp = linear_fitting(X.scalarlist, Y.scalarlist)
                self.slope = temp[0]
                self.intercept = temp[1]
                self.R2 = temp[2]
            else:
                raise TypeError("Type value of Input X and Y should be 'list' or 'scalarlist'.")
        self.tp = 'double group'
    

    def printfull(self):
        if self.tp == 'uf':
            pass
        elif self.tp == 'ufdata':
            pass
        elif self.tp == 'uflist':
            pass
        elif self.tp == 'ufdatalist':
            pass
        elif self.tp == 'scalar':
            pass
        elif self.tp == 'scalarlist':
            pass
        elif self.tp == 'double group':
            pass
        else:
            pass
        
# 2.x.x버젼을 빠르게 완성시키고, 디버깅을 시작해야함.
# 타입이 변환되는 경우 없애야 하는 변수는 무엇이고 생겨야 하는 변수는 무엇인지
    # 정확하게 인지하고 이를 코드에 반영해야 한다.
        # __init__으로 해결
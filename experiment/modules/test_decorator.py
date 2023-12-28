import attr
# decorator experiments
@attr.define
class C:
    _a: int = 1
    b: int = 2

print('Value Comparison is defined by the decorator || Also recognise different instance of the same class, even admiting same values')
print(C() == C(), C() is C())

print('Initialization on private variable can be done without the underscores')
print(C(a=1)) # Actually C(_a=1) gives error... lazy implementation i guess

@attr.define(kw_only=False)
class C:
    _a: int = 1
    b: int = 2
print('Initialization is possible without explicit mentioning Keyword')
print(C(2, 4))

@attr.define(kw_only=True)
class C:
    _a: int = 1
    b: int = 2
print('KeyWord free initialization is completely turned-off')
try:
    print(C(2, 4))
    print('Still works with kw-free initialization')
except:
    print(C(a=2, b=4))
    print('Only works with kw-only initialization')


# Property -- access computed value without parenthesis
@attr.define(kw_only=True)
class Cplus(C):
    @property
    def prod(self) -> int:
        return self._a * self.b
print('Calling the prod function like a variable here: ')
cp = Cplus()
print(cp.prod)

# @abs.abstractmethod -- used to skip implementation in parent class, but gives error when 
# sub-class doesn't implement it
import abc
@attr.define(kw_only=True)
class C_(abc.ABC): # Note that AbstractClass (abc.ABC) can't be instantiated (pure blueprint)
    _a: int = 1
    b: int = 2
    @property
    @abc.abstractmethod
    def prod(self) -> int:
        # abs.abstractmethod used only for non-implemented function
        # --- any subclass that doesn't implement this is Error-reported ! 
        pass

@attr.define(kw_only=True)
class Cplus(C_):
    def shit(self):
        return 'Shit'

print('Subclass will be invalid & errored if the abstractmethod is not implemented!')
try:
    C_()
except:
    print('abc.ABC abstract class can not be instantiated')
try:
    Cplus()
    print('Subclass inheriting from abstract class C_ can be instantiated')
except:
    print('Subclass also can not be instantiated')
try:
    Cplus().prod
except:
    print('Subclss non-implemented abstractmethod can NOT be called')


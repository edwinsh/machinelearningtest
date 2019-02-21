
#from TestClass import MyClass
#import MyClass

import sys
import TestClass

#import TestClass
#thing = TestClass.MyClass()

from TestClass import MyClass as MC
import TestClass as TC
thing = MC()
thing.apple()

print(TC.a[1])
print(MC.b[1])
print(thing.tangerine)

print("\n\n")
#print(thing.tangerine)
print(sys.path)
print(dir())
print("\n\n")

#from testdir2.TestClass2 import MyClass2 as MC2
from testdir.TestClass2 import MyClass2 as MC2
from testdir import TestClass2

thing = MC2()
thing.apple()
thing = TestClass2.MyClass2()
thing.apple()
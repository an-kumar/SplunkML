from numpyfuncs import *
from utils.test import *


tprint(".array()")
SA_one = array([1,2,3])
SA_two = array(['field1','field2'])
assert SA_one.elems[0][0] == np.array([1,2,3])[0]
assert SA_two.elems[0][1] == np.array(['field1', 'field2'])[1]
assert SA_one.shape == (1,3)
assert SA_two.shape == (1,2)
tprint("passed")

tprint(".sum()")
SA = array([1,2,3,4,5])
# new_sa = 
import timing
timings = timing.Timing()

import os,sys,inspect
def execution_path(filename):
      return os.path.join(os.path.dirname(inspect.getfile(sys._getframe(1))), filename)

def bin(a):
        s=''
        t={'0':'000','1':'001','2':'010','3':'011',
           '4':'100','5':'101','6':'110','7':'111'}
        for c in oct(a)[1:]:
                s+=t[c]
        return s

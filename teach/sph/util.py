import os,sys,inspect
def execution_path(filename):
      return os.path.join(os.path.dirname(inspect.getfile(sys._getframe(1))), filename)

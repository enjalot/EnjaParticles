
import time
from time import clock as clk

def print_timing(func):
    def wrapper(*arg):
        t1 = time.time()
        res = func(*arg)
        t2 = time.time()
        #print "%s took %f ms with args: %s" % (func.func_name, (t2-t1)*1000.0, arg)
        print "%s took %f ms" % (func.func_name, (t2-t1)*1000.0)
        return res
    return wrapper

#----------------------------------------------------------------------

class Timer:
	def tic(self):
		self.time = clk()

	def toc(self, msg=""):
		self.time = clk() - self.time
		if msg:
			print "(%s), time: %f sec" % (msg, 1000.*self.time)
		else:
			print "time: %f sec" % (1000.*self.time)

#----------------------------------------------------------------------

if __name__ == "__main__":

	@print_timing
	def add(x,y):
		for i in range(10000):
			c = x + y
		return c


	add(3.,4.)

	t = Timer()
	x = 3.

	t.tic()
	add(5.,7.)
	t.toc()

	t.tic()
	for i in range(1000):
		x = log(2.+cos(x))
	t.toc()



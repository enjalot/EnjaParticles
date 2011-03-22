
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

def timing_collector(out):
    while True:
        (name, t) = (yield)
        if name in out:
            out[name] += [t]
        else:
            out[name] = [t]


class Timing(object):
    def __init__(self):#,cor):
        self.timings = {}
        self.col = self.__collector()
        self.col.next()                 #coroutine syntax

    def __collector(self):
        while True:
            (name, t) = (yield)         #coroutine syntax
            if name in self.timings:
                self.timings[name]["timings"] += [t]
                self.timings[name]["count"] += 1
                self.timings[name]["total"] += t
            else:
                self.timings[name] = {} #if this entry doesn't exist yet
                self.timings[name]["timings"] = [t]
                self.timings[name]["count"] = 1
                self.timings[name]["total"] = t

    def __call__(self, func):
        """Turn the object into a decorator"""
        def wrapper(*arg, **kwargs):
            t1 = time.time()                #start time
            res = func(*arg, **kwargs)      #call the originating function
            t2 = time.time()                #stop time
            t = (t2-t1)*1000.0              #time in milliseconds
            data = (func.__name__, t)
            self.col.send(data)             #collect the data
            return res
        return wrapper

    def __str__(self):
        s = "Timings:\n"
        print dir(self)
        keys = self.timings.keys()
        for key in sorted(keys):
            s += "%s | " % key
            ts = self.timings[key]["timings"]
            count = self.timings[key]["count"]
            total = self.timings[key]["total"]
            s += "average: %s | total: %s | count: %s\n" % (total / count, total, count)
        return "%s" % s

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


    timings = Timing()
    #@print_timing

    @timings
    def add(x,y):
        for i in range(10000):
            c = x + y
        return c

    #@print_timing
    @timings
    def multiply(x,y):
        for i in range(10000):
            c = x * y
        return c


    for i in range(100):
        add(3.,4.)
        multiply(3., 4.)

    print timings

    """
	t = Timer()
	x = 3.

	t.tic()
	add(5.,7.)
	t.toc()

	t.tic()
	for i in range(1000):
		x = log(2.+cos(x))
	t.toc()
    """

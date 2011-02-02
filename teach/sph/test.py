import numpy as N

class InfoArray(N.ndarray):

    def __new__(subtype, data, info=None, dtype=None, copy=False):
        print "__new__ received %s" % type(data)
        # Make sure we are working with an array, and copy the data if requested
        subarr = N.array(data, dtype=dtype, copy=copy)

        # Transform 'subarr' from an ndarray to our new subclass.
        subarr = subarr.view(subtype)

        # Use the specified 'info' parameter if given
        if info is not None:
            subarr.info = info
        # Otherwise, use data's info attribute if it exists
        elif hasattr(data, 'info'):
                subarr.info = data.info

        # Finally, we must return the newly created object:
        return subarr

    def __array_finalize__(self,obj):
        # We use the getattr method to set a default if 'obj' doesn't have the 'info' attribute
        self.info = getattr(obj, 'info', {})
        # We could have checked first whether self.info was already defined:
        #if not hasattr(self, 'info'):
        #    self.info = getattr(obj, 'info', {})

    def __repr__(self):
        desc="""\
array(data=
  %(data)s,
      tag=%(tag)s)"""
        return desc % {'data': str(self), 'tag':self.info }


import numpy as np

class RealisticInfoArray(np.ndarray):

    def __new__(cls, input_array, info=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.info = info
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.info = getattr(obj, 'info', None)

x = RealisticInfoArray(N.arange(5), info='information')
print type(x)
print x.info

#x = InfoArray(N.arange(10), info={'name':'x'})
#print x

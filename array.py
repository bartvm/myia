import numpy as np
from itertools import accumulate
from functools import reduce
from operator import mul


class Array(object):
    """Symbolic variable that forms in- and outputs to primitives."""
    def __init__(self, type, shape, dtype, tape, strides=None,
                 pool=None, storage=None):
        self.type = type
        self.shape = shape
        self.dtype = dtype
        if strides is None:
            strides = tuple(accumulate((dtype.itemsize,) + shape[:0:-1],
                                       mul))[::-1]
        self.strides = strides
        if storage is None:
            # If this array is a view on another array storage will be given
            storage = Storage(self.nbytes, pool)
        self.storage = storage

        self.tape = tape

    @property
    def size(self):
        if not self.shape:
            return 0
        else:
            return reduce(mul, self.shape)

    @property
    def nbytes(self):
        return self.size * self.dtype.itemsize

    @property
    def data(self):
        """Construct an array with the given shape and dtype.

        Storage will allocate data if needed.

        """
        assert self.storage.memory is not None
        return self.type(self.shape, dtype=self.dtype,
                         buffer=self.storage.memory.data, strides=self.strides)


class Storage(object):
    """The storage that an array refers to.

    Note that this isn't the actual memory, which is only assigned at runtime.

    """
    def __init__(self, size, pool):
        self.size = size
        # The pool is used to create new memory if necessary
        self.pool = pool
        # Keeps track of how many arrays require this storage to be alive
        # Will be incremented for each time this storage is used as an input
        # or output to a node, will be decremented by op or after op
        self.refcount = 0
        # Initially memory-less, will be allocated during runtime by op
        self._memory = None

    def increment(self):
        self.refcount += 1

    def decrement(self):
        # Should only occur during runtime
        assert self._memory is not None
        assert self.refcount > 0
        self.refcount -= 1
        self.memory.refcount -= 1
        if self.refcount == 0:
            self._memory = None

    @property
    def memory(self):
        return self._memory

    @memory.setter
    def memory(self, memory):
        """Set reference count on memory when assigned."""
        # Can be 0 if this is the input
        assert self.refcount >= 0
        self._memory = memory
        self._memory.refcount += self.refcount

    def allocate(self):
        """Either this is called, or the memory is set directly."""
        # Refcount can be 0 if this is the input
        assert self.refcount >= 0
        # Property setter will assign reference count
        self.memory = Memory(self.pool.allocate(self.size), self.pool)


class Memory(object):
    """"This holds the actual data, it must be initialized.

    Can be freed if the refcount reaches 0.

    """
    def __init__(self, data, pool):
        self.data = data
        self.pool = pool
        self._refcount = 0

    @property
    def refcount(self):
        return self._refcount

    @refcount.setter
    def refcount(self, value):
        assert value >= 0
        assert value == 0 or value != self._refcount
        if value == 0:
            self.pool.free(self.data)
            self.data = None
        self._refcount = value


class Pool(object):
    """A memory pool"""
    pass


class NumPyPool(Pool):
    def __init__(self):
        self.pool = {}

    def allocate(self, size):
        data = self.pool.get(size)
        if data:
            return data.pop()
        else:
            return np.empty((size,), dtype=np.dtype('uint8')).data

    def free(self, data):
        self.pool.setdefault(data.nbytes, []).append(data)


class Primitive(object):
    """A user-facing operation with a given number of in and outputs.

    Based on the type of inputs it has to add a series of backend-specific
    operations to the tape. Note that one primitive can consist of multiple
    operations. The output should be a node of the relevant type (usually
    on the same backend).

    """
    def __init__(self, fun):
        self.fun = fun

    def __call__(self, *inputs):
        tape = inputs[0].tape
        for input in inputs:
            input.storage.increment()
        tape.extend(self.fun(*inputs))
        for output in tape[-1].outputs:
            output.storage.increment()
        return tape[-1].outputs


@Primitive
def add(left, right):
    """Returns the op (numpy_add) and a new array."""
    # TODO Find a better way to give the primitive access to the tape and pool
    return [NumPyAdd((left, right),
                     (Array(np.ndarray, left.shape, left.dtype,
                            left.tape, pool=left.storage.pool),))]


@Primitive
def transpose(x):
    return [NumPyTranspose((x,),
                           (Array(np.ndarray, x.shape[::-1], x.dtype,
                                  x.tape, strides=x.strides[::-1],
                                  storage=x.storage),))]


class Node(object):
    """Representation of an operation in the tape."""
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __call__(self):
        pass


class NumPyAdd(Node):
    """Back-end specific implementation"""
    grad_requires = ((), ())

    def __call__(self):
        # Allocate new data for output
        if (self.inputs[0].storage.refcount == 1) or \
                (self.inputs[0].storage.refcount == 2 and
                 self.inputs[0].storage is self.inputs[1].storage):
            self.outputs[0].storage.memory = self.inputs[0].storage.memory
        elif self.inputs[1].storage.refcount == 1:
            self.outputs[0].storage.memory = self.inputs[1].storage.memory
        else:
            self.outputs[0].storage.allocate()
        np.add(self.inputs[0].data, self.inputs[1].data, self.outputs[0].data)
        for array in self.inputs + self.outputs:
            array.storage.decrement()

    def gradients(self, required):
        for i in required:
            self.inputs[i].grad_storage = self.outputs[0].grad_storage
            self.inputs[i].grad_strides = self.outputs[0].grad_strides


class NumPyTranspose(Node):
    pass


def forward(tape):
    # Make sure the answer doesn't get freed
    for output in tape[-1].outputs:
        output.storage.increment()
    for node in tape:
        node()
    return [output.data for output in node.outputs]

if __name__ == "__main__":
    # There should be a single global pool (and tape?)
    pool = NumPyPool()

    # Construct the input array
    x = Array(np.ndarray, (3, 3), np.dtype('float64'), [], pool=pool)

    # Construct the comptuation graph
    y, = add(x, x)

    # Allocate data after creating graph so that memory refcount is correct
    x_data = np.random.rand(3, 3).data
    x.storage.allocate()
    x.storage.memory.data = x_data

    # Check results
    print(y.tape)
    print(y.tape[0].inputs, y.tape[0].outputs)
    print(forward(y.tape))
    print(pool.pool)

# Myia

A Python AD library.

Outline of structure:

1. Variables `x` and `y` are created. The objects themselves are backend-agnostic and only contain shape information, but contain pointers to backend-specific data (e.g. a NumPy array or a CUDA handle).
  1. Do they only need shape information or also data type?
2. A user performs a call `sum(x, y)` inside a decorated function (or context manager).
3. Based no the input data types a backend-specific operation e.g. `numpy_sum(x, y)` is added to a tape, but not actually performed (expressions are evaluated lazily).
4. All operations are added to the same tape, but the actual tree structure is maintained through the input/output relations between variables and operations.
4. Steps (2) and (3) are repeated until a call is made that requires evaluation (e.g. the value is printed to the screen, compared to a constant, or the function ends)
5. Optional: The tape between the last eveluation and the current point is optimized (like a tracing JIT e.g. constant folding, merging of operations).
8. The required values are calculated. Each node registers whether it requires its outputs and/or inputs for the backpropagation (so that in-place versions of ops can be called if not).
9. Whenever an output is calculated, memory is requested from a memory pool (which has backend-specific implementations). This allows memory to be released and reused as quickly as possible.

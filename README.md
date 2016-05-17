# Myia

A Python AD library.

Outline of structure:

1. Variables `x` and `y` are created. The objects themselves are backend-agnostic and only contain shape information, but contain pointers to backend-specific data (e.g. a NumPy array or a CUDA handle).
  1. Do they only need shape information or also data type?
2. A user performs a call `sum(x, y)`.
3. The operation `sum(x, y)` is added to a tape, but not actually performed (expressions are evaluated lazily).
4. Steps (2) and (3) are repeated until a call is made that requires evaluation (e.g. the value is printed to the screen or compared to a constant)
5. Optional: The tape between the last eveluation and the current point is optimized (like a tracing JIT e.g. constant folding, merging of operations) using backend-agnostic optimizations.
6. The tape operations are transformed into back-end specific implementations based on the data associated with the variables. Note that some general operations can consist of multiple backend-specific operations (e.g. if the backend kernels are more granular or lack a certain method). The backend can change during execution if the user explicitly changes the data type.
  1. Can back-end specific operations only become more granular? In that case the mapping between the two tapes might be simpler.
7. Optional: The tape is once again optimized, but this time with backend specific optimizations.
  1. This could complicate the mapping between the backend-agnostic and backend-specific tapes.
8. The required values are calculated. Intermediary values required for backpropagation are stored if necessary.
  1. Are the intermediary values stored at points in the backend-specific or backend-agnostic tape?

A user requests the gradient of a function, `grad(func)(x, y)`. We go through the tapes (one tape per input variable) in reverse: Do we play the backend-specific or backend-agnostic tape in reverse? Backend-agnostic means that not each backend-specific op needs a gradient defined, but the gradient of a softmax for NumPy would simply be the gradient of a series of forward ops. Advantages of expressing backward prop in elementary operations.

1. Backend-agnostic and backend-specific optimizations can once again be applied.
  1. Need to keep track of the intermediary values, which might no longer match the boundaries of the original (optimized) reversed tape.

Memory management

1. Heavily platform dependent e.g. CNMeM, but in some cases very important to re-use tensors in subsequent runs to avoid slow memory allocations.
  1. Could have a backend-agnostic memory pool, alloc and free calls, and a backend-specific implementation of these things. Need to consider how datatypes play a role here.

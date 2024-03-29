"""Definition of shape inference for primitives."""

import operator
from functools import partial, reduce

from ..infer import ANYTHING, GraphInferrer, register_inferrer, \
    PartialInferrer, Track, MyiaShapeError, Inferrer
from ..ir import Graph
from ..dtype import Array

from . import ops as P
from .ops import Primitive


def prod(iterable):
    """Return the product of the elements of the iterator."""
    return reduce(operator.mul, iterable, 1)


shape_inferrer_constructors = {}


class ScalarShapeInferrer(Inferrer):
    """Shape inferrer for all primitives that don't take arrays."""

    def __init__(self, engine):
        """Initialize the ScalarShapeInferrer."""
        super().__init__(engine, 'scalar_shape_inferrer')

    async def __call__(self, *args):
        """Since no arrays are involved, the shape is always ()."""
        return ()

    def provably_equivalent(self, other):
        """This is always equal to itself."""
        return type(self) == type(other)


class ShapeTrack(Track):
    """Infer the shape of a constant."""

    def __init__(self, engine, name, *,
                 constructors=shape_inferrer_constructors):
        """Initialize a ShapeTrack."""
        super().__init__(engine, name)
        self.constructors = constructors

    def from_value(self, v, context):
        """Infer the shape of a constant."""
        if isinstance(v, Primitive):
            if v in self.constructors:
                return self.constructors[v](self.engine)
            else:
                return ScalarShapeInferrer(self.engine)
        elif isinstance(v, Graph):
            return GraphInferrer(self.engine, 'shape', v, context)
        else:
            return getattr(v, 'shape', ())


shape_inferrer = partial(register_inferrer,
                         constructors=shape_inferrer_constructors)


@shape_inferrer(P.return_, nargs=1)
async def infer_shape_return(engine, v):
    """Infer the shape of return."""
    return await v['shape']


@shape_inferrer(P.if_, nargs=3)
async def infer_shape_if(engine, cond, tb, fb):
    """Infer the shape of if."""
    tb_inf = await tb['shape']
    fb_inf = await fb['shape']
    v = await cond['value']
    if v is True:
        # We only visit the first branch if the condition is provably true
        return await tb_inf()
    elif v is False:
        # We only visit the second branch if the condition is provably false
        return await fb_inf()
    elif v is ANYTHING:
        # The first branch to finish will return immediately. When the other
        # branch finishes, its result will be checked against the other.
        return await engine.assert_same('shape', tb_inf(), fb_inf())
    else:
        raise AssertionError("Invalid condition value for if.")


@shape_inferrer(P.partial, nargs=None)
async def infer_shape_partial(engine, fn, *args):
    """Infer the return type of partial."""
    fn_t = await fn['shape']
    return PartialInferrer(engine, fn_t, args)


@shape_inferrer(P.map_array, nargs=2)
async def infer_shape_map_array(engine, fn, ary):
    """Infer the shape of map_array."""
    return await ary['shape']


@shape_inferrer(P.scan_array, nargs=4)
async def infer_shape_scan_array(engine, fn, init, ary, ax):
    """Infer the shape of scan_array."""
    return await ary['shape']


@shape_inferrer(P.reduce_array, nargs=4)
async def infer_shape_reduce_array(engine, fn, init, ary, ax):
    """Infer the shape of reduce_array."""
    shp = await ary['shape']
    ax_v = await ax['value']
    if ax_v == ANYTHING:
        return (ANYTHING,) * (len(shp) - 1)
    else:
        lshp = list(shp)
        del lshp[ax_v]
        return tuple(lshp)


@shape_inferrer(P.distribute, nargs=2)
async def infer_shape_distribute(engine, v, shape):
    """Infer the shape of distribute."""
    shp = await shape['value']
    if shp == ANYTHING:
        shp_t = await shape['type']
        shp = (ANYTHING,) * len(shp_t.elements)
    v_t = await v['type']
    if isinstance(v_t, Array):
        v_shp = await v['shape']
        if len(shp) < len(v_shp):
            raise MyiaShapeError("Cannot distribute to smaller shape")
        for vs, s in zip(v_shp, shp):
            if vs != s and vs not in (1, ANYTHING) and s not in (1, ANYTHING):
                raise MyiaShapeError("Cannot change shape when distributing")
    return shp


@shape_inferrer(P.reshape, nargs=2)
async def infer_shape_reshape(engine, v, shape):
    """Infer the shape of reshape."""
    shp = await shape['value']
    if shp == ANYTHING:
        shp_t = await shape['type']
        shp = (ANYTHING,) * len(shp_t.elements)
    v_shp = await v['shape']
    if (all(s is not ANYTHING for s in shp) and
        all(s is not ANYTHING for s in v_shp) and
            prod(shp) != prod(v_shp)):
        raise MyiaShapeError("Cannot change the total number of elements "
                             "in reshape")
    return shp


@shape_inferrer(P.dot, nargs=2)
async def infer_shape_dot(engine, a, b):
    """Infer the shape of dot."""
    a_shp = await a['shape']
    b_shp = await b['shape']
    if len(a_shp) != 2 or len(b_shp) != 2:
        raise MyiaShapeError("dot needs matrix inputs")
    if (a_shp[1] != b_shp[0] and
        a_shp[1] is not ANYTHING and
            b_shp[0] is not ANYTHING):
        raise MyiaShapeError("Incompatible shapes in dot")
    return (a_shp[0], b_shp[1])

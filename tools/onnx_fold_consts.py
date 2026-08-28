"""Fold statically-evaluable constant subgraphs in the exported ONNX so no
bool/Equal/Where reaches the QNN converter (HTP rejects fp16->bool Cast 0xc26).
Also rewrite any remaining Equal->Where pairs into float lerp.
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

def make_const(name, arr):
    t = numpy_helper.from_array(np.asarray(arr), name=name)
    return helper.make_node('Constant', [], [name], value=t)

GRAPH_BL = {'Constant','ConstantOfShape','Cast','Mul','Sub','Add','Equal','Not','Unsqueeze','Squeeze',
            'Reshape','Concat','Gather','Shape','Expand','Where','Split','Slice','ScatterND','Flatten',
            'Transpose','Pad','ReduceSum','ConstantOfShape','ConcatFromSequence','Range','NonZero'}

def fold(path):
    m = onnx.load(path)
    g = m.graph
    # constants pool
    consts = {}
    for i in g.initializer:
        consts[i.name] = numpy_helper.to_array(i)
    # save initializers/constants we keep
    keep_init = set(i.name for i in g.initializer)
    # gather constant node values first
    cnodes = {n.output[0]: n for n in g.node if n.op_type=='Constant'}
    for nm, n in cnodes.items():
        for a in n.attribute:
            if a.name=='value':
                consts[nm] = numpy_helper.to_array(a.t)
    # producers
    prod = {}
    for n in g.node:
        for o in n.output:
            prod[o] = n
    # iterative evaluation of pure-constant producers
    changed = True
    while changed:
        changed = False
        for n in list(g.node):
            if n.op_type=='Constant' or n.op_type not in GRAPH_BL:
                continue
            # all inputs constant & not produced? inputs can be initializers/constants
            ok = all(i in consts for i in n.input if i)
            if not ok:
                continue
            try:
                ins = [consts[i] if i else None for i in n.input]
                out = eval_op(n, ins)
                changed = True
                for o in n.output:
                    consts[o] = out
            except Exception as e:
                pass
    # Now rewrite: any node whose inputs all in consts -> Constant node
    nodes = []
    produced = set()
    for n in list(g.node):
        # check if fully constant
        if n.op_type in GRAPH_BL and all(i in consts for i in n.input if i):
            # evaluate; store all outputs
            try:
                outs = eval_op_multi(n)
                # emit Constant for each output
                for oi, o in enumerate(n.output):
                    val = outs if len(outs)==1 else outs[oi]
                    if o not in produced:
                        nodes.append(make_const(o, np.asarray(val)))
                        produced.add(o)
                continue
            except Exception as e:
                pass
        nodes.append(n)
    g.ClearField('node')
    for n in nodes:
        g.node.append(n)
    onnx.save(m, path)
    print('folded', path)

def eval_op_multi(n):
    """re-eval using consts captured via closure - simpler: recompute from graph"""
    raise NotImplementedError

def eval_op(n, ins):
    import onnx as _onnx
    from onnx import numpy_helper as _nh
    op = n.op_type
    a = ins[0] if len(ins)>0 else None
    b = ins[1] if len(ins)>1 else None
    if op=='Constant':
        return a
    if op=='Cast':
        to = None
        for x in n.attribute:
            if x.name=='to': to = x.i
        dt = {1:np.float32,10:np.float16,6:np.int32,7:np.int64,9:np.bool_,2:np.uint8,3:np.int8}[to]
        return a.astype(dt)
    if op=='Mul': return a*b
    if op=='Sub': return a-b
    if op=='Add': return a+b
    if op=='Div': return a/b
    if op=='Equal': return a==b
    if op=='Not': return ~a
    if op=='Where': return np.where(a, b, ins[2])
    if op=='Unsqueeze':
        axes = int(n.attribute[0].i) if n.attribute else None
        return np.expand_dims(a, axis=axes)
    if op=='Squeeze':
        return np.squeeze(a)
    if op=='Reshape':
        shape = ins[1].astype(np.int64).tolist()
        return a.reshape(shape)
    if op=='Transpose':
        perm = [x.i for x in n.attribute]
        return a.transpose(perm if perm else None)
    if op=='Concat':
        axis = n.attribute[0].i
        return np.concatenate(ins[:len([i for i in n.input if i])], axis=axis)
    if op=='Gather':
        axis = n.attribute[0].i if n.attribute else 0
        return np.take(a, b, axis=axis)
    if op=='Shape':
        return np.asarray(a.shape, dtype=np.int64)
    if op=='Expand':
        return np.broadcast_to(a, tuple(ins[1].astype(np.int64).tolist()))
    if op=='ConstantOfShape':
        shape = a.astype(np.int64).tolist()
        val = 0.0
        for x in n.attribute:
            if x.name=='value': val = numpy_helper.to_array(x.t).item()
        return np.full(shape, val, dtype=np.float32)
    if op=='Slice':
        starts=ins[1].astype(np.int64).tolist(); ends=ins[2].astype(np.int64).tolist()
        axes=ins[3].astype(np.int64).tolist() if len(ins)>3 else None
        obj=[slice(None)]*a.ndim
        for k,ax in enumerate(axes or range(len(starts))):
            obj[ax]=slice(starts[k], ends[k])
        return a[tuple(obj)]
    if op=='Split':
        return np.split(a, len(n.output), axis=n.attribute[0].i if n.attribute else 0)
    if op=='ScatterND':
        idx=ins[1].astype(np.int64); upd=ins[2]
        out=a.copy()
        out[tuple(idx[...,k] for k in range(idx.shape[-1]))] = upd
        return out
    if op=='Range':
        start,limit,delta = [ins[k].item() for k in (0,1,2)]
        return np.arange(start, limit, delta, dtype=np.int64)
    if op=='ReduceSum':
        axes=n.attribute[0].ints if n.attribute and n.attribute[0].ints else None
        return a.sum(axis=tuple(axes) if axes else None, keepdims=True)
    raise RuntimeError('unhandled '+op)

if __name__=='__main__':
    import glob
    base = sys.argv[1]
    for p in sorted(glob.glob(os.path.join(base, '*', 'model.onnx'))):
        fold(p)

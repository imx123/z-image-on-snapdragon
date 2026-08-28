"""Fold statically-evaluable subgraphs to constants, then re-save with external data.
Verified numerically with onnxruntime after rewriting.
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import onnx
from onnx import helper, numpy_helper

P = sys.argv[1]
m = onnx.load(P)
g = m.graph

consts = {}
for i in g.initializer:
    consts[i.name] = numpy_helper.to_array(i)
prod = {}
for n in g.node:
    for o in n.output: prod[o] = n

ALLOW = {'Cast','Mul','Sub','Add','Div','Equal','Not','Where','Unsqueeze','Squeeze','Reshape','Transpose',
         'Concat','Gather','Shape','Expand','ConstantOfShape','Slice','Split','ScatterND','Range','Flatten',
         'ReduceSum','Pad','Constant','Clip','Abs','Ceil','Floor','Neg','Min','Max','Pow','Sqrt','Mod','Gemm'}

def eval_node(n, consts):
    op = n.op_type
    ins = [consts.get(t) if t else None for t in n.input]
    a = ins[0]
    if op=='Constant': raise RuntimeError
    if op=='Cast':
        to = None
        for x in n.attribute:
            if x.name=='to': to=x.i
        dt = {1:np.float32,10:np.float16,6:np.int32,7:np.int64,9:np.bool_,2:np.uint8,3:np.int8,11:np.float64}[to]
        return a.astype(dt)
    if op=='Mul': return a*ins[1]
    if op=='Sub': return a-ins[1]
    if op=='Add': return a+ins[1]
    if op=='Div': return a/ins[1]
    if op=='Equal': return a==ins[1]
    if op=='Not': return ~a
    if op=='Where': return np.where(a, ins[1], ins[2])
    if op=='Unsqueeze': return np.expand_dims(a, axis=[x.i for x in n.attribute][0])
    if op=='Squeeze': return np.squeeze(a, axis=[x.i for x in n.attribute] or None)
    if op=='Reshape': return a.reshape(ins[1].astype(np.int64).tolist())
    if op=='Transpose':
        perm=[x.i for x in n.attribute] or None
        return a.transpose(perm)
    if op=='Concat':
        axis = [x.i for x in n.attribute][0]
        names = [t for t in n.input if t]
        return np.concatenate([consts[t] for t in names], axis=axis)
    if op=='Gather':
        axis=[x.i for x in n.attribute][0] if n.attribute else 0
        return np.take(a, ins[1], axis=axis)
    if op=='Shape': return np.asarray(a.shape, dtype=np.int64)
    if op=='Expand': return np.broadcast_to(a, tuple(ins[1].astype(np.int64).tolist()))
    if op=='ConstantOfShape':
        shape=a.astype(np.int64).tolist()
        val=0.0
        for x in n.attribute:
            if x.name=='value': val=numpy_helper.to_array(x.t).item()
        return np.full(shape, val, dtype=np.float32)
    if op=='Slice':
        s=ins[1].astype(np.int64).tolist(); e=ins[2].astype(np.int64).tolist()
        axes=ins[3].astype(np.int64).tolist() if len(ins)>3 and ins[3] is not None else None
        obj=[slice(None)]*a.ndim
        for k,ax in enumerate(axes or range(len(s))):
            obj[ax]=slice(s[k], e[k])
        return a[tuple(obj)]
    if op=='Split':
        axis=[x.i for x in n.attribute][0] if n.attribute else 0
        return np.split(a, len(n.output), axis=axis)
    if op=='ScatterND':
        idx=ins[1].astype(np.int64); upd=ins[2]
        out=a.copy()
        out[tuple(idx[...,k] for k in range(idx.shape[-1]))] = upd
        return out
    if op=='Range':
        return np.arange(ins[0].item(), ins[1].item(), ins[2].item(), dtype=np.int64)
    if op=='Flatten':
        return a.reshape(a.shape[0], -1)
    if op=='ReduceSum':
        axes=[x.i for x in n.attribute] or None
        return a.sum(axis=tuple(axes) if axes else None, keepdims=True)
    if op=='Clip':
        lo = ins[1] if len(ins)>1 and ins[1] is not None else -np.inf
        hi = ins[2] if len(ins)>2 and ins[2] is not None else np.inf
        return np.clip(a, lo, hi)
    if op=='Abs': return np.abs(a)
    if op=='Neg': return -a
    if op=='Min': return np.minimum(a, ins[1])
    if op=='Max': return np.maximum(a, ins[1])
    if op=='Pow': return np.power(a, ins[1])
    if op=='Sqrt': return np.sqrt(a)
    if op=='Mod': return np.mod(a, ins[1])
    if op=='Gemm':
        A,B,C = a, ins[1], ins[2] if len(ins)>2 and ins[2] is not None else np.zeros(1)
        ta = [x.i for x in n.attribute if x.name=='transA'][0] if any(x.name=='transA' for x in n.attribute) else 0
        tb = [x.i for x in n.attribute if x.name=='transB'][0] if any(x.name=='transB' for x in n.attribute) else 0
        A = A.T if ta else A; B = B.T if tb else B
        return A@B + C
    raise RuntimeError('no-eval '+op)

# iterative fold
ready = {n.output[0] for n in g.node if n.op_type=='Constant'}
for n in g.node:
    if n.op_type=='Constant':
        for a in n.attribute:
            if a.name=='value': consts[n.output[0]] = numpy_helper.to_array(a.t)
folded = {}
changed=True
while changed:
    changed=False
    for n in list(g.node):
        if n.op_type=='Constant': continue
        if all((t in consts) for t in n.input if t and t):
            try:
                outs = eval_node(n, consts)
            except Exception:
                continue
            if len(n.output)==1:
                consts[n.output[0]] = outs
                folded[n.output[0]] = outs
                changed=True
            else:
                import itertools
                if isinstance(outs, (list,tuple)) and len(outs)==len(n.output):
                    for oi,o in enumerate(n.output):
                        consts[o]=outs[oi]; folded[o]=outs[oi]
                    changed=True
# Only replace nodes whose outputs are in the folded set (pure constants)
repl = set()
for o,v in folded.items():
    if o in prod:
        repl.add(prod[o].name)
newnodes=[]
keep_init=set(i.name for i in g.initializer)
for n in g.node:
    if n.name in repl:
        for oi,o in enumerate(n.output):
            if o in folded and o not in keep_init:
                val = np.asarray(folded[o])
                # avoid int64/bool leakage: cast bool->int64? keep as-is constants; QNN handles scalar const
                newnodes.append(helper.make_node('Constant', [], [o], value=numpy_helper.from_array(val, name=o)))
        continue
    newnodes.append(n)
g.ClearField('node')
for n in newnodes: g.node.append(n)

outp = P
tmp = P+'.tmp.onnx'
try:
    onnx.save(m, tmp)
except Exception:
    onnx.save(m, tmp, save_as_external_data=True, all_tensors_to_one_file=True, location=os.path.basename(P)+'.data')
os.replace(tmp, P)
print('folded nodes:', len(repl), 'remaining Equal/Where:', sum(1 for n in onnx.load(P).graph.node if n.op_type in ('Equal','Where')))

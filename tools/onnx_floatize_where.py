"""Rewrite Equal->Where (BOOL) subgraphs into float lerp so QNN FP16 conversion
emits no fp16->bool Cast (HTP rejects it: 0xc26). Numerically identical:
Where(cond,A,B) == B + cond_f*(A-B), cond_f = 1-clamp(|X-C|,0,1) for X==C.
"""
import sys, os, io
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

def const_val(name, graph):
    # initializer or Constant node
    for i in graph.initializer:
        if i.name == name: return numpy_helper.to_array(i)
    for n in graph.node:
        if n.op_type == 'Constant' and n.output[0] == name:
            for a in n.attribute:
                if a.name=='value': return numpy_helper.to_array(a.t)
    return None

def rewrite(path):
    m = onnx.load(path)
    g = m.graph
    node_names = [n.output[0] for n in g.node]
    new_nodes = []
    consumed_removed = set()
    used_inputs = set(t for n in g.node for t in n.input)
    # collect Equal->Where pairs
    equals = {n.output[0]: n for n in g.node if n.op_type=='Equal'}
    replaced = set()
    name_ct = [0]
    def fresh(p): name_ct[0]+=1; return f'{p}_{name_ct[0]}'
    for n in list(g.node):
        if n.op_type=='Where' and n.input[0] in equals:
            eq = equals[n.input[0]]
            A, B = n.input[1], n.input[2]
            X, C = eq.input[0], eq.input[1]
            cv = const_val(C, g) if const_val(C, g) is not None else None
            if cv is None:
                cv = const_val(X, g)
                if cv is None:
                    new_nodes.append(n); continue
                X, C = C, X
            def tof32(t):
                v = const_val(t, g)
                if v is not None:
                    nm = fresh('c32'); new_nodes.append(helper.make_node('Constant', [], [nm], value=numpy_helper.from_array(np.asarray(v, dtype=np.float32), name=nm)))
                    return nm
                nm = fresh('f32'); new_nodes.append(helper.make_node('Cast', [t], [nm], to=TensorProto.FLOAT))
                return nm
            Xf = tof32(X); Cf = tof32(C); Af = tof32(A); Bf = tof32(B)
            d = fresh('d'); new_nodes.append(helper.make_node('Sub', [Xf, Cf], [d]))
            ad = fresh('ad'); new_nodes.append(helper.make_node('Abs', [d], [ad]))
            zero = fresh('z'); one = fresh('o')
            new_nodes.append(helper.make_node('Constant', [], [zero], value=numpy_helper.from_array(np.float32(0.0), name=zero)))
            new_nodes.append(helper.make_node('Constant', [], [one], value=numpy_helper.from_array(np.float32(1.0), name=one)))
            cl = fresh('cl'); new_nodes.append(helper.make_node('Clip', [ad, zero, one], [cl]))
            cf_f = fresh('cf'); new_nodes.append(helper.make_node('Sub', [one, cl], [cf_f]))
            ab = fresh('ab'); new_nodes.append(helper.make_node('Sub', [Af, Bf], [ab]))
            mul = fresh('mul'); new_nodes.append(helper.make_node('Mul', [cf_f, ab], [mul]))
            addf = fresh('addf'); new_nodes.append(helper.make_node('Add', [Bf, mul], [addf]))
            out = n.output[0]
            # downstream is shape/int64 (Expand/Unsqueeze) -> cast back to INT64
            new_nodes.append(helper.make_node('Cast', [addf], [out], to=TensorProto.INT64))
            replaced.add(eq.name); replaced.add(n.name)
        else:
            new_nodes.append(n)
    # sanity: any missing inputs after removal?
    produced = set()
    for x in new_nodes: produced.update(x.output)
    kept_inits = set(i.name for i in g.initializer)
    missing = [t for x in new_nodes for t in x.input if t and t not in produced and t not in kept_inits]
    # Drop Equal/Where nodes (they were replaced); keep everything else.
    final = [n for n in new_nodes if n.name not in replaced]
    g.ClearField('node')
    for n in final:
        g.node.append(n)
    try:
        onnx.save(m, path)
    except Exception:
        onnx.save(m, path, save_as_external_data=True, all_tensors_to_one_file=True,
                  location=os.path.basename(path) + '.data')
    if missing:
        print('WARN missing inputs:', missing[:6], ' in', os.path.dirname(path))
    rc = [n for n in final if n.op_type in ('Equal','Where')]
    print(os.path.basename(os.path.dirname(path)), 'replaced', len(replaced), 'remaining Equal/Where:', len(rc))

if __name__=='__main__':
    import glob
    base = sys.argv[1]
    for p in sorted(glob.glob(os.path.join(base, '*', 'model.onnx'))):
        rewrite(p)

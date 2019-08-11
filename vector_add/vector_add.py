from __future__ import absolute_import, print_function
import tvm
import numpy as np

tgt_host="llvm"
# tgt="llvm"
tgt="c"

n = tvm.var("n")
A = tvm.placeholder((n,), name='A')
B = tvm.placeholder((n,), name='B')
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")
print(type(C))
s = tvm.create_schedule(C.op)
bx, tx = s[C].split(C.op.axis[0], factor=64)

fadd = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name="myadd")

# print(fadd.get_source())

ctx = tvm.context(tgt, 0)

n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
fadd(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

print("finished.")
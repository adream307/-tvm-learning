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

with tvm.build_config(dump_pass_ir=True):
    fadd = tvm.build(s, [A, B, C], tgt,name="myadd")

print(fadd.get_source())

print("finished.")

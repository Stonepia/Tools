

# Original file: ./soft_actor_critic___60.0/soft_actor_critic___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/fs/cfsiioehxa4hk75dw3oqqthdx6uo2nyahd77zc5ls4ssg6rvfkl7.py
# Source Nodes: [add, add_1, exp, mul, tanh], Original ATen: [aten.add, aten.exp, aten.mul, aten.tanh]
# add => add
# add_1 => add_1
# exp => exp
# mul => mul
# tanh => tanh
triton_poi_fused_add_exp_mul_tanh_1 = async_compile.triton('triton_poi_fused_add_exp_mul_tanh_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[256], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_exp_mul_tanh_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_exp_mul_tanh_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + (2*x0)), xmask).to(tl.float32)
    tmp1 = libdevice.tanh(tmp0)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 6.0
    tmp5 = tmp3 * tmp4
    tmp6 = -10.0
    tmp7 = tmp5 + tmp6
    tmp8 = tl.exp(tmp7)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''')

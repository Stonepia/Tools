

# Original file: ./AllenaiLongformerBase__22_backward_143.5/AllenaiLongformerBase__22_backward_143.5.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/6h/c6hrq7brgbfdie3pju4fr7cbrf5r2dv5g63dfpdj42f4gn46k6eg.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd]

triton_poi_fused_constant_pad_nd_21 = async_compile.triton('triton_poi_fused_constant_pad_nd_21', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_21', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_constant_pad_nd_21(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 37748736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 512) % 512
    x2 = (xindex // 262144) % 48
    x3 = (xindex // 12582912)
    x4 = xindex % 262144
    tmp0 = x1
    tmp1 = tl.full([1], 513, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + ((513*x2) + (513*((x4 + (262656*x3)) // 787968)) + (24624*(x4 // 513)) + (12607488*x3) + (x4 % 513)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + ((513*x2) + (513*((x4 + (262656*x3)) // 787968)) + (24624*(x4 // 513)) + (12607488*x3) + (x4 % 513)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp5, 0.0)
    tl.store(out_ptr0 + (x4 + (262144*x3) + (786432*x2)), tmp6, None)
''')

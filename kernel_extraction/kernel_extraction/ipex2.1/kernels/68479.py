

# Original file: ./tf_mixnet_l___60.0/tf_mixnet_l___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/gs/cgsd5hkntrfif5o32vii4ufvsyqoa6ii5s7rsewlom2khbifii6r.py
# Source Nodes: [pad_14], Original ATen: [aten.constant_pad_nd]
# pad_14 => constant_pad_nd_14
triton_poi_fused_constant_pad_nd_81 = async_compile.triton('triton_poi_fused_constant_pad_nd_81', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_81', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_constant_pad_nd_81(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13547520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5040) % 21
    x1 = (xindex // 240) % 21
    x0 = xindex % 240
    x3 = (xindex // 105840)
    x6 = xindex
    tmp0 = (-3) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-3) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-42480) + x0 + (960*x1) + (13440*x2) + (188160*x3)), tmp10, other=0.0)
    tmp12 = tl.where(tmp10, tmp11, 0.0)
    tl.store(out_ptr0 + (x6), tmp12, None)
''')
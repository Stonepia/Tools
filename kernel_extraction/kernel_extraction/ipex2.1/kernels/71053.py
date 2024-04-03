

# Original file: ./timm_nfnet___60.0/timm_nfnet___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/ro/crokme2zowbpj2fccysg3oij7i6gc4dlva344urmxmekizdkgiq2.py
# Source Nodes: [mul__19, pad_3], Original ATen: [aten.constant_pad_nd, aten.mul]
# mul__19 => mul_143
# pad_3 => constant_pad_nd_3
triton_poi_fused_constant_pad_nd_mul_14 = async_compile.triton('triton_poi_fused_constant_pad_nd_mul_14', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_mul_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_constant_pad_nd_mul_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 61440000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 19200) % 25
    x1 = (xindex // 768) % 25
    x3 = (xindex // 480000)
    x4 = xindex % 19200
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 24, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + (18432*x2) + (442368*x3)), tmp5, other=0.0)
    tmp7 = 1.7015043497085571
    tmp8 = tmp6 * tmp7
    tmp9 = tl.where(tmp5, tmp8, 0.0)
    tl.store(out_ptr0 + (x5), tmp9, None)
''')

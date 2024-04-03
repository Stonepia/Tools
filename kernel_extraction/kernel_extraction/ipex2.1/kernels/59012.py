

# Original file: ./DALLE2_pytorch__42_inference_82.22/DALLE2_pytorch__42_inference_82.22.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/eu/ceuy5kaavljklkorlbol5khxj2cxe7uetgq2yp5kvocvtuzpxbpu.py
# Source Nodes: [cos, mul_1, sin], Original ATen: [aten.cos, aten.mul, aten.sin]
# cos => cos
# mul_1 => mul_2
# sin => sin
triton_poi_fused_cos_mul_sin_0 = async_compile.triton('triton_poi_fused_cos_mul_sin_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[64], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cos_mul_sin_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cos_mul_sin_0(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tmp1.to(tl.float32)
    tmp3 = x0
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = 0.0
    tmp8 = tmp6 + tmp7
    tmp9 = -0.14619587892025687
    tmp10 = tmp8 * tmp9
    tmp11 = tl.exp(tmp10)
    tmp12 = tmp2 * tmp11
    tmp13 = tl.sin(tmp12)
    tmp14 = tl.cos(tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')

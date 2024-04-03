

# Original file: ./sam___60.0/sam___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/74/c74hyujgs7g64ppxx3xvocbbwmiwnxspc52ehmr2jkeouca76vxb.py
# Source Nodes: [cos, mul_163, sin], Original ATen: [aten.cos, aten.mul, aten.sin]
# cos => cos
# mul_163 => mul_387
# sin => sin
triton_poi_fused_cos_mul_sin_40 = async_compile.triton('triton_poi_fused_cos_mul_sin_40', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cos_mul_sin_40', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cos_mul_sin_40(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = 6.283185307179586
    tmp2 = tmp0 * tmp1
    tmp3 = tl.sin(tmp2)
    tmp4 = tl.cos(tmp2)
    tl.store(out_ptr0 + (x0 + (256*x1)), tmp3, None)
    tl.store(out_ptr1 + (x0 + (256*x1)), tmp4, None)
''')


# Original file: ./crossvit_9_240___60.0/crossvit_9_240___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/dw/cdwzq4nx5ge7biuqcu4ienx7qpe7jpz36e5qusuk3qbiwevnt6yr.py
# Source Nodes: [interpolate], Original ATen: [aten.add, aten.floor, aten.mul, aten.rsub, aten.sub]
# interpolate => add_1, add_10, floor, mul, mul_10, mul_8, mul_9, sub, sub_1, sub_10, sub_9
triton_poi_fused_add_floor_mul_rsub_sub_12 = async_compile.triton('triton_poi_fused_add_floor_mul_rsub_sub_12', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[256], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_floor_mul_rsub_sub_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_floor_mul_rsub_sub_12(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0714285714285714
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = libdevice.floor(tmp6)
    tmp8 = tmp6 - tmp7
    tmp9 = 1.0
    tmp10 = tmp9 - tmp8
    tmp11 = 1.25
    tmp12 = tmp10 * tmp11
    tmp13 = 2.25
    tmp14 = tmp12 - tmp13
    tmp15 = tmp14 * tmp10
    tmp16 = tmp15 * tmp10
    tmp17 = tmp16 + tmp9
    tl.store(out_ptr0 + (x0), tmp17, xmask)
''')

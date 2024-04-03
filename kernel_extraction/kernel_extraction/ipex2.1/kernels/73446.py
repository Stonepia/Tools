

# Original file: ./DALLE2_pytorch__31_inference_71.11/DALLE2_pytorch__31_inference_71.11.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/hx/chx6hrjxeeyfu4cxb44cqfrdo64q4lt5p7a5iblim6m2f6epmtdx.py
# Source Nodes: [add_2, mul_10, mul_7, mul_8, mul_9], Original ATen: [aten.add, aten.mul]
# add_2 => add_2
# mul_10 => mul_10
# mul_7 => mul_7
# mul_8 => mul_8
# mul_9 => mul_9
triton_poi_fused_add_mul_8 = async_compile.triton('triton_poi_fused_add_mul_8', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_mul_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x3 = (xindex // 32)
    x4 = xindex % 8320
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*x3)), xmask)
    tmp1 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x5), xmask)
    tmp6 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7 * tmp3
    tmp9 = tmp4 + tmp8
    tl.store(out_ptr0 + (x0 + (64*x3)), tmp9, xmask)
''')

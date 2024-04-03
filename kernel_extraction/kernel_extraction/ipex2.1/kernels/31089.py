

# Original file: ./vision_maskrcnn__39_inference_79.19/vision_maskrcnn__39_inference_79.19_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/i5/ci5gsml5bkfvyxc4k4waezeakuuzjlwof6lm5xug5kwfcokuti37.py
# Source Nodes: [stack], Original ATen: [aten.stack]
# stack => cat
triton_poi_fused_stack_1 = async_compile.triton('triton_poi_fused_stack_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_stack_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 91000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 91)
    tmp0 = tl.load(in_ptr0 + (1 + (4*x2)), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (3 + (4*x1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (1 + (4*x1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (3 + (4*x2)), xmask).to(tl.float32)
    tmp1 = 10.0
    tmp2 = tmp0 / tmp1
    tmp5 = tmp3 - tmp4
    tmp6 = tmp2 * tmp5
    tmp7 = 0.5
    tmp8 = tmp5 * tmp7
    tmp9 = tmp4 + tmp8
    tmp10 = tmp6 + tmp9
    tmp12 = 5.0
    tmp13 = tmp11 / tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = 4.135166556742356
    tmp16 = triton_helpers.minimum(tmp14, tmp15)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.exp(tmp17)
    tmp19 = tmp18 * tmp5
    tmp20 = tmp7 * tmp19
    tmp21 = tmp10 - tmp20
    tmp22 = tmp10 + tmp20
    tl.store(out_ptr0 + (4*x2), tmp21, xmask)
    tl.store(out_ptr1 + (4*x2), tmp22, xmask)
''')

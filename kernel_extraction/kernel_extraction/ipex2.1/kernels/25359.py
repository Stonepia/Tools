

# Original file: ./sam___60.0/sam___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/wd/cwddip4emh7ocxvq22am3xwxqd6blddh6hs5edd7yzizmctfyets.py
# Source Nodes: [contiguous_3], Original ATen: [aten.clone]
# contiguous_3 => clone_10
triton_poi_fused_clone_12 = async_compile.triton('triton_poi_fused_clone_12', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6272000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x6 = (xindex // 89600)
    x7 = (xindex // 1280) % 70
    x5 = xindex % 89600
    x2 = (xindex // 17920) % 5
    x3 = (xindex // 89600) % 14
    x4 = (xindex // 1254400)
    x8 = xindex % 17920
    x0 = xindex % 1280
    tmp0 = x6
    tmp1 = tl.full([1], 64, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x7
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x5 + (81920*x6)), tmp5 & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x5 + (81920*x6)), tmp5 & xmask, other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.load(in_ptr2 + (x8 + (17920*x3) + (250880*x2) + (1254400*x4)), tmp5 & xmask, other=0.0)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.load(in_ptr3 + (x5 + (81920*x6)), tmp5 & xmask, other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tl.load(in_ptr4 + (x7 + (64*x6)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 - tmp13
    tmp15 = tl.load(in_ptr5 + (x7 + (64*x6)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = 1280.0
    tmp17 = tmp15 / tmp16
    tmp18 = 1e-06
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tl.load(in_ptr6 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr7 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.where(tmp5, tmp25, 0.0)
    tl.store(out_ptr0 + (x8 + (17920*x3) + (250880*x2) + (1254400*x4)), tmp26, xmask)
''')
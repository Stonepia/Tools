

# Original file: ./sam___60.0/sam___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/2i/c2iep3gd66ins3tpjjahihn6dyela4bbhstrsysjcx36oq4tixdv.py
# Source Nodes: [contiguous], Original ATen: [aten.clone]
# contiguous => clone_2
triton_poi_fused_clone_2 = async_compile.triton('triton_poi_fused_clone_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6272000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x6 = (xindex // 89600)
    x7 = (xindex // 1280) % 70
    x5 = xindex % 89600
    x0 = xindex % 1280
    x2 = (xindex // 17920) % 5
    x3 = (xindex // 89600) % 14
    x4 = (xindex // 1254400)
    x8 = xindex % 17920
    tmp0 = x6
    tmp1 = tl.full([1], 64, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x7
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x5 + (81920*x6)), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (x5 + (81920*x6)), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.load(in_ptr2 + (x7 + (64*x6)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + (x7 + (64*x6)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 1280.0
    tmp14 = tmp12 / tmp13
    tmp15 = 1e-06
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tmp18 = tmp11 * tmp17
    tmp19 = tl.load(in_ptr4 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 * tmp20
    tmp22 = tl.load(in_ptr5 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 + tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tl.where(tmp5, tmp25, 0.0)
    tl.store(out_ptr0 + (x8 + (17920*x3) + (250880*x2) + (1254400*x4)), tmp26, xmask)
''')

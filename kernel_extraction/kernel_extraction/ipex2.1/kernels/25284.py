

# Original file: ./sam___60.0/sam___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/sv/csvvl2sqzvfvldfunmbqtbfqxrnzvmjhqklhyxb466wdgyri6o7p.py
# Source Nodes: [l__self___image_encoder_blocks_0_attn_qkv], Original ATen: [aten._to_copy]
# l__self___image_encoder_blocks_0_attn_qkv => convert_element_type_3
triton_poi_fused__to_copy_2 = async_compile.triton('triton_poi_fused__to_copy_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*bf16', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6272000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 17920) % 25
    x3 = (xindex // 448000)
    x1 = (xindex // 1280) % 14
    x4 = xindex % 17920
    x0 = xindex % 1280
    tmp0 = x3 + (14*(x2 // 5))
    tmp1 = tl.full([1], 64, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1 + (14*(x2 % 5))
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + (17920*(x2 % 5)) + (81920*x3) + (1146880*(x2 // 5))), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (x4 + (17920*(x2 % 5)) + (81920*x3) + (1146880*(x2 // 5))), tmp5 & xmask, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr2 + (x1 + (14*(x2 % 5)) + (64*x3) + (896*(x2 // 5))), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + (x1 + (14*(x2 % 5)) + (64*x3) + (896*(x2 // 5))), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 1280.0
    tmp14 = tmp12 / tmp13
    tmp15 = 1e-06
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tmp18 = tmp11 * tmp17
    tmp19 = tl.load(in_ptr4 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 * tmp19
    tmp21 = tl.load(in_ptr5 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.where(tmp5, tmp22, 0.0)
    tmp24 = tmp23.to(tl.float32)
    tl.store(out_ptr0 + (x4 + (17920*x3) + (250880*x2)), tmp24, xmask)
''')

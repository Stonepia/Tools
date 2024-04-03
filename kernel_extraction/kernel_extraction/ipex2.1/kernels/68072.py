

# Original file: ./tf_mixnet_l___60.0/tf_mixnet_l___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/oy/coyfihofdjf6xsisyhfko5vsp4kzmghrkrd557lnxzmjzdvcdl6n.py
# Source Nodes: [pad_3], Original ATen: [aten.constant_pad_nd]
# pad_3 => constant_pad_nd_3
triton_poi_fused_constant_pad_nd_4 = async_compile.triton('triton_poi_fused_constant_pad_nd_4', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[134217728], filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*bf16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_constant_pad_nd_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112140288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 7488) % 117
    x1 = (xindex // 64) % 117
    x0 = xindex % 64
    x3 = (xindex // 876096)
    x7 = xindex
    tmp0 = (-2) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-43264) + x0 + (192*x1) + (21504*x2) + (2408448*x3)), tmp10, other=0.0).to(tl.float32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tl.load(in_ptr1 + (128 + x0), tmp10, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 - tmp13
    tmp15 = tl.load(in_ptr2 + (128 + x0), tmp10, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 * tmp15
    tmp17 = tl.load(in_ptr3 + (128 + x0), tmp10, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (128 + x0), tmp10, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = triton_helpers.maximum(0, tmp21)
    tmp23 = tl.where(tmp10, tmp22, 0.0)
    tl.store(out_ptr0 + (x7), tmp23, None)
''')

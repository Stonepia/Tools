

# Original file: ./tf_mixnet_l___60.0/tf_mixnet_l___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/a3/ca3utush77x6a427e3qgcgkg4ehajqz5elyeh2c4d2elppkhbzaz.py
# Source Nodes: [pad_1], Original ATen: [aten.constant_pad_nd]
# pad_1 => constant_pad_nd_1
triton_poi_fused_constant_pad_nd_2 = async_compile.triton('triton_poi_fused_constant_pad_nd_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[134217728], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_constant_pad_nd_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 104603648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 7232) % 113
    x1 = (xindex // 64) % 113
    x0 = xindex % 64
    x3 = (xindex // 817216)
    x4 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 112, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (192*x1) + (21504*x2) + (2408448*x3)), tmp5, other=0.0).to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (x0), tmp5, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 - tmp8
    tmp10 = tl.load(in_ptr2 + (x0), tmp5, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 * tmp10
    tmp12 = tl.load(in_ptr3 + (x0), tmp5, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.load(in_ptr4 + (x0), tmp5, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = triton_helpers.maximum(0, tmp16)
    tmp18 = tl.where(tmp5, tmp17, 0.0)
    tl.store(out_ptr0 + (x4), tmp18, None)
''')

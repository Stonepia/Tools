

# Original file: ./AllenaiLongformerBase__22_backward_143.5/AllenaiLongformerBase__22_backward_143.5.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/hk/chkx6tnvwmghgrl4gv6vk3tv5oazkw55ecjahb5pu2d4qxqs3j37.py
# Source Nodes: [], Original ATen: [aten.copy, aten.slice_scatter, aten.zeros_like]

triton_poi_fused_copy_slice_scatter_zeros_like_16 = async_compile.triton('triton_poi_fused_copy_slice_scatter_zeros_like_16', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_scatter_zeros_like_16', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_slice_scatter_zeros_like_16(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6279120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 513
    x2 = (xindex // 24624)
    x4 = xindex
    tmp17 = tl.load(in_ptr0 + (24624 + x4), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, 0.0)
    tmp8 = 1 + x2
    tmp9 = tmp8 < tmp3
    tmp10 = tl.full([1], 257, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = tmp11 & tmp9
    tmp13 = tl.where(tmp12, tmp6, 0.0)
    tmp14 = tl.load(in_ptr0 + (24624 + x4), tmp9 & xmask, other=0.0)
    tmp15 = tl.where(tmp11, tmp13, tmp14)
    tmp16 = tl.where(tmp9, tmp15, 0.0)
    tmp18 = tl.where(tmp9, tmp16, tmp17)
    tmp19 = tl.load(in_ptr1 + (257 + x0 + (257*x2)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = (tmp19 != 0)
    tmp21 = tl.load(in_ptr0 + (24624 + x4), tmp12 & xmask, other=0.0)
    tmp22 = tl.where(tmp20, tmp6, tmp21)
    tmp23 = tl.where(tmp12, tmp22, 0.0)
    tmp24 = tl.where(tmp11, tmp23, tmp6)
    tmp25 = tl.where(tmp9, tmp24, 0.0)
    tmp26 = tl.where(tmp9, tmp25, tmp6)
    tmp27 = tmp18 + tmp26
    tmp28 = tl.where(tmp5, tmp7, tmp27)
    tl.store(out_ptr0 + (x4), tmp28, xmask)
''')

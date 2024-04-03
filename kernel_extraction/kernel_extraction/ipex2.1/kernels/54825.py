

# Original file: ./AllenaiLongformerBase__22_forward_137.2/AllenaiLongformerBase__22_forward_137.2_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/th/cthiejk545msdaafey3r73tsrvd23msbrcjij5nauz4aospemcl6.py
# Source Nodes: [bool_4, full_like_2, setitem_11, where_3], Original ATen: [aten._to_copy, aten.copy, aten.full_like, aten.slice_scatter, aten.where]
# bool_4 => convert_element_type_13
# full_like_2 => full_default_7
# setitem_11 => copy_11, slice_scatter_40
# where_3 => where_6
triton_poi_fused__to_copy_copy_full_like_slice_scatter_where_20 = async_compile.triton('triton_poi_fused__to_copy_copy_full_like_slice_scatter_where_20', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_copy_full_like_slice_scatter_where_20', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_copy_full_like_slice_scatter_where_20(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 525312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 513
    x1 = (xindex // 513) % 256
    x2 = (xindex // 131328)
    x3 = xindex % 131328
    x4 = xindex
    tmp33 = tl.load(in_ptr2 + (393984 + x3 + (525312*x2)), xmask).to(tl.float32)
    tmp0 = x0
    tmp1 = tl.full([1], 256, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-256) + x0 + (257*x1)), tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp4 = (tmp3 != 0)
    tmp5 = 768 + x1
    tmp6 = tmp5 < tmp1
    tmp7 = tmp6 & tmp2
    tmp8 = tl.full([1], 257, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr1 + (197376 + x0 + (257*x1)), tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp12 = (tmp11 != 0)
    tmp13 = tl.load(in_ptr2 + (393984 + x3 + (525312*x2)), tmp10 & xmask, other=0.0).to(tl.float32)
    tmp14 = float("-inf")
    tmp15 = tl.where(tmp12, tmp14, tmp13)
    tmp16 = tl.where(tmp10, tmp15, 0.0)
    tmp17 = tl.load(in_ptr2 + (393984 + x3 + (525312*x2)), tmp7 & xmask, other=0.0).to(tl.float32)
    tmp18 = tl.where(tmp9, tmp16, tmp17)
    tmp19 = tl.where(tmp7, tmp18, 0.0)
    tmp20 = tl.load(in_ptr2 + (393984 + x3 + (525312*x2)), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp21 = tl.where(tmp6, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp14, tmp21)
    tmp23 = tl.where(tmp2, tmp22, 0.0)
    tmp24 = tmp9 & tmp6
    tmp25 = tl.load(in_ptr1 + (197376 + x0 + (257*x1)), tmp24 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = (tmp25 != 0)
    tmp27 = tl.load(in_ptr2 + (393984 + x3 + (525312*x2)), tmp24 & xmask, other=0.0).to(tl.float32)
    tmp28 = tl.where(tmp26, tmp14, tmp27)
    tmp29 = tl.where(tmp24, tmp28, 0.0)
    tmp30 = tl.load(in_ptr2 + (393984 + x3 + (525312*x2)), tmp6 & xmask, other=0.0).to(tl.float32)
    tmp31 = tl.where(tmp9, tmp29, tmp30)
    tmp32 = tl.where(tmp6, tmp31, 0.0)
    tmp34 = tl.where(tmp6, tmp32, tmp33)
    tmp35 = tl.where(tmp2, tmp23, tmp34)
    tl.store(out_ptr0 + (x4), tmp35, xmask)
''')

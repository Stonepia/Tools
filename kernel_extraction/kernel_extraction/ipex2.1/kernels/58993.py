

# Original file: ./AllenaiLongformerBase__22_backward_143.5/AllenaiLongformerBase__22_backward_143.5.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/xh/cxhpg5tpad4nfpbjcgsez2hmllbih6ilmfwe25y6bw4ei46o3myy.py
# Source Nodes: [], Original ATen: [aten.add, aten.copy, aten.select_scatter, aten.slice, aten.slice_scatter]

triton_poi_fused_add_copy_select_scatter_slice_slice_scatter_17 = async_compile.triton('triton_poi_fused_add_copy_select_scatter_slice_slice_scatter_17', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy_select_scatter_slice_slice_scatter_17', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_copy_select_scatter_slice_slice_scatter_17(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25214976
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = (xindex // 6303744)
    x2 = (xindex // 24624) % 256
    x4 = xindex % 6303744
    x0 = xindex % 513
    x6 = (xindex // 24624)
    x7 = xindex
    tmp19 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr1 + (x7), None)
    tmp0 = x3
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x2
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.load(in_ptr0 + ((-24624) + x4), tmp5, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.where(tmp5, tmp6, 0.0)
    tmp8 = tl.full([1], 256, tl.int64)
    tmp9 = tmp3 < tmp8
    tmp10 = x0
    tmp11 = tl.full([1], 257, tl.int64)
    tmp12 = tmp10 < tmp11
    tmp13 = tmp12 & tmp9
    tmp14 = 0.0
    tmp15 = tl.where(tmp13, tmp14, 0.0)
    tmp16 = tl.load(in_ptr1 + (x4), tmp9, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.where(tmp12, tmp15, tmp16)
    tmp18 = tl.where(tmp9, tmp17, 0.0)
    tmp20 = tl.where(tmp9, tmp18, tmp19)
    tmp21 = tl.load(in_ptr2 + (x0 + (257*x2)), tmp13, eviction_policy='evict_last', other=0.0)
    tmp22 = (tmp21 != 0)
    tmp23 = tl.load(in_ptr1 + (x4), tmp13, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.where(tmp22, tmp14, tmp23)
    tmp25 = tl.where(tmp13, tmp24, 0.0)
    tmp26 = tl.where(tmp12, tmp25, tmp14)
    tmp27 = tl.where(tmp9, tmp26, 0.0)
    tmp28 = tl.where(tmp9, tmp27, tmp14)
    tmp29 = tmp20 + tmp28
    tmp30 = tl.where(tmp5, tmp7, tmp29)
    tmp31 = x6
    tmp32 = tmp31 < tmp8
    tmp33 = tmp12 & tmp32
    tmp34 = tl.where(tmp33, tmp14, 0.0)
    tmp35 = tl.load(in_ptr1 + (x7), tmp32, other=0.0)
    tmp36 = tl.where(tmp12, tmp34, tmp35)
    tmp37 = tl.where(tmp32, tmp36, 0.0)
    tmp39 = tl.where(tmp32, tmp37, tmp38)
    tmp40 = tl.load(in_ptr2 + (x0 + (257*x6)), tmp33, eviction_policy='evict_last', other=0.0)
    tmp41 = (tmp40 != 0)
    tmp42 = tl.load(in_ptr1 + (x7), tmp33, other=0.0)
    tmp43 = tl.where(tmp41, tmp14, tmp42)
    tmp44 = tl.where(tmp33, tmp43, 0.0)
    tmp45 = tl.where(tmp12, tmp44, tmp14)
    tmp46 = tl.where(tmp32, tmp45, 0.0)
    tmp47 = tl.where(tmp32, tmp46, tmp14)
    tmp48 = tmp39 + tmp47
    tmp49 = tl.where(tmp2, tmp30, tmp48)
    tl.store(out_ptr0 + (x7), tmp49, None)
''')

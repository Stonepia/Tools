

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/xe/cxes222cifnujgty53aljb4aajubdw5oxztsdreomksrixf3webp.py
# Source Nodes: [clone_1, iadd_52, iadd_53, mul_100, mul_145, mul_99, neg_54, setitem_104, setitem_105, sub_46, sub_47, truediv_73, truediv_74], Original ATen: [aten.clone, aten.copy, aten.div, aten.mul, aten.neg, aten.select_scatter, aten.slice, aten.slice_scatter, aten.sub]
# clone_1 => clone_1
# iadd_52 => select_scatter_139, slice_761, slice_762, slice_765, slice_766, slice_scatter_73, slice_scatter_74
# iadd_53 => select_scatter_142, slice_791, slice_792, slice_scatter_77
# mul_100 => mul_103
# mul_145 => mul_148
# mul_99 => mul_102
# neg_54 => neg_54
# setitem_104 => copy_104, select_scatter_137, slice_scatter_70, slice_scatter_71, slice_scatter_72
# setitem_105 => select_scatter_141, slice_scatter_75, slice_scatter_76
# sub_46 => sub_46
# sub_47 => sub_47
# truediv_73 => div_70
# truediv_74 => div_71
triton_poi_fused_clone_copy_div_mul_neg_select_scatter_slice_slice_scatter_sub_78 = async_compile.triton('triton_poi_fused_clone_copy_div_mul_neg_select_scatter_slice_slice_scatter_sub_78', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_copy_div_mul_neg_select_scatter_slice_slice_scatter_sub_78', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_copy_div_mul_neg_select_scatter_slice_slice_scatter_sub_78(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3246048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3) % 26
    x0 = xindex % 3
    x4 = (xindex // 78)
    x3 = (xindex // 15912)
    x2 = (xindex // 78) % 204
    x5 = (xindex // 3) % 5304
    x6 = xindex
    tmp38 = tl.load(in_ptr1 + (x0 + (3*x4)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp39 = tl.load(in_ptr2 + (x0 + (3*x4)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp48 = tl.load(in_ptr4 + (x6), xmask).to(tl.float32)
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 25, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x0
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = tmp6 == tmp7
    tmp9 = tl.load(in_ptr0 + ((-1) + x1 + (24*x4)), tmp5 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tmp0 == tmp7
    tmp11 = tl.load(in_ptr1 + (x0 + (3*x4)), tmp5 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr2 + (x0 + (3*x4)), tmp5 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = x3
    tmp14 = tl.full([1], 2, tl.int64)
    tmp15 = tmp13 >= tmp14
    tmp16 = tl.full([1], 202, tl.int64)
    tmp17 = tmp13 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tmp18 & tmp5
    tmp20 = x2
    tmp21 = tmp20 >= tmp14
    tmp22 = tmp20 < tmp16
    tmp23 = tmp21 & tmp22
    tmp24 = tmp23 & tmp19
    tmp25 = tl.load(in_ptr3 + ((-10452) + x5 + (5200*x3)), tmp24 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr4 + (x6), tmp24 & xmask, other=0.0).to(tl.float32)
    tmp27 = tl.where(tmp8, tmp25, tmp26)
    tmp28 = tl.where(tmp24, tmp27, 0.0)
    tmp29 = tl.load(in_ptr4 + (x6), tmp19 & xmask, other=0.0).to(tl.float32)
    tmp30 = tl.where(tmp23, tmp28, tmp29)
    tmp31 = tl.where(tmp19, tmp30, 0.0)
    tmp32 = tl.load(in_ptr4 + (x6), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp33 = tl.where(tmp18, tmp31, tmp32)
    tmp34 = tl.where(tmp10, tmp12, tmp33)
    tmp35 = tl.where(tmp10, tmp11, tmp34)
    tmp36 = tl.where(tmp8, tmp9, tmp35)
    tmp37 = tl.where(tmp5, tmp36, 0.0)
    tmp40 = tmp23 & tmp18
    tmp41 = tl.load(in_ptr3 + ((-10452) + x5 + (5200*x3)), tmp40 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp42 = tl.load(in_ptr4 + (x6), tmp40 & xmask, other=0.0).to(tl.float32)
    tmp43 = tl.where(tmp8, tmp41, tmp42)
    tmp44 = tl.where(tmp40, tmp43, 0.0)
    tmp45 = tl.load(in_ptr4 + (x6), tmp18 & xmask, other=0.0).to(tl.float32)
    tmp46 = tl.where(tmp23, tmp44, tmp45)
    tmp47 = tl.where(tmp18, tmp46, 0.0)
    tmp49 = tl.where(tmp18, tmp47, tmp48)
    tmp50 = tl.where(tmp10, tmp39, tmp49)
    tmp51 = tl.where(tmp10, tmp38, tmp50)
    tmp52 = tl.where(tmp5, tmp37, tmp51)
    tl.store(out_ptr0 + (x6), tmp52, xmask)
''')



# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/3k/c3kj4vfglnn2gl3b7rgez54lkn62dhnou7ob7crtnptslpuv3psp.py
# Source Nodes: [abs_10, add_39, add_40, min_8, mul_100, mul_95, mul_99, neg_16, neg_17, setitem_25, setitem_27, sub_16, tanh_7, tensor, truediv_21, truediv_22], Original ATen: [aten.abs, aten.add, aten.copy, aten.div, aten.lift_fresh, aten.minimum, aten.mul, aten.neg, aten.select_scatter, aten.slice_scatter, aten.sub, aten.tanh]
# abs_10 => abs_10
# add_39 => add_46
# add_40 => add_47
# min_8 => minimum_7
# mul_100 => mul_100
# mul_95 => mul_95
# mul_99 => mul_99
# neg_16 => neg_16
# neg_17 => neg_17
# setitem_25 => slice_scatter_121
# setitem_27 => copy_27, select_scatter_16, select_scatter_17, slice_scatter_132, slice_scatter_133, slice_scatter_134
# sub_16 => sub_16
# tanh_7 => tanh_7
# tensor => full_default_1
# truediv_21 => div_21
# truediv_22 => div_22
triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_27 = async_compile.triton('triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_27', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_27', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_27(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4328064
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = (xindex // 21216)
    x3 = (xindex // 104) % 204
    x1 = (xindex // 2) % 2
    x0 = xindex % 2
    x5 = (xindex // 4) % 5304
    x6 = xindex
    tmp32 = tl.load(in_ptr2 + (x6), xmask)
    tmp0 = x4
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x3
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tmp6 < tmp3
    tmp10 = tmp8 & tmp9
    tmp11 = tmp10 & tmp5
    tmp12 = x1
    tmp13 = tl.full([1], 1, tl.int32)
    tmp14 = tmp12 == tmp13
    tmp15 = tl.load(in_ptr0 + ((-20956) + x0 + (2*x5) + (10452*x4)), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp5 & tmp11
    tmp17 = tl.load(in_ptr1 + ((-42432) + x6), tmp16 & xmask, other=0.0)
    tmp18 = tl.where(tmp16, tmp17, 0.0)
    tmp19 = tl.load(in_ptr2 + (x6), tmp11 & xmask, other=0.0)
    tmp20 = tl.where(tmp5, tmp18, tmp19)
    tmp21 = tl.where(tmp14, tmp15, tmp20)
    tmp22 = tl.where(tmp11, tmp21, 0.0)
    tmp23 = tmp5 & tmp5
    tmp24 = tl.load(in_ptr1 + ((-42432) + x6), tmp23 & xmask, other=0.0)
    tmp25 = tl.where(tmp23, tmp24, 0.0)
    tmp26 = tl.load(in_ptr2 + (x6), tmp5 & xmask, other=0.0)
    tmp27 = tl.where(tmp5, tmp25, tmp26)
    tmp28 = tl.where(tmp10, tmp22, tmp27)
    tmp29 = tl.where(tmp5, tmp28, 0.0)
    tmp30 = tl.load(in_ptr1 + ((-42432) + x6), tmp5 & xmask, other=0.0)
    tmp31 = tl.where(tmp5, tmp30, 0.0)
    tmp33 = tl.where(tmp5, tmp31, tmp32)
    tmp34 = tl.where(tmp5, tmp29, tmp33)
    tl.store(out_ptr0 + (x6), tmp34, xmask)
''')

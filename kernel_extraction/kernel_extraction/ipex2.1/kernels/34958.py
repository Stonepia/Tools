

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/xq/cxqfmuh2ceayvb3mylrsywjxgekmtop4ev4tcesy44c5itpnqzif.py
# Source Nodes: [abs_8, add_31, add_32, iadd_5, max_6, min_6, mul_66, mul_75, mul_77, mul_78, neg_12, neg_13, sub_14, tanh_5, tensor, tensor_1, truediv_17, truediv_18], Original ATen: [aten.abs, aten.add, aten.div, aten.lift_fresh, aten.maximum, aten.minimum, aten.mul, aten.neg, aten.sub, aten.tanh]
# abs_8 => abs_8
# add_31 => add_36
# add_32 => add_37
# iadd_5 => add_38
# max_6 => maximum_5
# min_6 => minimum_5
# mul_66 => mul_66
# mul_75 => mul_75
# mul_77 => mul_77
# mul_78 => mul_78
# neg_12 => neg_12
# neg_13 => neg_13
# sub_14 => sub_14
# tanh_5 => tanh_5
# tensor => full_default_1
# tensor_1 => full_default_2
# truediv_17 => div_17
# truediv_18 => div_18
triton_poi_fused_abs_add_div_lift_fresh_maximum_minimum_mul_neg_sub_tanh_28 = async_compile.triton('triton_poi_fused_abs_add_div_lift_fresh_maximum_minimum_mul_neg_sub_tanh_28', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: '*fp64', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_div_lift_fresh_maximum_minimum_mul_neg_sub_tanh_28', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_div_lift_fresh_maximum_minimum_mul_neg_sub_tanh_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1005000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5025)
    x1 = (xindex // 25) % 201
    x0 = xindex % 25
    x5 = xindex
    tmp23 = tl.load(in_ptr0 + (10635 + x0 + (26*x1) + (5304*x2)), xmask)
    tmp25 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (10635 + x0 + (26*x1) + (5304*x2)), xmask)
    tmp28 = tl.load(in_ptr3 + (10635 + x0 + (26*x1) + (5304*x2)), xmask)
    tmp29 = tl.load(in_out_ptr0 + (x5), xmask)
    tmp31 = tl.load(in_ptr4 + (x5), xmask)
    tmp0 = 2 + x2
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 1 + x1
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tmp6 < tmp3
    tmp10 = tmp8 & tmp9
    tmp11 = tmp10 & tmp5
    tmp12 = 1 + x0
    tmp13 = tmp12 >= tmp7
    tmp14 = tmp13 & tmp11
    tmp15 = tl.load(in_ptr0 + (10635 + x0 + (26*x1) + (5304*x2)), tmp14 & xmask, other=0.0)
    tmp16 = tl.where(tmp14, tmp15, 0.0)
    tmp17 = tl.load(in_ptr0 + (10635 + x0 + (26*x1) + (5304*x2)), tmp11 & xmask, other=0.0)
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tl.where(tmp11, tmp18, 0.0)
    tmp20 = tl.load(in_ptr0 + (10635 + x0 + (26*x1) + (5304*x2)), tmp5 & xmask, other=0.0)
    tmp21 = tl.where(tmp10, tmp19, tmp20)
    tmp22 = tl.where(tmp5, tmp21, 0.0)
    tmp24 = tl.where(tmp5, tmp22, tmp23)
    tmp27 = tmp25 * tmp26
    tmp30 = -tmp29
    tmp32 = tl.full([1], 0.0, tl.float64)
    tmp33 = triton_helpers.minimum(tmp32, tmp31)
    tmp34 = tl.full([1], 1e-20, tl.float64)
    tmp35 = tmp33 - tmp34
    tmp36 = tmp30 / tmp35
    tmp37 = tl.abs(tmp36)
    tmp38 = -tmp37
    tmp39 = tl.full([1], 0.001, tl.float64)
    tmp40 = tmp38 + tmp39
    tmp41 = tmp40 / tmp39
    tmp42 = libdevice.tanh(tmp41)
    tmp43 = tl.full([1], 1.0, tl.float64)
    tmp44 = tmp42 + tmp43
    tmp45 = tl.full([1], 0.5, tl.float64)
    tmp46 = tmp44 * tmp45
    tmp47 = tmp28 * tmp46
    tmp48 = tl.full([1], 50.0, tl.float64)
    tmp49 = triton_helpers.maximum(tmp48, tmp47)
    tmp50 = tmp27 * tmp49
    tmp51 = tmp24 + tmp50
    tl.store(in_out_ptr0 + (x5), tmp51, xmask)
''')

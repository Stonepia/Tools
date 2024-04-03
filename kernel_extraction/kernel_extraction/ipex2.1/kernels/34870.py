

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/fx/cfxtifvxxrsvkxqexy4nvubigtips7vg4s7snrgunqbfprxrgjwe.py
# Source Nodes: [abs_18, add_65, add_66, iadd_15, min_13, mul_175, mul_176, mul_177, mul_178, mul_179, neg_32, neg_33, pow_8, sub_21, tanh_15, tensor, truediv_38, truediv_39], Original ATen: [aten.abs, aten.add, aten.div, aten.lift_fresh, aten.minimum, aten.mul, aten.neg, aten.pow, aten.sub, aten.tanh]
# abs_18 => abs_18
# add_65 => add_80
# add_66 => add_81
# iadd_15 => add_82
# min_13 => minimum_12
# mul_175 => mul_175
# mul_176 => mul_176
# mul_177 => mul_177
# mul_178 => mul_178
# mul_179 => mul_179
# neg_32 => neg_32
# neg_33 => neg_33
# pow_8 => pow_8
# sub_21 => sub_21
# tanh_15 => tanh_15
# tensor => full_default_1
# truediv_38 => div_38
# truediv_39 => div_39
triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_47 = async_compile.triton('triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_47', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_47', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_47(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5000)
    x0 = xindex % 25
    x1 = (xindex // 25) % 200
    x4 = (xindex // 25)
    x5 = xindex
    tmp27 = tl.load(in_ptr3 + (2 + x1), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr4 + (2 + x1), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr5 + (10660 + x0 + (26*x1) + (5304*x2)), xmask)
    tmp32 = tl.load(in_ptr6 + (x5), xmask)
    tmp34 = tl.load(in_out_ptr0 + (x5), xmask)
    tmp52 = tl.load(in_ptr7 + (10660 + x0 + (26*x1) + (5304*x2)), xmask)
    tmp0 = 2 + x2
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (52 + x0 + (26*x1) + (5304*x2)), tmp5 & xmask, other=0.0)
    tmp7 = tl.where(tmp5, tmp6, 0.0)
    tmp8 = 2 + x1
    tmp9 = tmp8 >= tmp1
    tmp10 = tmp8 < tmp3
    tmp11 = tmp9 & tmp10
    tmp12 = tmp11 & tmp5
    tmp13 = tl.load(in_ptr1 + (x0 + (26*x4)), tmp12 & xmask, other=0.0)
    tmp14 = tl.where(tmp12, tmp13, 0.0)
    tmp15 = tmp5 & tmp5
    tmp16 = tl.load(in_ptr2 + (52 + x0 + (26*x1) + (5304*x2)), tmp15 & xmask, other=0.0)
    tmp17 = tl.where(tmp15, tmp16, 0.0)
    tmp18 = 0.0
    tmp19 = tl.where(tmp5, tmp17, tmp18)
    tmp20 = tl.where(tmp11, tmp14, tmp19)
    tmp21 = tl.where(tmp5, tmp20, 0.0)
    tmp22 = tl.load(in_ptr2 + (52 + x0 + (26*x1) + (5304*x2)), tmp5 & xmask, other=0.0)
    tmp23 = tl.where(tmp5, tmp22, 0.0)
    tmp24 = tl.where(tmp5, tmp23, tmp18)
    tmp25 = tl.where(tmp5, tmp21, tmp24)
    tmp26 = tl.where(tmp5, tmp7, tmp25)
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = -tmp32
    tmp35 = triton_helpers.minimum(tmp18, tmp34)
    tmp36 = 1e-20
    tmp37 = tmp35 - tmp36
    tmp38 = tmp33 / tmp37
    tmp39 = tl.abs(tmp38)
    tmp40 = -tmp39
    tmp41 = 0.001
    tmp42 = tmp40 + tmp41
    tmp43 = tmp42 / tmp41
    tmp44 = libdevice.tanh(tmp43)
    tmp45 = 1.0
    tmp46 = tmp44 + tmp45
    tmp47 = 0.5
    tmp48 = tmp46 * tmp47
    tmp49 = tmp31 * tmp48
    tmp50 = tmp38 * tmp38
    tmp51 = tmp49 * tmp50
    tmp53 = tmp51 * tmp52
    tmp54 = tmp26 + tmp53
    tl.store(in_out_ptr0 + (x5), tmp54, xmask)
''')
